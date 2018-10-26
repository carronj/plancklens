from __future__ import print_function

import healpy as hp
import numpy  as np
import pickle as pk
import os


from plancklens2018 import mpi
from plancklens2018 import utils
from plancklens2018.filt import filt_simple
from plancklens2018.qcinv import opfilt_pp, opfilt_tt
from plancklens2018.qcinv import util, util_alm
from plancklens2018.qcinv import multigrid, cd_solve

#FIXME: hashes, qcinv missing bits, relative imports
#FIXME: paths def.

class library_cinv_sepTP(filt_simple.library_sepTP):
    """
        Library to perform inverse variance filtering on the sim_lib library.
        The lid_dir path is setup at instantiation and will contain the ivfs filtered map
        as alm fits file.
    """

    def __init__(self, lib_dir, sim_lib, cinv_t, cinv_p, soltn_lib=None):
        self.cinv_t = cinv_t
        self.cinv_p = cinv_p
        super(library_cinv_sepTP, self).__init__(lib_dir, sim_lib, soltn_lib=soltn_lib)

        if mpi.rank == 0:
            fname_mask = os.path.join(self.lib_dir, "fmask.fits.gz")
            if not os.path.exists(fname_mask):
                fmask = self.cinv_t.get_fmask()
                assert np.all(fmask == self.cinv_p.get_fmask())
                hp.write_map(fname_mask)

        mpi.barrier()
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'r')), self.hashdict())

    def hashdict(self):
        return {'cinv_t': self.cinv_t.hashdict(),
                'cinv_p': self.cinv_p.hashdict(),
                'sim_lib': self.sim_lib.hashdict()}

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_tal(self, a, lmax=None):
        assert (a.lower() in ['t', 'e', 'b']), a
        if a.lower() == 't':
            return self.cinv_t.get_tal(a, lmax=lmax)
        else:
            return self.cinv_p.get_tal(a, lmax=lmax)

    def get_ftl(self, lmax=None):
        return self.cinv_t.get_ftl(lmax=lmax)

    def get_fel(self, lmax=None):
        return self.cinv_p.get_fel(lmax=lmax)

    def get_fbl(self, lmax=None):
        return self.cinv_p.get_fbl(lmax=lmax)

    def apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)

    def get_tmliklm(self, idx):
        return  hp.almxfl(self.get_sim_tlm(idx), self.cinv_t.cl['tt'])

    def get_emliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'cleb')
        return  hp.almxfl(self.get_sim_elm(idx), self.cinv_t.cl['ee'])

    def get_bmliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'cleb')
        return  hp.almxfl(self.get_sim_blm(idx), self.cinv_t.cl['bb'])

class cinv(object):
    def __init__(self, lib_dir, lmax):
        self.lib_dir = lib_dir
        self.lmax = lmax

    def get_tal(self, a, lmax=None):
        if lmax is None: lmax = self.lmax
        assert a.lower() in ['t', 'e', 'b'], a
        ret = np.loadtxt(self.lib_dir + "/tal.dat")
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax +1 ]

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_ftl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(self.lib_dir + "/ftl.dat")
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fel(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(self.lib_dir + "/fel.dat")
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fbl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(self.lib_dir + "/fbl.dat")
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

class cinv_p_vmap(cinv):
    def __init__(self, lib_dir, lmax, nside, cl, transf, mask_list, ninv_vmaps, marge_maps=(), pcf='default'):
        assert lmax >= 1024 and nside >= 512, (lmax, nside)
        assert not isinstance(mask_list[0], list)  # Same mask for QQ QU UU here
        super(cinv_p_vmap, self).__init__(lib_dir, lmax)

        self.nside = nside
        self.cl = cl
        self.transf = transf
        self.mask_list = mask_list
        self.ninv_vmaps = ninv_vmaps
        self.marge_maps = marge_maps

        pcf = lib_dir + "/dense.pk" if pcf == 'default' else None
        chain_descr = [[2, ["split(dense(" + pcf + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
                       [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
                       [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]
        n_inv_filt = util.jit(opfilt_pp.alm_filter_ninv, self.ninv_wmask(), transf[0:lmax + 1], marge_maps=marge_maps)
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_pp, chain_descr, cl, n_inv_filt)

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/filt_hash.pk"):
                pk.dump(self.hashdict(), open(lib_dir + "/filt_hash.pk", 'w'))

            if not os.path.exists(self.lib_dir + "/fbl.dat"):
                fel, fbl = self.calc_febl()
                np.savetxt(self.lib_dir + "/fel.dat", fel)
                np.savetxt(self.lib_dir + "/fbl.dat", fbl)

            if not os.path.exists(self.lib_dir + "/tal.dat"):
                np.savetxt(self.lib_dir + "/tal.dat", self.calc_tal())

            if not os.path.exists(self.lib_dir + "/fmask.fits.gz"):
                hp.write_map(self.lib_dir + "/fmask.fits.gz", self.calc_mask())

        mpi.barrier()
        utils.hash_check(pk.load(open(lib_dir + "/filt_hash.pk", 'r')), self.hashdict())

    def ninv_wmask(self):
        if isinstance(self.ninv_vmaps[0], list):
            assert len(self.ninv_vmaps) == 3, len(self.ninv_vmaps)  # QQ QU UU ninv maps
            return [self.mask_list + vmaps for vmaps in self.ninv_vmaps]
        else:
            return [self.mask_list + self.ninv_vmaps]

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'clee': utils.clhash(self.cl.get('ee', np.array([0.]))),
                'clbb': utils.clhash(self.cl.get('bb', np.array([0.]))),
                'cleb': utils.clhash(self.cl.get('eb', np.array([0.]))),
                'transf': utils.clhash(self.transf),
                'ninv': self.ninv_hash(),
                'marge_maps': self.marge_maps}

    def calc_nlevp(self):
        if len(self.chain.n_inv_filt.n_inv) == 1:
            ninv = self.chain.n_inv_filt.n_inv[0]
            npix = len(ninv)
            noiseP_uK_arcmin = np.sqrt(
                4. * np.pi / npix / np.sum(ninv[:]) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        else:
            assert len(self.chain.n_inv_filt.n_inv) == 3
            ninv = self.chain.n_inv_filt.n_inv
            noiseP_uK_arcmin = 0.5 * np.sqrt(
            4. * np.pi / len(ninv[0]) / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
            noiseP_uK_arcmin += 0.5 * np.sqrt(
            4. * np.pi / len(ninv[2]) / np.sum(ninv[2]) * len(np.where(ninv[2] != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_p_vmap::calc_febl. noiseP_uk_arcmin = %.2f"%noiseP_uK_arcmin)
        return noiseP_uK_arcmin

    def calc_febl(self):
        assert 'eb' not in self.chain.s_cls.keys()
        if len(self.chain.n_inv_filt.n_inv) == 1:
            ninv = self.chain.n_inv_filt.n_inv[0]
            npix = len(ninv)
            noiseP_uK_arcmin = np.sqrt(
                4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        else:
            assert len(self.chain.n_inv_filt.n_inv) == 3
            ninv = self.chain.n_inv_filt.n_inv
            noiseP_uK_arcmin = 0.5 * np.sqrt(
                4. * np.pi / len(ninv[0]) / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
            noiseP_uK_arcmin += 0.5 * np.sqrt(
                4. * np.pi / len(ninv[2]) / np.sum(ninv[2]) * len(np.where(ninv[2] != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_p_vmap::calc_febl. noiseP_uk_arcmin = %.2f"%noiseP_uK_arcmin)

        s_cls = self.chain.s_cls
        b_transf = self.chain.n_inv_filt.b_transf[:self.lmax + 1]

        fel = 1.0 / (s_cls['ee'][:self.lmax + 1] + (noiseP_uK_arcmin * np.pi / 180. / 60.) ** 2 / b_transf ** 2)
        fel[:2] = 0.0

        fbl = 1.0 / (s_cls['bb'][:self.lmax + 1] + (noiseP_uK_arcmin * np.pi / 180. / 60.) ** 2 / b_transf ** 2)
        fbl[:2] = 0.0

        return fel, fbl

    def calc_tal(self):
        return utils.cli(self.transf)

    def calc_mask(self):
        """ returns the inverse mask map / max(inverse mask map) """
        n_inv = self.mask_list
        if isinstance(n_inv, list):
            # The following line does nothing is if the first element of n_inv is not a string.
            n_inv_prod = util.load_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * util.load_map(n)
            n_inv = n_inv_prod
        else:
            n_inv = util.load_map(n_inv)
            # This changes the attributes of qcinv to the full mask.
        return n_inv / np.max(n_inv)

    def apply_ivf(self, tmap, soltn=None):
        assert len(tmap) == 2, (len(tmap))
        assert soltn is None, 'not implemented'
        telm = np.zeros(util_alm.lmax2nlm(self.lmax), dtype=np.complex)
        tblm = np.zeros(util_alm.lmax2nlm(self.lmax), dtype=np.complex)
        talm = util_alm.eblm([telm, tblm])

        self.chain.solve(talm, [tmap[0], tmap[1]])

        relm = util_alm.alm2rlm(talm.elm)
        rblm = util_alm.alm2rlm(talm.blm)
        del talm, telm, tblm

        return relm, rblm

    def ninv_hash(self):
        ret = []
        if isinstance(self.ninv_vmaps[0], list):
            assert len(self.ninv_vmaps) == 3, len(self.ninv_vmaps)
            for i in range(3):
                for ninv_comp in self.mask_list + self.ninv_vmaps[i]:
                    if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                        # in order not to store full maps as hash.
                        ret.append(utils.clhash(ninv_comp))
                    else:
                        ret.append(ninv_comp)
            return [ret]
        else:
            for ninv_comp in self.mask_list + self.ninv_vmaps:
                if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                    # in order not to store full maps as hash.
                    ret.append(utils.clhash(ninv_comp))
                else:
                    ret.append(ninv_comp)
            return [ret]