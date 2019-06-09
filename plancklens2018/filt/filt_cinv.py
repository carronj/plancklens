"""conjugate gradient solver CMB filtering module.

"""

from __future__ import print_function
from __future__ import absolute_import

import healpy as hp
import numpy  as np
import pickle as pk
import os

from plancklens2018.helpers import mpi
from plancklens2018 import helpers
from . import filt_simple
from plancklens2018.qcinv import opfilt_pp, opfilt_tt
from plancklens2018.qcinv import util, util_alm
from plancklens2018.qcinv import multigrid, cd_solve

class library_cinv_sepTP(filt_simple.library_sepTP):
    """Library to perform inverse variance filtering on the sim_lib library.

    Args:
        lib_dir (str):
        sim_lib:
        cinv_t:
        cinv_p:
        soltn_lib (optional):
    """

    def __init__(self, lib_dir, sim_lib, cinv_t, cinv_p, cl_weights, soltn_lib=None):
        self.cinv_t = cinv_t
        self.cinv_p = cinv_p
        super(library_cinv_sepTP, self).__init__(lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib)

        if mpi.rank == 0:
            fname_mask = os.path.join(self.lib_dir, "fmask.fits.gz")
            if not os.path.exists(fname_mask):
                fmask = self.cinv_t.get_fmask()
                assert np.all(fmask == self.cinv_p.get_fmask())
                hp.write_map(fname_mask, fmask)

        mpi.barrier()
        helpers.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

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

    def _apply_ivf_t(self, tmap, soltn=None):
        return self.cinv_t.apply_ivf(tmap, soltn=soltn)

    def _apply_ivf_p(self, pmap, soltn=None):
        return self.cinv_p.apply_ivf(pmap, soltn=soltn)

    def get_tmliklm(self, idx):
        return  hp.almxfl(self.get_sim_tlm(idx), self.cinv_t.cl['tt'])

    def get_emliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'eb')
        return  hp.almxfl(self.get_sim_elm(idx), self.cinv_t.cl['ee'])

    def get_bmliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'eb')
        return  hp.almxfl(self.get_sim_blm(idx), self.cinv_t.cl['bb'])

class cinv(object):
    def __init__(self, lib_dir, lmax):
        self.lib_dir = lib_dir
        self.lmax = lmax

    def get_tal(self, a, lmax=None):
        if lmax is None: lmax = self.lmax
        assert a.lower() in ['t', 'e', 'b'], a
        ret = np.loadtxt(os.path.join(self.lib_dir, "tal.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax +1 ]

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_ftl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "ftl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fel(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fel.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]

    def get_fbl(self, lmax=None):
        if lmax is None: lmax = self.lmax
        ret = np.loadtxt(os.path.join(self.lib_dir, "fbl.dat"))
        assert len(ret) > lmax, (len(ret), lmax)
        return ret[:lmax + 1]



class cinv_t(cinv):
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv,
                 marge_monopole=True, marge_dipole=True, marge_maps=(), pcf='default', chain_descr=None):
        """

        Args:
            ninv: inverse pixel variance map. Must be a list of paths or 1d-arrays with the right size.
                  Can contain a mask but only homogeneous noise. Use cinv_t_vmap for inhomog. variance maps.
        """
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        assert isinstance(ninv, list)
        super(cinv_t, self).__init__(lib_dir, lmax)


        self.nside = nside
        self.cl = cl
        self.transf = transf
        self.ninv = ninv
        self.marge_monopole = marge_monopole
        self.marge_dipole = marge_dipole
        self.marge_maps = marge_maps

        pcf = os.path.join(lib_dir, "dense.pk") if pcf == 'default' else '' # Dense matrices will be cached there.
        if chain_descr is None : chain_descr = \
            [[3, ["split(dense(" + pcf + "), 64, diag_cl)"], 256, 128, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [2, ["split(stage(3),  256, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]

        n_inv_filt = util.jit(opfilt_tt.alm_filter_ninv, ninv, transf[0:lmax + 1],
                        marge_monopole=marge_monopole, marge_dipole=marge_dipole, marge_maps=marge_maps)
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_tt, chain_descr, cl, n_inv_filt)
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(os.path.join(lib_dir, "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir, "ftl.dat")):
                np.savetxt(os.path.join(self.lib_dir, "ftl.dat"), self._calc_ftl())

            if not os.path.exists(os.path.join(self.lib_dir, "tal.dat")):
                np.savetxt(os.path.join(self.lib_dir, "tal.dat"),  self._calc_tal())

            if not os.path.exists(os.path.join(self.lib_dir, "fmask.fits.gz")):
                hp.write_map(os.path.join(self.lib_dir, "fmask.fits.gz"), self._calc_mask())

        mpi.barrier()
        helpers.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(helpers.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return ret

    def _calc_ftl(self):
        ninv = self.chain.n_inv_filt.n_inv
        npix = len(ninv[:])
        NlevT_uKamin = np.sqrt(4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_t::noiseT_uk_arcmin = %.3f"%NlevT_uKamin)

        s_cls = self.chain.s_cls
        b_transf = self.chain.n_inv_filt.b_transf

        if s_cls['tt'][0] == 0.: assert self.chain.n_inv_filt.marge_monopole
        if s_cls['tt'][1] == 0.: assert self.chain.n_inv_filt.marge_dipole

        ftl = helpers.cli(s_cls['tt'][0:self.lmax + 1] + (NlevT_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)
        if self.chain.n_inv_filt.marge_monopole: ftl[0] = 0.0
        if self.chain.n_inv_filt.marge_dipole: ftl[1] = 0.0

        return ftl

    def _calc_tal(self):
        return helpers.cli(self.transf)

    def _calc_mask(self):
        ninv = self.chain.n_inv_filt.n_inv
        assert hp.npix2nside(len(ninv)) == self.nside
        return np.where(ninv > 0, 1., 0.)

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'cltt': helpers.clhash(self.cl['tt'][:self.lmax + 1]),
                'transf': helpers.clhash(self.transf[:self.lmax + 1]),
                'ninv': self._ninv_hash(),
                'marge_monopole': self.marge_monopole,
                'marge_dipole': self.marge_dipole,
                'marge_maps': self.marge_maps}

    def apply_ivf(self, tmap, soltn=None):
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=np.complex)
        else:
            talm = soltn.copy()
        self.chain.solve(talm, tmap)
        return talm


class cinv_p(cinv):
    """Missing doc.

    Note:
        This implementation does not support template projection.

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, pcf='default', chain_descr=None):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        super(cinv_p, self).__init__(lib_dir, lmax)

        self.nside = nside
        self.cl = cl
        self.transf = transf
        self.ninv = ninv

        pcf = os.path.join(lib_dir, "dense.pk") if pcf == 'default' else None
        if chain_descr is None: chain_descr = \
            [[2, ["split(dense(" + pcf + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,cd_solve.cache_mem()],
             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]
        n_inv_filt = util.jit(opfilt_pp.alm_filter_ninv, ninv, transf[0:lmax + 1])
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_pp, chain_descr, cl, n_inv_filt)

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(os.path.join(lib_dir, "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir, "fbl.dat")):
                fel, fbl = self._calc_febl()
                np.savetxt(os.path.join(self.lib_dir, "fel.dat"), fel)
                np.savetxt(os.path.join(self.lib_dir, "fbl.dat"), fbl)

            if not os.path.exists(os.path.join(self.lib_dir, "tal.dat")):
                np.savetxt(os.path.join(self.lib_dir, "tal.dat"), self._calc_tal())

            if not os.path.exists(os.path.join(self.lib_dir,  "fmask.fits.gz")):
                hp.write_map(os.path.join(self.lib_dir,  "fmask.fits.gz"),  self._calc_mask())

        mpi.barrier()
        helpers.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict())

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'clee': helpers.clhash(self.cl.get('ee', np.array([0.]))),
                'cleb': helpers.clhash(self.cl.get('eb', np.array([0.]))),
                'clbb': helpers.clhash(self.cl.get('bb', np.array([0.]))),
                'transf':helpers.clhash(self.transf),
                'ninv': self._ninv_hash()}


    def apply_ivf(self, tmap, soltn=None):
        if soltn is not None:
            print("**** cinv_p::Discarding soltn in cinv_p")
        assert len(tmap) == 2

        telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        talm = util_alm.eblm([telm, tblm])

        self.chain.solve(talm, [tmap[0], tmap[1]])

        return talm.elm, talm.blm

    def _calc_febl(self):
        assert not 'eb' in self.chain.s_cls.keys()

        if len(self.chain.n_inv_filt.n_inv) == 1:
            ninv = self.chain.n_inv_filt.n_inv[0]
            npix = len(ninv)
            NlevP_uKamin = np.sqrt(
                4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        else:
            assert len(self.chain.n_inv_filt.n_inv) == 3
            ninv = self.chain.n_inv_filt.n_inv
            NlevP_uKamin= 0.5 * np.sqrt(
                4. * np.pi / len(ninv[0]) / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
            NlevP_uKamin += 0.5 * np.sqrt(
                4. * np.pi / len(ninv[2]) / np.sum(ninv[2]) * len(np.where(ninv[2] != 0.0)[0])) * 180. * 60. / np.pi


        print("cinv_p::noiseP_uk_arcmin = %.3f"%NlevP_uKamin)

        s_cls = self.chain.s_cls
        b_transf = self.chain.n_inv_filt.b_transf
        fel = 1.0 / (s_cls['ee'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)
        fbl = 1.0 / (s_cls['bb'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 / b_transf[0:self.lmax + 1] ** 2)

        fel[0:2] *= 0.0
        fbl[0:2] *= 0.0

        return fel, fbl

    def _calc_tal(self):
        return helpers.cli(self.transf)

    def _calc_mask(self):
        mask = np.ones(hp.nside2npix(self.nside), dtype=float)
        for ninv in self.chain.n_inv_filt.n_inv:
            assert hp.npix2nside(len(ninv)) == self.nside
            mask *= (ninv > 0.)
        return mask

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv[0]:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(helpers.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return [ret]