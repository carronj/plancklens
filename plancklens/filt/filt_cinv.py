"""conjugate gradient solver CMB filtering module.

"""

from __future__ import print_function
from __future__ import absolute_import

import healpy as hp
import numpy  as np
import pickle as pk
import os

from plancklens.helpers import mpi
from plancklens import utils
from plancklens.filt import filt_simple
from plancklens.qcinv import opfilt_pp, opfilt_tt, opfilt_tp
from plancklens.qcinv import util, util_alm
from plancklens.qcinv import multigrid, cd_solve



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
    r"""Temperature-only inverse-variance (or Wiener-)filtering instance.

        Args:
            lib_dir: mask and other things will be cached there
            lmax: filtered alm's are reconstructed up to lmax
            nside: healpy resolution of maps to filter
            cl: fiducial CMB spectra used to filter the data (dict with 'tt' key)
            transf: CMB maps transfer function (array)
            ninv: inverse pixel variance map. Must be a list of paths or of healpy maps with consistent nside.
            rescal_cl: isotropic rescaling of the map prior the cg inversion. This just makes the convergence criterium change a bit

        Note:

            The only difference of the original plancklens filter is the rescaling of the maps. In effect, the modes of :math'`D_\ell` rather than :math'`C_\ell` are reconstructed
            This changes nothing to the iterations, but gives the dot product testing for convergence more sensitvity to relevant scales

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, rescal_cl='default',
                 marge_monopole=True, marge_dipole=True, marge_maps=(), pcf='default', chain_descr=None):

        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        assert isinstance(ninv, list)
        super(cinv_t, self).__init__(lib_dir, lmax)


        if rescal_cl in ['default', None]:
            default_rescal = True
            rescal_cl = np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float) / 2. / np.pi)
        else:
            default_rescal = False
            
            # otherwise will throw index error if rescal_cl is too small 
            assert len(rescal_cl) >= lmax + 1, [rescal_cl.shape, lmax]
            
        dl = {k: rescal_cl[:lmax + 1] ** 2 * cl[k][:lmax + 1] for k in cl.keys()}  # rescaled cls (Dls by default)
        transf_dl = transf[:lmax + 1] * utils.cli(rescal_cl)

        self.nside = nside

        self.cl = cl
        self.dl = dl

        self.transf = transf[:lmax + 1]
        self.rescaled_transf =transf_dl
        self.rescal_cl = rescal_cl
        self.default_rescal = default_rescal  # track default behaviour for hash key

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

        n_inv_filt = util.jit(opfilt_tt.alm_filter_ninv, ninv, transf_dl,
                        marge_monopole=marge_monopole, marge_dipole=marge_dipole, marge_maps=marge_maps)
        self.chain_descr = chain_descr
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_tt, self.chain_descr, dl, n_inv_filt)
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
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict(), fn=os.path.join(lib_dir, "filt_hash.pk"))

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
                # Get only filename (useful for runs on different scratch systems of NERSC)
                # ret.append(os.path.basename(ninv_comp))
        return ret

    def _calc_ftl(self):
        ninv = self.chain.n_inv_filt.n_inv
        npix = len(ninv[:])
        NlevT_uKamin = np.sqrt(4. * np.pi / npix / np.sum(ninv) * len(np.where(ninv != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_t::noiseT_uk_arcmin = %.3f"%NlevT_uKamin)

        s_cls = self.cl
        b_transf = self.transf

        if s_cls['tt'][0] == 0.: assert self.chain.n_inv_filt.marge_monopole
        if s_cls['tt'][1] == 0.: assert self.chain.n_inv_filt.marge_dipole

        ftl = utils.cli(s_cls['tt'][0:self.lmax + 1] + (NlevT_uKamin * np.pi / 180. / 60.) ** 2 *utils.cli(b_transf[0:self.lmax + 1] ** 2))
        if self.chain.n_inv_filt.marge_monopole: ftl[0] = 0.0
        if self.chain.n_inv_filt.marge_dipole: ftl[1] = 0.0

        return ftl

    def _calc_tal(self):
        return utils.cli(self.transf)

    def _calc_mask(self):
        ninv = self.chain.n_inv_filt.n_inv
        assert hp.npix2nside(len(ninv)) == self.nside
        return np.where(ninv > 0, 1., 0.)

    def hashdict(self):
        hd = {'lmax': self.lmax,
                'nside': self.nside,
                'cltt': utils.clhash(self.cl['tt'][:self.lmax + 1]),
                'transf': utils.clhash(self.transf[:self.lmax + 1]),
                'ninv': self._ninv_hash(),
                'marge_monopole': self.marge_monopole,
                'marge_dipole': self.marge_dipole,
                'marge_maps': self.marge_maps}
        
        # Don't hash rescaling unless deviates from default behaviour (Dls) ? Otherwise will produce key error in hash check for 
        # maps filtered before this change.
        # Alternatively: Could automatically update hash pk if missing 'rescal_cl' key? 
        if self.default_rescal is False:
            hd['rescal_cl'] = utils.clhash(self.rescal_cl)

        return hd
            
    def apply_ivf(self, tmap, soltn=None):
        if soltn is None:
            talm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        else:
            talm = soltn.copy()
        self.chain.solve(talm, tmap)
        hp.almxfl(talm, self.rescal_cl, inplace=True)
        return talm


class cinv_p(cinv):
    r"""Polarization-only inverse-variance (or Wiener-)filtering instance.

        Args:
            lib_dir: mask and other things will be cached there
            lmax: filtered alm's are reconstructed up to lmax
            nside: healpy resolution of maps to filter
            cl: fiducial CMB spectra used to filter the data (dict with 'tt' key)
            transf: CMB E-mode polarization transfer function (array)
            ninv: inverse pixel variance maps. Must be a list of either 3 (QQ, QU, UU) or 1 (QQ = UU noise) elements.
                  These element are themselves list of paths or of healpy maps with consistent nside.
            transf_blm(optional): B-polarization transfer function (if different from E-mode one)

        Note:
            This implementation now supports template projection

    """
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv, pcf='default',
                 chain_descr=None, transf_blm=None, marge_qmaps=(), marge_umaps=()):
        assert lib_dir is not None and lmax >= 1024 and nside >= 512, (lib_dir, lmax, nside)
        super(cinv_p, self).__init__(lib_dir, lmax)

        self.nside = nside
        self.cl = cl
        self.transf_e = transf
        self.transf_b = transf if transf_blm is None else transf_blm
        self.transf = transf if transf_blm is None else 0.5 * self.transf_e + 0.5 * self.transf_b
        self.ninv = ninv

        pcf = os.path.join(lib_dir, "dense.pk") if pcf == 'default' else None
        if chain_descr is None: chain_descr = \
            [[2, ["split(dense(" + pcf + "), 32, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,cd_solve.cache_mem()],
             [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
             [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]
        n_inv_filt = util.jit(opfilt_pp.alm_filter_ninv, ninv, transf[0:lmax + 1],
                              b_transf_b=transf_blm, marge_umaps=marge_umaps, marge_qmaps=marge_qmaps)
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
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict(), fn=os.path.join(lib_dir, "filt_hash.pk"))

    def hashdict(self):
        return {'lmax': self.lmax,
                'nside': self.nside,
                'clee': utils.clhash(self.cl.get('ee', np.array([0.]))),
                'cleb': utils.clhash(self.cl.get('eb', np.array([0.]))),
                'clbb': utils.clhash(self.cl.get('bb', np.array([0.]))),
                'transf':utils.clhash(self.transf),
                'ninv': self._ninv_hash()}


    def apply_ivf(self, tmap, soltn=None):
        if soltn is not None:
            assert len(soltn) == 2
            assert hp.Alm.getlmax(soltn[0].size) == self.lmax, (hp.Alm.getlmax(soltn[0].size), self.lmax)
            assert hp.Alm.getlmax(soltn[1].size) == self.lmax, (hp.Alm.getlmax(soltn[1].size), self.lmax)
            talm = util_alm.eblm([soltn[0], soltn[1]])
        else:
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            talm = util_alm.eblm([telm, tblm])

        assert len(tmap) == 2
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
        b_transf_e = self.chain.n_inv_filt.b_transf_e
        b_transf_b = self.chain.n_inv_filt.b_transf_b
        fel = utils.cli(s_cls['ee'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 * utils.cli(b_transf_e[0:self.lmax + 1] ** 2))
        fbl = utils.cli(s_cls['bb'][:self.lmax + 1] + (NlevP_uKamin * np.pi / 180. / 60.) ** 2 * utils.cli(b_transf_b[0:self.lmax + 1] ** 2))

        fel[0:2] *= 0.0
        fbl[0:2] *= 0.0

        return fel, fbl

    def _calc_tal(self):
        return utils.cli(self.transf)

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
                ret.append(utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return [ret]


class cinv_tp:
    def __init__(self, lib_dir, lmax, nside, cl, transf, ninv,
                 marge_maps_t=(), marge_monopole=False, marge_dipole=False,
                 pcf='default', rescal_cl='default', chain_descr=None, transf_p=None):
        """Instance for joint temperature-polarization filtering

            Args:
                lib_dir: a few quantities might get cached there
                lmax: CMB filtering performed up to multipole lmax
                nside: healpy resolution of the input maps
                cl: fiducial CMB spectra used to filter the data (dict with 'tt', 'te', 'ee', 'bb' keys)
                transf: CMB transfer function in temperature
                ninv: list of lists with mask paths and / or inverse pixel noise levels.
                        TT, (QQ + UU) / 2 if len(ninv) == 2 or TT, QQ, QU UU if == 4
                        e.g. [[iNevT,mask1,mask2,..],[iNevP,mask1,mask2...]]
                marge_maps_t: maps to project out in the filtering (T-part)
                marge_monopole: marginalizes out the T monopole if set
                marge_dipole: marginalizes out the T dipole if set

                chain_descr: preconditioner mulitgrid chain description (if different from default)
                transf_p: polarization transfer function (if different from temperature)


        """
        assert (lmax >= 1024)
        assert (nside >= 512)
        assert len(ninv) == 2 or len(ninv) == 4  # TT, (QQ + UU)/2 or TT,QQ,QU,UU

        if rescal_cl == 'default':
            rescal_cl = {a: np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float) / 2. / np.pi) for a in ['t', 'e', 'b']}
        elif rescal_cl is None:
            rescal_cl = {a: np.ones(lmax  + 1, dtype=float) for a in ['t', 'e', 'b']}
        elif rescal_cl == 'tonly':
            rescal_cl = {a: np.ones(lmax  + 1, dtype=float) for a in ['e', 'b']}
            rescal_cl['t'] = np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2, dtype=float) / 2. / np.pi)
        else:
            assert 0
        for k in rescal_cl.keys():
            rescal_cl[k] /= np.mean(rescal_cl[k]) # in order not mess around with the TEB relative weights of the spectra
        dl = {k: rescal_cl[k[0]] * rescal_cl[k[1]] * cl[k][:lmax + 1] for k in cl.keys()}  # rescaled cls (Dls by default)
        if transf_p is None:
            transf_p = transf
        transf_dls = {a: transf_p[:lmax + 1] * utils.cli(rescal_cl[a]) for a in ['e', 'b']}
        transf_dls['t'] = transf[:lmax + 1] * utils.cli(rescal_cl['t'])

        self.lmax = lmax
        self.nside = nside
        self.cl = cl
        self.transf_t = transf
        self.transf_p = transf_p
        self.ninv = ninv
        self.marge_maps_t = marge_maps_t
        self.marge_maps_p = []

        self.lib_dir = lib_dir
        self.rescal_cl = rescal_cl

        if chain_descr is None:
            pcf = lib_dir + "/dense_tp.pk" if pcf == 'default' else None
            chain_descr = [[3, ["split(dense(" + pcf + "), 64, diag_cl)"], 256, 128, 3, 0.0, cd_solve.tr_cg,
                            cd_solve.cache_mem()],
                           [2, ["split(stage(3),  256, diag_cl)"], 512, 256, 3, 0.0, cd_solve.tr_cg,
                            cd_solve.cache_mem()],
                           [1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg,
                            cd_solve.cache_mem()],
                           [0, ["split(stage(1), 1024, diag_cl)"], lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg,
                            cd_solve.cache_mem()]]

        n_inv_filt = util.jit(opfilt_tp.alm_filter_ninv, ninv, transf_dls['t'], b_transf_e=transf_dls['e'], b_transf_b=transf_dls['b'],
                            marge_maps_t=marge_maps_t, marge_monopole=marge_monopole, marge_dipole=marge_dipole)
        self.chain = util.jit(multigrid.multigrid_chain, opfilt_tp, chain_descr, dl, n_inv_filt)

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(os.path.join(lib_dir,  "filt_hash.pk")):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir,  "filt_hash.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(lib_dir,  "fal.pk")):
                pk.dump(self._calc_fal(), open(os.path.join(lib_dir,  "fal.pk"), 'wb'), protocol=2)

            if not os.path.exists(os.path.join(self.lib_dir,  "fmask.fits.gz")):
                fmask = self.calc_mask()
                hp.write_map(os.path.join(self.lib_dir,  "fmask.fits.gz"), fmask)

        mpi.barrier()
        utils.hash_check(pk.load(open(os.path.join(lib_dir,  "filt_hash.pk"), 'rb')), self.hashdict(), fn=os.path.join(lib_dir, "filt_hash.pk"))

    def hashdict(self):
        ret = {'lmax': self.lmax,
                'nside': self.nside,
                'rescal_cl':{k: utils.clhash(self.rescal_cl[k]) for k in self.rescal_cl.keys()},
                'cls':{k : utils.clhash(self.cl[k]) for k in self.cl.keys()},
                'transf': utils.clhash(self.transf_t),
                'ninv': self._ninv_hash(),
                'marge_maps_t': self.marge_maps_t,
                'marge_maps_p': self.marge_maps_p}
        if self.transf_p is not self.transf_t:
            ret['transf_p'] = utils.clhash(self.transf_p)
        return ret

    def get_fal(self):
        return pk.load(open(os.path.join(self.lib_dir,  "fal.pk"), 'rb'))

    def _calc_fal(self):
        """Isotropic approximation to filtering matrix

            Used e.g. for plancklens response calculations

        """
        ninv = self.chain.n_inv_filt.n_inv
        assert len(ninv) == 2, 'implement this, easy'
        assert ninv[0].size == 12 * self.nside ** 2
        assert ninv[1].size == 12 * self.nside ** 2
        npix = 12 * self.nside ** 2
        nlevt = np.sqrt(4. * np.pi / npix / np.sum(ninv[0]) * len(np.where(ninv[0] != 0.0)[0])) * 180. * 60. / np.pi
        nlevp = np.sqrt(4. * np.pi / npix / np.sum(ninv[1]) * len(np.where(ninv[1] != 0.0)[0])) * 180. * 60. / np.pi
        print("cinv_tp::noiseT_uk_arcmin = %.3f"%nlevt)
        print("cinv_tp::noiseP_uk_arcmin = %.3f"%nlevp)

        fals = np.zeros((self.lmax + 1, 3, 3), dtype=float)
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b']):
                fals[:, i, j] = self.cl.get(a + b, self.cl.get(b + a, np.zeros(self.lmax + 1)))[:self.lmax+1]
        fals[1:, 0, 0] += ( (nlevt / 180 / 60 * np.pi) / self.transf_t[1:self.lmax + 1] ) ** 2
        fals[2:, 1, 1] += ( (nlevp / 180 / 60 * np.pi) / self.transf_p[2:self.lmax + 1] ) ** 2
        fals[2:, 2, 2] += ( (nlevp / 180 / 60 * np.pi) / self.transf_p[2:self.lmax + 1] ) ** 2
        fals = np.linalg.pinv(fals)
        fals_dict = {}
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b'][i:]):
                if np.any(fals[:, i, i + j]):
                    fals_dict[a + b] = fals[:, i, i + j]
        return fals_dict

    def calc_mask(self):
        mask = np.ones(hp.nside2npix(self.nside), dtype=float)
        for ninv in self.chain.n_inv_filt.n_inv:
            assert hp.npix2nside(len(ninv)) == self.nside
            mask *= (ninv > 0.)
        return mask

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def apply_ivf(self, tqumap, soltn=None, apply_fini=''):
        assert (len(tqumap) == 3)
        if soltn is None:
            ttlm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            telm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
            tblm = np.zeros(hp.Alm.getsize(self.lmax), dtype=complex)
        else:
            ttlm, telm, tblm = soltn
            hp.almxfl(ttlm, self.rescal_cl['t'], inplace=True)
            hp.almxfl(telm, self.rescal_cl['e'], inplace=True)
            hp.almxfl(tblm, self.rescal_cl['b'], inplace=True)
        talm = opfilt_tp.teblm([ttlm, telm, tblm])
        self.chain.solve(talm, [tqumap[0], tqumap[1], tqumap[2]], apply_fini=apply_fini)
        hp.almxfl(talm.tlm, self.rescal_cl['t'], inplace=True)
        hp.almxfl(talm.elm, self.rescal_cl['e'], inplace=True)
        hp.almxfl(talm.blm, self.rescal_cl['b'], inplace=True)
        return talm.tlm, talm.elm, talm.blm

    def _ninv_hash(self):
        ret = []
        for ninv_comp in self.ninv:
            if isinstance(ninv_comp, np.ndarray) and ninv_comp.size > 1:
                ret.append(utils.clhash(ninv_comp))
            else:
                ret.append(ninv_comp)
        return [ret]


class library_cinv_sepTP(filt_simple.library_sepTP):
    """Library to perform inverse-variance filtering of a simulation library.

        Suitable for separate temperature and polarization filtering.

        Args:
            lib_dir (str): a
            sim_lib: simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
            cinvt: temperature-only filtering library
            cinvp: poalrization-only filtering library
            soltn_lib (optional): simulation libary providing starting guesses for the filtering.

    """

    def __init__(self, lib_dir, sim_lib, cinvt:cinv_t, cinvp:cinv_p, cl_weights:dict, soltn_lib=None):
        self.cinv_t = cinvt
        self.cinv_p = cinvp
        super(library_cinv_sepTP, self).__init__(lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib)

        if mpi.rank == 0:
            fname_mask = os.path.join(self.lib_dir, "fmask.fits.gz")
            if not os.path.exists(fname_mask):
                fmask = self.cinv_t.get_fmask()
                assert np.all(fmask == self.cinv_p.get_fmask())
                hp.write_map(fname_mask, fmask)

        mpi.barrier()
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict(), fn=os.path.join(lib_dir, "filt_hash.pk"))

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
        return  hp.almxfl(self.get_sim_elm(idx), self.cinv_p.cl['ee'])

    def get_bmliklm(self, idx):
        assert not hasattr(self.cinv_p.cl, 'eb')
        return  hp.almxfl(self.get_sim_blm(idx), self.cinv_p.cl['bb'])

class library_cinv_jTP(filt_simple.library_jTP):
    """Library to perform inverse-variance filtering of a simulation library.

        Suitable for separate temperature and polarization filtering.

        Args:
            lib_dir (str): a place to cache the maps
            sim_lib: simulation library instance (requires get_sim_tmap, get_sim_pmap methods)
            cinv_jtp: temperature and pol joint filtering library
            cl_weights: spectra used to build the Wiener filtered leg from the inverse-variance maps
            soltn_lib (optional): simulation libary providing starting guesses for the filtering.


    """

    def __init__(self, lib_dir:str, sim_lib, cinv_jtp:cinv_tp, cl_weights:dict, soltn_lib=None):
        self.cinv_tp = cinv_jtp
        super(library_cinv_jTP, self).__init__(lib_dir, sim_lib, cl_weights, soltn_lib=soltn_lib)

        if mpi.rank == 0:
            fname_mask = os.path.join(self.lib_dir, "fmask.fits.gz")
            if not os.path.exists(fname_mask):
                fmask = self.cinv_tp.get_fmask()
                assert np.all(fmask == self.cinv_tp.get_fmask())
                hp.write_map(fname_mask, fmask)

        mpi.barrier()
        utils.hash_check(pk.load(open(os.path.join(lib_dir, "filt_hash.pk"), 'rb')), self.hashdict(), fn=os.path.join(lib_dir, "filt_hash.pk"))

    def hashdict(self):
        return {'cinv_tp': self.cinv_tp.hashdict(),
                'clw':{k:utils.clhash(self.cl[k]) for k in self.cl.keys()},
                'sim_lib': self.sim_lib.hashdict()}

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, "fmask.fits.gz"))

    def get_fal(self, lmax=None):
        return self.cinv_tp.get_fal(lmax=lmax)

    def _apply_ivf(self, tqumap, soltn=None):
        return self.cinv_tp.apply_ivf(tqumap, soltn=soltn)