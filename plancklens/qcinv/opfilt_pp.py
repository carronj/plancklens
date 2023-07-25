"""Pol-only Wiener and inverse variance filtering module.

This module collects definitions for the polarization-only forward and pre-conditioners operations.
There are three types of pre-conditioners: dense, diagonal in harmonic space, and multigrid stage.

 $$ S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}$$

This now allows for template projection in polarization, and differing E and B-mode transfer functions

"""

import hashlib
import numpy  as np
import healpy as hp

from plancklens.shts import alm2map_spin, map2alm_spin
#: Exporting these two methods so that they can be easily customized / optimized.

from plancklens.utils import clhash, enumerate_progress
from plancklens.qcinv import template_removal

from plancklens.qcinv.util_alm import eblm
from plancklens.qcinv import dense, util



class dot_op:
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        assert alm1.lmax == alm2.lmax
        tcl = hp.alm2cl(alm1.elm, alm2.elm) + hp.alm2cl(alm1.blm, alm2.blm)
        return np.sum(tcl[2:] * (2. * np.arange(2, alm1.lmax + 1) + 1))


class fwd_op:
    """Missing doc. """
    def __init__(self, s_cls, n_inv_filt):
        lmax = len(n_inv_filt.b_transf) - 1
        self.s_inv_filt = alm_filter_sinv(s_cls, lmax)
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'s_inv_filt': self.s_inv_filt.hashdict(),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, alm):
        nlm = alm * 1.0
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(alm)
        return nlm + slm

class pre_op_diag:
    """Missing doc. """
    def __init__(self, s_cls, n_inv_filt):
        lmax = len(n_inv_filt.b_transf) - 1
        s_inv_filt = alm_filter_sinv(s_cls, lmax)
        assert ((s_inv_filt.lmax + 1) >= len(n_inv_filt.b_transf))

        ninv_fel, ninv_fbl = n_inv_filt.get_febl()

        flmat = s_inv_filt.slinv
        flmat[:, 0, 0] += ninv_fel[:lmax + 1]
        flmat[:, 1, 1] += ninv_fbl[:lmax + 1]
        flmat = np.linalg.pinv(flmat)

        self.flmat = flmat

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        tmat = self.flmat
        relm = hp.almxfl(alm.elm, tmat[:, 0, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 0, 1], inplace=False)
        rblm = hp.almxfl(alm.elm, tmat[:, 1, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 1, 1], inplace=False)
        return eblm([relm, rblm])


def pre_op_dense(lmax, fwd_op, cache_fname=None):
    """Missing doc. """
    return dense.pre_op_dense_pp(lmax, fwd_op, cache_fname=cache_fname)

class alm_filter_sinv:
    """Missing doc. """
    def __init__(self, s_cls, lmax):
        slmat = np.zeros((lmax + 1, 2, 2), dtype=float)
        slmat[:, 0, 0] = s_cls.get('ee', np.zeros(lmax + 1))[:lmax + 1]
        slmat[:, 0, 1] = s_cls.get('eb', np.zeros(lmax + 1))[:lmax + 1]
        slmat[:, 1, 0] = s_cls.get('eb', np.zeros(lmax + 1))[:lmax + 1]
        slmat[:, 1, 1] = s_cls.get('bb', np.zeros(lmax + 1))[:lmax + 1]

        slinv = np.zeros((lmax + 1, 2, 2))
        for l in range(0, lmax + 1):
            slinv[l, :, :] = np.linalg.pinv(slmat[l])

        self.lmax = lmax
        self.slinv = slinv

    def calc(self, alm):
        tmat = self.slinv
        relm = hp.almxfl(alm.elm, tmat[:, 0, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 0, 1], inplace=False)
        rblm = hp.almxfl(alm.elm, tmat[:, 1, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 1, 1], inplace=False)
        return eblm([relm, rblm])

    def hashdict(self):
        return {'slinv': clhash(self.slinv.flatten())}


class alm_filter_ninv(object):
    def __init__(self, n_inv, b_transf,
                 nlev_febl=None, b_transf_b=None, marge_qmaps=(), marge_umaps=()):
        """Inverse-variance filtering instance for polarization only

            Args:
                n_inv: inverse pixel variance maps or masks
                b_transf: filter fiducial transfer function
                nlev_febl(optional): isotropic approximation to the noise level across the entire map
                                     this is used e.g. in the diag. preconditioner of cg inversion.
                b_transf_b: B-mode transfer func if different from E-mode

            Note:
                This allows for independent Q and U map marginalization

        """

        self.b_transf_e = b_transf
        self.b_transf_b = b_transf_b if b_transf_b is not None else b_transf
        self.b_transf = 0.5 * (self.b_transf_e + self.b_transf_b)

        # These three things will be instantiated later on
        self.nside = None
        self.n_inv = None

        self.nlev_febl = nlev_febl
        self._n_inv = n_inv # could be paths or list of paths


        self.marge_qmaps = marge_qmaps
        self.marge_umaps = marge_umaps

        self.wmarg = (max(len(self.marge_qmaps), len(self.marge_umaps)) > 0)
        self.tniti = None
        self.templates_p = []

    def _build_tniti(self):
        if not self.wmarg or self.tniti is not None:
            return
        tniti_m = []
        for im, marge_m in enumerate((self.marge_qmaps, self.marge_umaps)):
            if len(marge_m) > 0:
                this_n_inv = self.get_ninv()
                assert len(this_n_inv) == 1, 'QQ QU UU not implemented'  # QQ = UU, QU == 0
                templates = []
                tfunc = template_removal.template_qmap if im == 0 else template_removal.template_umap
                for m in marge_m:
                    templates.append(tfunc(m))
                nmodes = int(np.sum([t.nmodes for t in templates]))
                modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(templates)]))
                modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in templates]))
                Pt_Nn1_P = np.zeros((nmodes, nmodes))
                for i, ir in enumerate_progress(range(nmodes),
                                                label='filling template (%s) projection matrix' % nmodes):
                    pmap = [np.copy(this_n_inv[0])]  # QQ only or UU only
                    templates[modes_idx_t[ir]].apply_mode(pmap, int(modes_idx_i[ir]))
                    ic = 0
                    for tc in templates[0:modes_idx_t[ir] + 1]:
                        Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(pmap)
                        Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                        ic += tc.nmodes
                eigv, eigw = np.linalg.eigh(Pt_Nn1_P)
                eigv_inv = 1.0 / eigv
                tniti_m.append(np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw)))
                self.templates_p = self.templates_p + templates

        if len(tniti_m) > 0:  # put together the Q U marginalizations:
            nmodes = np.sum([tniti.shape[0] for tniti in tniti_m])
            self.tniti = np.zeros((nmodes, nmodes), dtype=float)
            idx = 0
            for tniti in tniti_m:
                nmodes = tniti.shape[0]
                self.tniti[idx:idx + nmodes, idx:idx + nmodes] = np.copy(tniti)
                idx += nmodes

    def _load_ninv(self):
        if self.n_inv is None:
            self.n_inv = []
            for i, tn in enumerate(self._n_inv):
                if isinstance(tn, list):
                    n_inv_prod = util.read_map(tn[0])
                    if len(tn) > 1:
                        for n in tn[1:]:
                            n_inv_prod = n_inv_prod * util.read_map(n)
                    self.n_inv.append(n_inv_prod)
                else:
                    self.n_inv.append(util.read_map(self._n_inv[i]))
            assert len(self.n_inv) in [1, 3], len(self.n_inv)
            self.nside = hp.npix2nside(len(self.n_inv[0]))

    def _calc_febl(self):
        self._load_ninv()
        if len(self.n_inv) == 1:
            nlev_febl = 10800. / np.sqrt(np.sum(self.n_inv[0]) / (4.0 * np.pi)) / np.pi
        elif len(self.n_inv) == 3:
            nlev_febl = 10800. / np.sqrt(np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi)) / np.pi
        else:
            assert 0
        print("ninv_febl: using %.2f uK-amin noise Cl"%nlev_febl)
        return nlev_febl

    def get_ninv(self):
        self._load_ninv()
        return self.n_inv

    def get_mask(self):
        ninv = self.get_ninv()
        assert len(ninv) in [1, 3], len(ninv)
        self.nside = hp.npix2nside(len(ninv[0]))
        mask = np.where(ninv[0] > 0, 1., 0)
        for ni in ninv[1:]:
            mask *= (ni > 0)
        return mask

    def get_febl(self):
        if self.nlev_febl is None:
            self.nlev_febl = self._calc_febl()
        n_inv_cl_e = self.b_transf_e ** 2  / (self.nlev_febl / 180. / 60. * np.pi) ** 2
        n_inv_cl_b = self.b_transf_b ** 2  / (self.nlev_febl / 180. / 60. * np.pi) ** 2
        return n_inv_cl_e, n_inv_cl_b

    def hashdict(self):
        if not self.wmarg:
            ret = {'n_inv': [util.mask_hash(n, dtype=np.float16) for n in self._n_inv],
               'b_transf': clhash(self.b_transf), 'templates_p': []}
        else:
            t_hash  = [util.mask_hash(m, dtype=np.float32) for m in self.marge_qmaps]
            t_hash += [util.mask_hash(m, dtype=np.float32) for m in self.marge_umaps]
            ret = {'n_inv': [util.mask_hash(n, dtype=np.float16) for n in self._n_inv],
                   'b_transf': clhash(self.b_transf), 'templates_p': t_hash}
        return ret


    def degrade(self, nside):
        self._load_ninv()
        if nside == self.nside:
            return self
        else:
            return alm_filter_ninv([hp.ud_grade(n, nside, power=-2) for n in self.n_inv], self.b_transf_e, b_transf_b=self.b_transf_b)

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        self._load_ninv()
        lmax = alm.lmax

        hp.almxfl(alm.elm, self.b_transf_e, inplace=True)
        hp.almxfl(alm.blm, self.b_transf_b, inplace=True)
        qmap, umap = alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}
        npix = len(qmap)

        telm, tblm = map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm
        alm.blm[:] = tblm

        hp.almxfl(alm.elm, self.b_transf_e * (npix / (4. * np.pi)), inplace=True)
        hp.almxfl(alm.blm, self.b_transf_b * (npix / (4. * np.pi)), inplace=True)

    def apply_map(self, amap):
        self._load_ninv()
        [qmap, umap] = amap
        if len(self.n_inv) == 1:  # TT, QQ=UU
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]
            if self.wmarg:
                self._build_tniti()
                coeffs = np.concatenate(([t.dot([qmap, umap]) for t in self.templates_p]))
                coeffs = np.dot(self.tniti, coeffs)
                pmodes = [np.zeros_like(qmap), np.zeros_like(umap)]
                im = 0
                for t in self.templates_p:
                    t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                    im += t.nmodes
                pmodes[0] *= self.n_inv[0]
                pmodes[1] *= self.n_inv[0]
                qmap -= pmodes[0]
                umap -= pmodes[1]

        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0


def calc_prep(maps, s_cls, n_inv_filt:alm_filter_ninv):
    qmap = np.copy(util.read_map(maps[0]))
    umap = np.copy(util.read_map(maps[1]))
    assert len(qmap) == len(umap)
    lmax = len(n_inv_filt.b_transf) - 1
    npix = len(qmap)

    n_inv_filt.apply_map([qmap, umap])
    elm, blm = map2alm_spin([qmap, umap], 2, lmax=lmax)
    hp.almxfl(elm, n_inv_filt.b_transf_e * npix / (4. * np.pi), inplace=True)
    hp.almxfl(blm, n_inv_filt.b_transf_b * npix / (4. * np.pi), inplace=True)
    return eblm([elm, blm])


def apply_fini(alm, s_cls, n_inv_filt:alm_filter_ninv):
    sfilt = alm_filter_sinv(s_cls, alm.lmax)
    ret = sfilt.calc(alm)
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]