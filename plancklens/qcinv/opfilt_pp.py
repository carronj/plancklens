"""Pol-only Wiener and inverse variance filtering module.

This module collects definitions for the polarization-only forward and pre-conditioners operations.
There are three types of pre-conditioners: dense, diagonal in harmonic space, and multigrid stage.

 $$ S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}$$


"""

import numpy  as np
import healpy as hp

from healpy import alm2map_spin, map2alm_spin
#: Exporting these two methods so that they can be easily customized / optimized.

from plancklens.utils import clhash

from . import util
from .util_alm import eblm
from . import dense


def calc_prep(maps, s_cls, n_inv_filt):
    qmap = np.copy(util.read_map(maps[0]))
    umap = np.copy(util.read_map(maps[1]))
    assert len(qmap) == len(umap)
    lmax = len(n_inv_filt.b_transf) - 1
    npix = len(qmap)

    n_inv_filt.apply_map([qmap, umap])
    elm, blm = map2alm_spin([qmap, umap], 2, lmax=lmax)
    hp.almxfl(elm, n_inv_filt.b_transf * npix / (4. * np.pi), inplace=True)
    hp.almxfl(blm, n_inv_filt.b_transf * npix / (4. * np.pi), inplace=True)
    return eblm([elm, blm])


def apply_fini(alm, s_cls, n_inv_filt):
    sfilt = alm_filter_sinv(s_cls, alm.lmax)
    ret = sfilt.calc(alm)
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]

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
    def __init__(self, n_inv, b_transf, nlev_febl=None):
        """Inverse-variance filtering instance for polarization only

            Args:
                n_inv: inverse pixel variance maps or masks
                b_transf: filter fiducial transfer function
                nlev_febl(optional): isotropic approximation to the noise level across the entire map
                                     this is used e.g. in the diag. preconditioner of cg inversion.

            Note:
                This implementation does not support template projection.


        """
        self.n_inv = []
        for i, tn in enumerate(n_inv):
            if isinstance(tn, list):
                n_inv_prod = util.load_map(tn[0])
                if len(tn) > 1:
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * util.load_map(n)
                self.n_inv.append(n_inv_prod)
            else:
                self.n_inv.append(util.load_map(n_inv[i]))
        n_inv = self.n_inv

        assert len(n_inv) in [1, 3], len(n_inv)

        def std_av(inoise_map):
            if np.all(inoise_map == 0.):
                return 0.
            return np.std(n[np.where(inoise_map != 0.0)]) / np.average(n[np.where(inoise_map != 0.0)])


        npix = len(n_inv[0])
        nside = hp.npix2nside(npix)
        for n, S in zip(n_inv, ['Pol.' if len(n_inv) == 1 else 'QQ' , 'QU', 'UU']):
            assert len(n) == npix
            print("opfilt_pp: %s inverse noise map std dev / av = %.3e" % (S, std_av(n)))

        self.templates_p = [] # Templates not implemented for polarization
        self.n_inv = n_inv
        self.b_transf = b_transf

        self.npix = npix
        self.nside = nside
        if nlev_febl is None:
            if len(self.n_inv) == 1:
                nlev_febl =  10800. / np.sqrt(np.sum(self.n_inv[0]) / (4.0 * np.pi)) / np.pi
            elif len(self.n_inv) == 3:
                nlev_febl = 10800. / np.sqrt(np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi))  / np.pi
            else:
                assert 0
        self.nlev_febl = nlev_febl
        print("ninv_febl: using %.2f uK-amin noise Cl"%self.nlev_febl)

    def get_febl(self):
        n_inv_cl_p = self.b_transf ** 2  / (self.nlev_febl / 180. / 60. * np.pi) ** 2
        return n_inv_cl_p, n_inv_cl_p

    def hashdict(self):
        return {'n_inv': [clhash(n) for n in self.n_inv],
                'b_transf': clhash(self.b_transf), 'templates_p':self.templates_p}

    def degrade(self, nside):
        if nside == self.nside:
            return self
        else:
            return alm_filter_ninv([hp.ud_grade(n, nside, power=-2) for n in self.n_inv], self.b_transf)

    def apply_alm(self, alm):
        """B^dagger N^{-1} B"""
        lmax = alm.lmax

        hp.almxfl(alm.elm, self.b_transf, inplace=True)
        hp.almxfl(alm.blm, self.b_transf, inplace=True)
        qmap, umap = alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}
        npix = len(qmap)

        telm, tblm = map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm
        alm.blm[:] = tblm

        hp.almxfl(alm.elm, self.b_transf * (npix / (4. * np.pi)), inplace=True)
        hp.almxfl(alm.blm, self.b_transf * (npix / (4. * np.pi)), inplace=True)

    def apply_map(self, amap):
        [qmap, umap] = amap
        if len(self.n_inv) == 1:  # TT, QQ=UU
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]
        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0