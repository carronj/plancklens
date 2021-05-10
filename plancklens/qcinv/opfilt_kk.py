"""lending map Wiener and inverse variance filtering module.

This is literally the very same spin-0 inverse variance filtering codes than for temperatures,
with indices 'tt' replaced with 'pp' and potential to k remapping


"""
from __future__ import absolute_import
from __future__ import print_function

import hashlib
import numpy  as np
import healpy as hp

from healpy import alm2map, map2alm
#: Exporting these two methods so that they can be easily customized / optimized.

from plancklens.utils import clhash, enumerate_progress

from . import util
from . import template_removal
from . import dense

def _cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return ret

def p2k(lmax):
    return 0.5 * np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)
def pp2kk(lmax):
    return p2k(lmax) ** 2


def calc_prep(m, s_cls, n_inv_filt):
    """Missing doc."""
    kmap = np.copy(m)
    n_inv_filt.apply_map(kmap)
    alm = map2alm(kmap, lmax=len(n_inv_filt.b_transf) - 1, iter=0)
    hp.almxfl(alm, n_inv_filt.b_transf * (len(m) / (4. * np.pi)), inplace=True)
    return alm

def apply_fini(alm, s_cls, n_inv_filt):
    """ This final operation turns the Wiener-filtered klm cg-solution to the inverse-variance filtered klm.  """
    hp.almxfl(alm, _cli(s_cls['pp'] * pp2kk(len(s_cls['pp']) - 1)), inplace=True)

class dot_op:
    """Scalar product definition for kk cg-inversion

    """
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        lmax1 = hp.Alm.getlmax(alm1.size)
        assert lmax1 == hp.Alm.getlmax(alm2.size)
        return np.sum(hp.alm2cl(alm1, alms2=alm2) * (2. * np.arange(0, lmax1 + 1) + 1))


class fwd_op:
    """Conjugate-gradient inversion forward operation definition. """
    def __init__(self, s_cls, n_inv_filt):
        self.clkk_inv = _cli(s_cls['pp'] * pp2kk(len(s_cls['pp']) - 1))
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'clkk_inv': clhash(self.clkk_inv),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, klm):
        return self.calc(klm)

    def calc(self, klm):
        if np.all(klm == 0):  # do nothing if zero
            return klm
        alm = np.copy(klm)
        self.n_inv_filt.apply_alm(alm)
        alm += hp.almxfl(klm, self.clkk_inv)
        return alm


class pre_op_diag:
    def __init__(self, s_cls, n_inv_filt):
        """Harmonic space diagonal pre-conditioner operation. """
        clkk = pp2kk(len(s_cls['pp']) - 1) * s_cls['pp']
        assert len(clkk) >= len(n_inv_filt.b_transf)
        n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi)
        lmax = len(n_inv_filt.b_transf) - 1
        assert lmax <= (len(clkk) - 1)

        filt = _cli(clkk[:lmax + 1])
        filt += n_inv_cl * n_inv_filt.b_transf[:lmax + 1] ** 2
        self.filt = _cli(filt)

    def __call__(self, klm):
        return self.calc(klm)

    def calc(self, klm):
        return hp.almxfl(klm, self.filt)

def pre_op_dense(lmax, fwd_op, cache_fname=None):
    """Missing doc. """
    return dense.pre_op_dense_kk(lmax, fwd_op, cache_fname=cache_fname)

class alm_filter_ninv(object):
    """Missing doc. """
    def __init__(self, n_inv, b_transf,
                 marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_fkl=None):
        if isinstance(n_inv, list):
            n_inv_prod = util.load_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * util.load_map(n)
            n_inv = n_inv_prod
        else:
            n_inv = util.load_map(n_inv)
        print("opfilt_kk: inverse noise map std dev / av = %.3e" % (
                    np.std(n_inv[np.where(n_inv != 0.0)]) / np.average(n_inv[np.where(n_inv != 0.0)])))
        templates = []
        templates_hash = []
        for kmap in [util.load_map(m) for m in marge_maps]:
            assert (len(n_inv) == len(kmap))
            templates.append(template_removal.template_map(kmap))
            templates_hash.append(hashlib.sha1(kmap.view(np.uint8)).hexdigest())

        if marge_uptolmin >= 0:
            templates.append(template_removal.template_uptolmin(marge_uptolmin))
        else:
            if marge_monopole: templates.append(template_removal.template_monopole())
            if marge_dipole: templates.append(template_removal.template_dipole())

        if len(templates) != 0:
            nmodes = int(np.sum([t.nmodes for t in templates]))
            modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(templates)]))
            modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in templates]))
            Pt_Nn1_P = np.zeros((nmodes, nmodes))
            for i, ir in enumerate_progress(range(nmodes), label='filling template (%s) projection matrix'%nmodes):
                kmap = np.copy(n_inv)
                templates[modes_idx_t[ir]].apply_mode(kmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates[0:modes_idx_t[ir] + 1]:
                    Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(kmap)
                    Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                    ic += tc.nmodes
            eigv, eigw = np.linalg.eigh(Pt_Nn1_P)
            eigv_inv = 1.0 / eigv
            self.Pt_Nn1_P_inv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        self.n_inv = n_inv
        self.b_transf = b_transf
        self.npix = len(self.n_inv)

        self.nside = hp.npix2nside(self.npix)
        self.marge_monopole = marge_monopole
        self.marge_dipole = marge_dipole
        self.marge_uptolmin = marge_uptolmin
        self.templates = templates
        self.templates_hash = templates_hash

        if nlev_fkl is None:
            nlev_fkl =  10800. / np.sqrt(np.sum(self.n_inv) / (4.0 * np.pi)) / np.pi
        self.nlev_fkl = nlev_fkl
        print("ninv_fkl: using %.2e uK-amin noise Cl"%self.nlev_fkl)

    def hashdict(self):
        return {'n_inv': clhash(self.n_inv),
                'b_transf': clhash(self.b_transf),
                'marge_monopole': self.marge_monopole,
                'marge_dipole': self.marge_dipole,
                'templates_hash': self.templates_hash,
                'marge_uptolmin': self.marge_uptolmin}

    def get_fkl(self):
        return  self.b_transf ** 2 / (self.nlev_fkl / 60. /180. *  np.pi) ** 2


    def degrade(self, nside):
        """Missing doc. """
        if nside == hp.npix2nside(len(self.n_inv)):
            return self
        else:
            print("DEGRADING WITH NO MARGE MAPS")
            marge_maps = []
            return alm_filter_ninv(hp.ud_grade(self.n_inv, nside, power=-2), self.b_transf,
                        marge_monopole=self.marge_monopole, marge_dipole= self.marge_dipole,
                        marge_uptolmin=self.marge_uptolmin, marge_maps=marge_maps)

    def apply_alm(self, alm):
        """Missing doc. """
        npix = len(self.n_inv)
        hp.almxfl(alm, self.b_transf, inplace=True)
        kmap = alm2map(alm, hp.npix2nside(npix), verbose=False)
        self.apply_map(kmap)
        alm[:] = map2alm(kmap, lmax=hp.Alm.getlmax(alm.size), iter=0)
        hp.almxfl(alm, self.b_transf  *  (npix / (4. * np.pi)), inplace=True)


    def apply_map(self, kmap):
        """Missing doc. """
        kmap *= self.n_inv
        if len(self.templates) != 0:
            coeffs = np.concatenate(([t.dot(kmap) for t in self.templates]))
            coeffs = np.dot(self.Pt_Nn1_P_inv, coeffs)
            pmodes = np.zeros(len(self.n_inv))
            im = 0
            for t in self.templates:
                t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv
            kmap -= pmodes