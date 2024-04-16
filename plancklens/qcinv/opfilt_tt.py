"""Temperature-only Wiener and inverse variance filtering module.

This module collects definitions for the temperature-only forward and pre-conditioners operations.
There are three types of pre-conditioners: dense, diagonal in harmonic space, and multigrid stage.

 $$ S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}$$
"""
#FIXME: docs
from __future__ import absolute_import
from __future__ import print_function

import hashlib
import numpy  as np
import healpy as hp

from plancklens.shts import alm2map, map2alm
#: Exporting these two methods so that they can be easily customized / optimized.

from plancklens.utils import clhash, enumerate_progress

from . import util
from . import template_removal
from . import dense

def _cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0.)] = 1. / cl[np.where(cl != 0.)]
    return ret

def calc_prep(m, s_cls, n_inv_filt):
    """Missing doc."""
    tmap = np.copy(m)
    n_inv_filt.apply_map(tmap)
    alm = map2alm(tmap, lmax=len(n_inv_filt.b_transf) - 1, iter=0)
    hp.almxfl(alm, n_inv_filt.b_transf * (len(m) / (4. * np.pi)), inplace=True)
    return alm


def apply_fini(alm, s_cls, n_inv_filt):
    """ This final operation turns the Wiener-filtered CMB cg-solution to the inverse-variance filtered CMB.  """
    hp.almxfl(alm, _cli(s_cls['tt']), inplace=True)

class dot_op:
    """ Scalar product definition for cg-inversion """
    def __init__(self):
        pass

    def __call__(self, alm1, alm2):
        lmax1 = hp.Alm.getlmax(alm1.size)
        assert lmax1 == hp.Alm.getlmax(alm2.size)
        return np.sum(hp.alm2cl(alm1, alms2=alm2) * (2. * np.arange(0, lmax1 + 1) + 1))


class fwd_op:
    """Conjugate-gradient inversion forward operation definition. """
    def __init__(self, s_cls, n_inv_filt):
        self.cltt_inv = _cli(s_cls['tt'])
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'cltt_inv': clhash(self.cltt_inv),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if np.all(talm == 0):  # do nothing if zero
            return talm
        alm = np.copy(talm)
        self.n_inv_filt.apply_alm(alm)
        alm += hp.almxfl(talm, self.cltt_inv)
        return alm


class pre_op_diag:
    def __init__(self, s_cls, n_inv_filt):
        """Harmonic space diagonal pre-conditioner operation. """
        cltt = s_cls['tt']
        assert len(cltt) >= len(n_inv_filt.b_transf)
        n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi)
        lmax = len(n_inv_filt.b_transf) - 1
        assert lmax <= (len(cltt) - 1)

        filt = _cli(cltt[:lmax + 1])
        filt += n_inv_cl * n_inv_filt.b_transf[:lmax + 1] ** 2
        self.filt = _cli(filt)

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)

def pre_op_dense(lmax, fwd_op, cache_fname=None):
    """Missing doc. """
    return dense.pre_op_dense_tt(lmax, fwd_op, cache_fname=cache_fname)

class alm_filter_ninv(object):
    """Missing doc. """
    def __init__(self, n_inv, b_transf,
                 marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=(), nlev_ftl=None):
        if isinstance(n_inv, list):
            n_inv_prod = util.load_map(n_inv[0])
            if len(n_inv) > 1:
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * util.load_map(n)
            n_inv = n_inv_prod
        else:
            n_inv = util.load_map(n_inv)
        print("opfilt_tt: inverse noise map std dev / av = %.3e" % (
                    np.std(n_inv[np.where(n_inv != 0.0)]) / np.average(n_inv[np.where(n_inv != 0.0)])))
        templates = []
        templates_hash = []
        for tmap in [util.load_map(m) for m in marge_maps]:
            assert (len(n_inv) == len(tmap))
            templates.append(template_removal.template_map(tmap))
            templates_hash.append(hashlib.sha1(tmap.view(np.uint8)).hexdigest())

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
                tmap = np.copy(n_inv)
                templates[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates[0:modes_idx_t[ir] + 1]:
                    Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(tmap)
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

        if nlev_ftl is None:
            nlev_ftl =  10800. / np.sqrt(np.sum(self.n_inv) / (4.0 * np.pi)) / np.pi
        self.nlev_ftl = nlev_ftl
        print("ninv_ftl: using %.2f uK-amin noise Cl"%self.nlev_ftl)

    def hashdict(self):
        return {'n_inv': clhash(self.n_inv),
                'b_transf': clhash(self.b_transf),
                'marge_monopole': self.marge_monopole,
                'marge_dipole': self.marge_dipole,
                'templates_hash': self.templates_hash,
                'marge_uptolmin': self.marge_uptolmin}

    def get_ftl(self):
        return  self.b_transf ** 2 / (self.nlev_ftl / 60. /180. *  np.pi) ** 2


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
        tmap = alm2map(alm, hp.npix2nside(npix))
        self.apply_map(tmap)
        alm[:] = map2alm(tmap, lmax=hp.Alm.getlmax(alm.size), iter=0)
        hp.almxfl(alm, self.b_transf  *  (npix / (4. * np.pi)), inplace=True)


    def apply_map(self, tmap):
        """Missing doc. """
        tmap *= self.n_inv
        if len(self.templates) != 0:
            coeffs = np.concatenate(([t.dot(tmap) for t in self.templates]))
            coeffs = np.dot(self.Pt_Nn1_P_inv, coeffs)
            pmodes = np.zeros(len(self.n_inv))
            im = 0
            for t in self.templates:
                t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv
            tmap -= pmodes