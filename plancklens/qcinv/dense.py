# FIXME: zero eigenvalues message in opfilt_pp dense because of ell = 0 and 1, and then the eigv are set to unity.
# FIXME: better project on ell > 2 ? It wont change anything

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy  as np
import pickle as pk
from healpy import Alm

from .util_alm import eblm, teblm
from plancklens.utils import enumerate_progress

def alm2rlm(alm):
    """Converts a complex alm to 'real harmonic' coefficients rlm.

    This 'real harmonic' form is used for the dense matrix preconditioner tools.

     """
    lmax = Alm.getlmax(alm.size)
    rlm = np.zeros((lmax + 1) ** 2, dtype=float)

    ls = np.arange(0, lmax + 1)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in range(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].imag * rt2
    return rlm


def rlm2alm(rlm):
    """Converts 'real harmonic' coefficients rlm to complex alm.

    Inverse of alm2rlm.

    """
    lmax = int(np.sqrt(len(rlm)) - 1)
    assert (lmax + 1) ** 2 == len(rlm)

    alm = np.zeros(Alm.getsize(lmax), dtype=complex)
    ls = np.arange(0, lmax + 1, dtype=int)

    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in range(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1.j * rlm[l2s[m:] + 2 * m + 0]) * ir2
    return alm


class pre_op_dense_tt:
    """Constructs a low-l, low-nside dense preconditioner by brute force. """
    def __init__(self, lmax, fwd_op, cache_fname=None):
        if (cache_fname is not None) and os.path.exists(cache_fname):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load(open(cache_fname, 'rb'))
            self.minv = cache_minv
            if (lmax != cache_lmax) or (self.hashdict(lmax, fwd_op) != cache_hashdict):
                print("WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing.")
                os.remove(cache_fname)
                self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)
        else:
            self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname is not None:
            assert not os.path.exists(cache_fname)

        nrlm = (lmax + 1) ** 2
        trlm = np.zeros(nrlm)
        tmat = np.zeros((nrlm, nrlm))

        ntmpl = 0
        for t in fwd_op.n_inv_filt.templates:
            ntmpl += t.nmodes

        print("computing dense preconditioner:")
        print("     lmax  =", lmax)
        print("     ntmpl =", ntmpl)

        if cache_fname is not None: print(" will cache minv in " + cache_fname)

        for j, i in enumerate_progress(np.arange(0, nrlm), label= 'filling matrix'):
            trlm[i] = 1.0
            tmat[:, i] = alm2rlm(fwd_op(rlm2alm(trlm)))
            trlm[i] = 0.0

        print("   inverting M...")
        eigv, eigw = np.linalg.eigh(tmat)

        assert np.all(eigv[ntmpl:] > 0.)
        eigv_inv = np.zeros_like(eigv)
        eigv_inv[ntmpl:] = 1.0 / eigv[ntmpl:]

        if ntmpl > 0:
            print("     eigv[ntmpl-1] = ", eigv[ntmpl - 1])
            print("     eigv[ntmpl]   = ", eigv[ntmpl])
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        if cache_fname is not None:
            pk.dump([lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'wb'))

    @staticmethod
    def hashdict(lmax, fwd_op):
        return {'lmax': lmax,
                'fwd_op': fwd_op.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return rlm2alm(np.dot(self.minv, alm2rlm(talm)))

pre_op_dense_kk = pre_op_dense_tt

class pre_op_dense_pp:
    """Missing doc. """
    def __init__(self, lmax, fwd_op, cache_fname=None):
        if (cache_fname is not None) and os.path.exists(cache_fname):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load(open(cache_fname, 'rb'))
            self.minv = cache_minv

            if (lmax != cache_lmax) or (self.hashdict(lmax, fwd_op) != cache_hashdict):
                print("WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing.")
                os.remove(cache_fname)
                self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)
        else:
            self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)

    @staticmethod
    def alm2rlm(alm):
        rlm = np.zeros(2 * (alm.lmax + 1) ** 2, dtype=float)
        rlm[0 * (alm.lmax + 1) ** 2:1 * (alm.lmax + 1) ** 2] = alm2rlm(alm.elm)
        rlm[1 * (alm.lmax + 1) ** 2:2 * (alm.lmax + 1) ** 2] = alm2rlm(alm.blm)
        return rlm

    @staticmethod
    def rlm2alm(rlm):
        lmax = int(np.sqrt(len(rlm) / 2) - 1)
        return eblm([rlm2alm(rlm[0 * (lmax + 1) ** 2:1 * (lmax + 1) ** 2]),
                     rlm2alm(rlm[1 * (lmax + 1) ** 2:2 * (lmax + 1) ** 2])])

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname is not None:
            assert not os.path.exists(cache_fname)

        nrlm = 2 * (lmax + 1) ** 2
        trlm = np.zeros(nrlm)
        tmat = np.zeros((nrlm, nrlm))

        ntmpl = 0
        if getattr(fwd_op.n_inv_filt, 'templates_p', None) is None:
            print("dense: did not find templates_p attribute")
        else:
            for t in fwd_op.n_inv_filt.templates_p:
                ntmpl += t.nmodes
        ntmpl += 8  # (1 mono + 3 dip) * (e+b)

        print("computing dense preconditioner:")
        print("     lmax  =", lmax)
        print("     ntmpl =", ntmpl)

        for j, i in enumerate_progress(np.arange(0, nrlm), label= 'filling matrix'):
            trlm[i] = 1.0
            tmat[:, i] = self.alm2rlm(fwd_op(self.rlm2alm(trlm)))
            trlm[i] = 0.0

        print("   inverting M...")
        eigv, eigw = np.linalg.eigh(tmat)

        assert np.all(eigv[ntmpl:] > 0.)
        eigv_inv = np.zeros_like(eigv)
        eigv_inv[ntmpl:] = 1.0 / eigv[ntmpl:]

        if ntmpl > 0:
            # do nothing to the ntmpl eigenmodes
            # with the lowest eigenvalues.
            print("     eigv[ntmpl-1] = ", eigv[ntmpl - 1])
            print("     eigv[ntmpl]   = ", eigv[ntmpl])
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        if cache_fname is not None:
            pk.dump([lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'wb'))

    @staticmethod
    def hashdict(lmax, fwd_op):
        return {'lmax': lmax, 'fwd_op': fwd_op.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return self.rlm2alm(np.dot(self.minv, self.alm2rlm(talm)))

class pre_op_dense_tp:
    """Missing doc. """
    def __init__(self, lmax, fwd_op, cache_fname=None):
        if (cache_fname is not None) and os.path.exists(cache_fname):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load(open(cache_fname, 'rb'))
            self.minv = cache_minv

            if (lmax != cache_lmax) or (self.hashdict(lmax, fwd_op) != cache_hashdict):
                print("WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing.")
                os.remove(cache_fname)
                self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)
        else:
            self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)

    @staticmethod
    def alm2rlm(alm):
        rlm = np.zeros(3 * (alm.lmax + 1) ** 2, dtype=float)
        rlm[0 * (alm.lmax + 1) ** 2:1 * (alm.lmax + 1) ** 2] = alm2rlm(alm.tlm)
        rlm[1 * (alm.lmax + 1) ** 2:2 * (alm.lmax + 1) ** 2] = alm2rlm(alm.elm)
        rlm[2 * (alm.lmax + 1) ** 2:3 * (alm.lmax + 1) ** 2] = alm2rlm(alm.blm)

        return rlm

    @staticmethod
    def rlm2alm(rlm):
        lmax = int(np.sqrt(len(rlm) // 3) - 1)
        return teblm([rlm2alm(rlm[0 * (lmax + 1) ** 2:1 * (lmax + 1) ** 2]),
                      rlm2alm(rlm[1 * (lmax + 1) ** 2:2 * (lmax + 1) ** 2]),
                      rlm2alm(rlm[2 * (lmax + 1) ** 2:3 * (lmax + 1) ** 2])])

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname is not None:
            assert not os.path.exists(cache_fname)

        nrlm = 3 * (lmax + 1) ** 2
        trlm = np.zeros(nrlm)
        tmat = np.zeros((nrlm, nrlm))

        ntmpl = 0
        for t in fwd_op.n_inv_filt.templates_t:
            ntmpl += t.nmodes # This should include mono and possibly dip
        for t in fwd_op.n_inv_filt.templates_p:
            ntmpl += t.nmodes
        ntmpl += 8  # (1 mono + 3 dip) * (e+b)

        print("computing dense preconditioner:")
        print("     lmax  =", lmax)
        print("     ntmpl =", ntmpl)

        for j, i in enumerate_progress(np.arange(0, nrlm), label= 'filling matrix'):
            trlm[i] = 1.0
            tmat[:, i] = self.alm2rlm(fwd_op(self.rlm2alm(trlm)))
            trlm[i] = 0.0

        print("   inverting M...")
        eigv, eigw = np.linalg.eigh(tmat)

        assert np.all(eigv[ntmpl:] > 0.)
        eigv_inv = np.zeros_like(eigv)
        eigv_inv[ntmpl:] = 1.0 / eigv[ntmpl:]

        if ntmpl > 0:
            # do nothing to the ntmpl eigenmodes
            # with the lowest eigenvalues.
            print("     eigv[ntmpl-1] = ", eigv[ntmpl - 1])
            print("     eigv[ntmpl]   = ", eigv[ntmpl])
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        if cache_fname is not None:
            pk.dump([lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'wb'))

    @staticmethod
    def hashdict(lmax, fwd_op):
        return {'lmax': lmax, 'fwd_op': fwd_op.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return self.rlm2alm(np.dot(self.minv, self.alm2rlm(talm)))