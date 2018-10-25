from __future__ import absolute_import
from __future__ import print_function

import numpy  as np
import os
import pickle as pk

from . import util_alm


class pre_op_dense_tt:
    def __init__(self, lmax, fwd_op, cache_fname=None):
        # construct a low-l, low-nside dense preconditioner by brute force.
        # order of operations is O(nside**2 lmax**3) ~ O(lmax**5), so doing
        # by brute force is still comparable to matrix inversion, with
        # benefit of being very simple to implement.

        if (cache_fname is not None) and os.path.exists(cache_fname):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load(open(cache_fname, 'r'))
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

        for i in np.arange(0, nrlm):
            if np.mod(i, int(0.1 * nrlm)) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            trlm[i] = 1.0
            tmat[:, i] = util_alm.alm2rlm(fwd_op(util_alm.rlm2alm(trlm)))
            trlm[i] = 0.0

        print("   inverting M...")
        eigv, eigw = np.linalg.eigh(tmat)

        eigv_inv = 1.0 / eigv

        if ntmpl > 0:
            # do nothing to the ntmpl eigenmodes
            # with the lowest eigenvalues.
            print("     eigv[ntmpl-1] = ", eigv[ntmpl - 1])
            print("     eigv[ntmpl]   = ", eigv[ntmpl])
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        if cache_fname is not None:
            pk.dump([lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'w'))

    @staticmethod
    def hashdict(lmax, fwd_op):
        return {'lmax': lmax,
                'fwd_op': fwd_op.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return util_alm.rlm2alm(np.dot(self.minv, util_alm.alm2rlm(talm)))


class pre_op_dense_pp:
    def __init__(self, lmax, fwd_op, cache_fname=None):
        # construct a low-l, low-nside dense preconditioner by brute force.
        # order of operations is O(nside**2 lmax**3) ~ O(lmax**5), so doing
        # by brute force is still comparable to matrix inversion, with
        # benefit of being very simple to implement.

        if (cache_fname != None) and (os.path.exists(cache_fname)):
            [cache_lmax, cache_hashdict, cache_minv] = pk.load(open(cache_fname, 'r'))
            self.minv = cache_minv

            if (lmax != cache_lmax) or (self.hashdict(lmax, fwd_op) != cache_hashdict):
                print("WARNING: PRE_OP_DENSE CACHE: hashcheck failed. recomputing.")
                os.remove(cache_fname)
                self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)
        else:
            self.compute_minv(lmax, fwd_op, cache_fname=cache_fname)

    @staticmethod
    def alm2rlm(alm):
        rlm = np.zeros(2 * (alm.lmax + 1) ** 2)
        rlm[0 * (alm.lmax + 1) ** 2:1 * (alm.lmax + 1) ** 2] = util_alm.alm2rlm(alm.elm)
        rlm[1 * (alm.lmax + 1) ** 2:2 * (alm.lmax + 1) ** 2] = util_alm.alm2rlm(alm.blm)
        return rlm

    @staticmethod
    def rlm2alm(rlm):
        lmax = int(np.sqrt(len(rlm) / 2) - 1)
        return util_alm.eblm([util_alm.rlm2alm(rlm[0 * (lmax + 1) ** 2:1 * (lmax + 1) ** 2]),
                              util_alm.rlm2alm(rlm[1 * (lmax + 1) ** 2:2 * (lmax + 1) ** 2])])

    def compute_minv(self, lmax, fwd_op, cache_fname=None):
        if cache_fname is not None:
            assert not os.path.exists(cache_fname)

        nrlm = 2 * (lmax + 1) ** 2
        trlm = np.zeros(nrlm)
        tmat = np.zeros((nrlm, nrlm))

        ntmpl = 0
        for t in fwd_op.n_inv_filt.templates_p:
            ntmpl += t.nmodes
        ntmpl += 8  # (1 mono + 3 dip) * (e+b)

        print("computing dense preconditioner:")
        print("     lmax  =", lmax)
        print("     ntmpl =", ntmpl)

        for i in np.arange(0, nrlm):
            if np.mod(i, int(0.1 * nrlm)) == 0: print ("   filling M: %4.1f" % (100. * i / nrlm)), "%"
            trlm[i] = 1.0
            tmat[:, i] = self.alm2rlm(fwd_op(self.rlm2alm(trlm)))
            trlm[i] = 0.0

        print("   inverting M...")
        eigv, eigw = np.linalg.eigh(tmat)

        eigv_inv = 1.0 / eigv

        if ntmpl > 0:
            # do nothing to the ntmpl eigenmodes
            # with the lowest eigenvalues.
            print("     eigv[ntmpl-1] = ", eigv[ntmpl - 1])
            print("     eigv[ntmpl]   = ", eigv[ntmpl])
            eigv_inv[0:ntmpl] = 1.0

        self.minv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        if cache_fname is not None:
            pk.dump([lmax, self.hashdict(lmax, fwd_op), self.minv], open(cache_fname, 'w'))

    @staticmethod
    def hashdict(lmax, fwd_op):
        return {'lmax': lmax,
                'fwd_op': fwd_op.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return self.rlm2alm(np.dot(self.minv, self.alm2rlm(talm)))
