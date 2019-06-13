"""$N^{(1)}_L$ quadratic estimator bias calculation module.

    This modules contains the $N^{(1)}_L$ bias calculation scripts (for lensing and other quadratic estimators)

All calculations are performed using the flat-sky approximation.

"""
from __future__ import print_function

import os
import numpy as np
import pickle as pk
from scipy.interpolate import UnivariateSpline as spline

from plancklens2018.utils import hash_check, clhash, cli
from plancklens2018.helpers import sql

try:
    from . import n1f
    HASN1F = True
except:
    HASN1F = False

estimator_keys = ['ptt', 'pte', 'pet', 'pee', 'peb', 'pbe', 'ptb', 'pbt',
                  'xtt', 'xte', 'xet', 'xee', 'xeb', 'xbe', 'xtb', 'xbt',
                  'stt', 'ftt']
estimator_keys_derived = ['p', 'p_p', 'p_tp' 
                          'f', 'f_p', 'f_tp' 
                          'x', 'x_p', 'x_tp']


def _get_est_derived(k, lmax):
    """ Estimator combinations with some weighting.

    Args:
        k (str): Quadratic esimator key.
        lmax (int): weights are given up to lmax.

    """
    clo = np.ones(lmax + 1, dtype=float)
    if k in ['p', 'x', 'f']:
        ret = [('%stt' % k, clo),
               ('%ste' % k, 2. * clo),
               ('%stb' % k, 2. * clo),
               ('%see' % k, clo),
               ('%seb' % k, 2. * clo)]
    elif k in ['p_tp', 'x_tp', 'f_tp']:
        g = k[0]
        ret = [('%stt' % g, clo),
               ('%see' % g, clo),
               ('%seb' % g, 2. * clo)]
    elif k in ['p_p', 'x_p', 'f_p']:
        g = k[0]
        ret = [('%see' % g, clo),
               ('%seb' % g, 2. * clo)]
    elif k in ['p_te', 'x_te', 'p_tb', 'x_tb', 'p_eb', 'x_eb']:
        ret = [(k[0] + k[2] + k[3], 0.5 * clo), (k[0] + k[3] + k[2], 0.5 * clo)]
    else:
        assert 0, k
    return ret


if not HASN1F:
    print("*** n1f.so fortran shared object did not load properly")
    print('*** try f2py -c -m n1f ./n1f.f90 --f90flags="-fopenmp" -lgomp from the command line in n1 directory')
    print("*** Now falling back on python2 weave implementation")
    from . import n1_weave
    library_n1 =  n1_weave.library_n1
else:
    class library_n1:
        """

        """
        def __init__(self, lib_dir, cltt, clte, clee, lmaxphi=2500, dL=10, lps=None):
            """

            """
            if lps is None:
                lps = [1]
                for l in range(2, 111, 10):
                    lps.append(l)
                for l in range(lps[-1] + 30, 580, 30):
                    lps.append(l)
                for l in range(lps[-1] + 100, lmaxphi // 2, 100):
                    lps.append(l)
                for l in range(lps[-1] + 300, lmaxphi, 300):
                    lps.append(l)
                if lps[-1] != lmaxphi:
                    lps.append(lmaxphi)
                lps = np.array(lps)

            self.dL = dL
            self.lps = lps

            self.cltt = cltt
            self.clte = clte
            self.clee = clee

            self.lmaxphi = lps[-1]

            self.n1 = {}
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(os.path.join(lib_dir, 'n1_hash.pk')):
                pk.dump(self.hashdict(), open(os.path.join(lib_dir, 'n1_hash.pk'), 'wb'), protocol=2)
            hash_check(self.hashdict(), pk.load(open(os.path.join(lib_dir, 'n1_hash.pk'), 'rb')))
            self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))
            self.fldb = sql.fldb(os.path.join(lib_dir, 'fldb.db'))

            self.lib_dir = lib_dir

        def hashdict(self):
            return {'cltt': clhash(self.cltt), 'clte': clhash(self.clte), 'clee': clhash(self.clee),
                    'dL': self.dL, 'lps': self.lps}

        def get_n1(self, kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, kB=None, ftlB=None, felB=None, fblB=None,
                   clttfid=None, cltefid=None, cleefid=None, n1_flat=lambda ell: np.ones(len(ell), dtype=float), sglLmode=True):
            """

            """
            if kB is None: kB = kA
            # FIXME:
            if kA[0] == 's' or kB[0] == 's':
                assert kA[0] == kB[0], 'point source implented following DH gradient convention, you wd probably need to pick a sign there'
            if ftlB is None: ftlB = ftlA
            if felB is None: felB = felA
            if fblB is None: fblB = fblA

            clttfid = self.cltt if clttfid is None else clttfid
            cltefid = self.clte if cltefid is None else cltefid
            cleefid = self.clee if cleefid is None else cleefid


            if kA in estimator_keys and kB in estimator_keys:
                if kA < kB:
                    return self.get_n1(kB, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=kA,
                                       clttfid=clttfid, cltefid=cltefid, cleefid=cleefid, n1_flat=n1_flat)

                idx = 'splined_kA' + kA + '_kB' + kB + '_ind' + k_ind
                idx += '_clpp' + clhash(cl_kind)
                idx += '_ftlA' + clhash(ftlA)
                idx += '_felA' + clhash(felA)
                idx += '_fblA' + clhash(fblA)
                idx += '_ftlB' + clhash(ftlB)
                idx += '_felB' + clhash(felB)
                idx += '_fblB' + clhash(fblB)
                idx += '_clttfid' + clhash(clttfid)
                idx += '_cltefid' + clhash(cltefid)
                idx += '_cleefid' + clhash(cleefid)
                idx += '_Lmax%s' % Lmax

                if self.npdb.get(idx) is None:
                    Ls = np.unique(np.concatenate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.arange(1, Lmax + 1)[::10], [Lmax]]))
                    if sglLmode:
                        n1L = np.zeros(len(Ls), dtype=float)
                        for i, L in enumerate(Ls):
                            print("n1: doing L %s kA %s kB %s kind %s" % (L, kA, kB, k_ind))
                            n1L[i] = (self._get_n1_L(L, kA, kB, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid, cltefid, cleefid))
                    else: # entire vector from f90 openmp call
                        lmin_ftlA = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [ftlA, felA, fblA]])
                        lmin_ftlB = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [ftlB, felB, fblB]])
                        n1L = n1f.n1(Ls, cl_kind, kA, kB, k_ind, self.cltt, self.clte, self.clee,
                                     clttfid, cltefid, cleefid,  ftlA, felA, fblA, ftlB, felB, fblB,
                                      lmin_ftlA, lmin_ftlB,  self.dL, self.lps)
                    ret = np.zeros(Lmax + 1)
                    ret[1:] =  spline(Ls, np.array(n1L) * n1_flat(Ls), s=0., ext='raise', k=3)(np.arange(1, Lmax + 1) * 1.)
                    ret[1:] *= cli(n1_flat(np.arange(1, Lmax + 1) * 1.))
                    self.npdb.add(idx, ret)
                return self.npdb.get(idx)

            assert  np.all([np.all(ftlA == ftlB), np.all(felA == felB), np.all(fblA == fblB)]), \
                    'check the est. breakdown is OK for non-identical legs'
            if (kA in estimator_keys_derived) and (kB in estimator_keys_derived):
                ret = 0.
                for (tk1, cl1) in _get_est_derived(kA, Lmax):
                    for (tk2, cl2) in _get_est_derived(kB, Lmax):
                        tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB,
                                           clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                           kB=tk2, n1_flat=n1_flat)
                        tret *= cl1[:Lmax + 1]
                        tret *= cl2[:Lmax + 1]
                        ret += tret
                return ret
            elif (kA in estimator_keys_derived) and (kB in estimator_keys):
                ret = 0.
                for (tk1, cl1) in _get_est_derived(kA, Lmax):
                    tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=kB,
                                       clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                       n1_flat=n1_flat)
                    tret *= cl1[:Lmax + 1]
                    ret += tret
                return ret
            elif (kA in estimator_keys) and (kB in estimator_keys_derived):
                ret = 0.
                for (tk2, cl2) in _get_est_derived(kB, Lmax):
                    tret = self.get_n1(kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=tk2,
                                       clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                       n1_flat=n1_flat)
                    tret *= cl2[:Lmax + 1]
                    ret += tret
                return ret
            assert 0

        def _get_n1_L(self, L, kA, kB, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid, cltefid, cleefid):
            if kB is None: kB = kA
            assert kA in estimator_keys and kB in estimator_keys
            assert len(cl_kind) > self.lmaxphi
            if kA in estimator_keys and kB in estimator_keys:
                if kA < kB:
                    return self._get_n1_L(L, kB, kA, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid, cltefid, cleefid)
                else:
                    lmin_ftlA = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [ftlA, felA, fblA]])
                    lmin_ftlB = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [ftlB, felB, fblB]])
                    lmax_ftl = np.max([len(fal) for fal in [ftlA, felA, fblA, ftlB, felB, fblB]]) - 1
                    assert len(clttfid) > lmax_ftl and len(self.cltt) > lmax_ftl
                    assert len(cltefid) > lmax_ftl and len(self.clte) > lmax_ftl
                    assert len(cleefid) > lmax_ftl and len(self.clee) > lmax_ftl

                    idx = str(L) + 'kA' + kA + '_kB' + kB + '_ind' + k_ind
                    idx += '_clpp' + clhash(cl_kind)
                    idx += '_ftlA' + clhash(ftlA)
                    idx += '_felA' + clhash(felA)
                    idx += '_fblA' + clhash(fblA)
                    idx += '_ftlB' + clhash(ftlB)
                    idx += '_felB' + clhash(felB)
                    idx += '_fblB' + clhash(fblB)
                    idx += '_clttfid' + clhash(clttfid)
                    idx += '_cltefid' + clhash(cltefid)
                    idx += '_cleefid' + clhash(cleefid)

                    if self.fldb.get(idx) is None:
                        n1_L = n1f.n1l(L, cl_kind, kA, kB, k_ind,
                                      self.cltt, self.clte, self.clee, clttfid, cltefid, cleefid,
                                      ftlA, felA, fblA, ftlB, felB, fblB,
                                      lmin_ftlA, lmin_ftlB,  self.dL, self.lps)
                        self.fldb.add(idx, n1_L)
                        return n1_L
                    return self.fldb.get(idx)
            assert 0
