r""":math:`N^{(1)}_L` quadratic estimator bias calculation module.

    This module contains the :math:`N^{(1)}_L` bias calculation scripts (for lensing and other quadratic estimators)

    All calculations are performed using the flat-sky approximation from Fortran code.
    The Fortran code implements Eq. A.3. of the 2018 Planck lensing paper https://arxiv.org/abs/1807.06210,
    with integration on :math:`\bf{\ell_1}` and the anisotropy source wavevector.

    Note:

        For composed estimators, the N1 for all pairs will be calculated and stored in the process

    Note:

        The input spectra are those used in QE weights and the CMB responses (e.g. :math:`\tilde C^{T\nabla T}_\ell`)


"""
from __future__ import print_function

import os
import numpy as np
import pickle as pk
from scipy.interpolate import UnivariateSpline as spline

from plancklens.utils import hash_check, clhash, cli
from plancklens.helpers import sql, mpi

try:
    from . import n1f as n1f
    HASN1F = True
except:
    HASN1F = False

estimator_keys = ['ptt', 'pte', 'pet', 'pee', 'peb', 'pbe', 'ptb', 'pbt',
                  'xtt', 'xte', 'xet', 'xee', 'xeb', 'xbe', 'xtb', 'xbt',
                  'stt', 'ftt']
estimator_keys_derived = ['p', 'p_p', 'p_tp', 'p_eb', 'p_te', 'p_tb',
                          'f', 'f_p', 'f_tp', 'f_eb', 'f_te', 'f_tb',
                          'x', 'x_p', 'x_tp', 'x_eb', 'x_te', 'x_tb']


def _calc_n1L_sTP(L, cl_kind, kA, kB, k_ind, cltt, clte, clee, clttw, cltew, cleew,
                  ftlA, felA, fblA, ftlB, felB, fblB, lminA, lminB, dL, lps):
    """Direct call to f90 code for independent T-P fitlering

    """
    return n1f.n1l(L, cl_kind, kA, kB, k_ind,  cltt, clte, clee, clttw, cltew, cleew,
                   ftlA, felA, fblA, ftlB, felB, fblB, lminA, lminB, dL, lps)

def _get_est_derived(k, lmax):
    r""" Estimator combinations with some weighting.

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
        ret = [(k.replace('_', ''),  2. * clo)]
    elif k in estimator_keys:
        ret = [k, clo]
    else:
        assert 0, k
    return ret


if not HASN1F:
    print("*** n1f.so fortran shared object did not load properly")
    print('*** try f2py -c -m n1f ./n1f.f90 --f90flags="-fopenmp" -lgomp from the command line in n1 directory ?')

class library_n1:
    r"""Flexible library for calculation of the N1 quadratic estimator biases

        Args:
            lib_dir: results will be stored there
            cltt: CMB TT spectrum (used for map CMB spectrum and QE weights)
            clte: CMB TE spectrum (used for map CMB spectrum and QE weights)
            clee: CMB EE spectrum (used for map CMB spectrum and QE weights)
            lmaxphi: maximum multipole of the anistropy source (clpp for standard lensing N1) to consider
            dL: flat-sky numerical integration parameter, see n1.f90
            lps: flat-sky numerical integration parameter, see n1.f90


    """
    def __init__(self, lib_dir, cltt, clte, clee, lmaxphi=2500, dL=10, lps=None):

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
        hash_check(self.hashdict(), pk.load(open(os.path.join(lib_dir, 'n1_hash.pk'), 'rb')), fn=os.path.join(lib_dir, 'n1_hash.pk'))
        self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))
        self.fldb = sql.fldb(os.path.join(lib_dir, 'fldb.db'))

        self.lib_dir = lib_dir

    def hashdict(self):
        return {'cltt': clhash(self.cltt), 'clte': clhash(self.clte), 'clee': clhash(self.clee),
                'dL': self.dL, 'lps': self.lps}

    def get_n1(self, kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, kB=None, ftlB=None, felB=None, fblB=None,
               clttfid=None, cltefid=None, cleefid=None, n1_flat=lambda ell: np.ones(len(ell), dtype=float),
               recache=False, remove_only=False, sglLmode=True):
        r"""Calls a N1 bias

            Args:
                kA: qe_key of QE spectrum first leg
                k_ind: anisotropy source key ('p', for standard lensing N1)
                cl_kind: spectrum of anisotropy source ('p', for standard lensing N1)
                ftlA: first leg T-filtering isotropic approximation
                      (typically :math:`\frac{1}{C_\ell^{TT} + N_\ell^{TT}}`)
                felA: first leg E-filtering isotropic approximation
                      (typically :math:`\frac{1}{C_\ell^{EE} + N_\ell^{EE}}`)
                fblA: first leg B-filtering isotropic approximation
                     (typically :math:`\frac{1}{C_\ell^{BB} + N_\ell^{BB}}`)
                Lmax: maximum multipole of output N1
                kB(optional): qe_key of QE spectrum second leg (if different from the first)
                ftlB(optional): second leg T-filtering isotropic approximation (if different from the first)
                felB(optional): second leg E-filtering isotropic approximation (if different from the first)
                fblB(optional): second leg B-filtering isotropic approximation (if different from the first)
                clttfid(optional): CMB TT spectrum used in QE weights (if different from instance cltt for map-level CMB spectrum)
                cltefid(optional): CMB TE spectrum used in QE weights (if different from instance clte for map-level CMB spectrum)
                cleefid(optional): CMB EE spectrum used in QE weights (if different from instance clee for map-level CMB spectrum)
                n1_flat(optional): function used to flatten the discretized output before returning splined entire array

            Returns:
                N1 bias in the form of a numpy array of size Lmax + 1

            Note:
                This can be called with MPI using a number of processes; in this case the calculations for each multipole will be distributed among these.

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
                return self.get_n1(kB, k_ind, cl_kind, ftlB, felB, fblB, Lmax, ftlB=ftlA, felB=felA, fblB=fblA, kB=kA,
                                   clttfid=clttfid, cltefid=cltefid, cleefid=cleefid, n1_flat=n1_flat, sglLmode=sglLmode)

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

            ret = self.npdb.get(idx)
            if ret is not None:
                if not recache and not remove_only:
                    return ret
                else:
                    self.npdb.remove(idx)
                    if remove_only:
                        return np.zeros_like(ret)
                    ret = None
            if ret is None:
                Ls = np.unique(np.concatenate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.arange(1, Lmax + 1)[::20], [Lmax]]))
                if sglLmode:
                    n1L = np.zeros(len(Ls), dtype=float)
                    for i, L in enumerate(Ls[mpi.rank::mpi.size]):
                        print("n1: rank %s doing L %s kA %s kB %s kind %s" % (mpi.rank, L, kA, kB, k_ind))
                        n1L[i] = (self._get_n1_L(L, kA, kB, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid, cltefid, cleefid, remove_only=remove_only))
                    if mpi.size > 1:
                        mpi.barrier()
                        for i, L in enumerate(Ls): # reoading cached n1L's
                            n1L[i] = (self._get_n1_L(L, kA, kB, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid,
                                             cltefid, cleefid, remove_only=remove_only))
                        mpi.barrier()

                else: # entire vector from f90 openmp call
                    lmin_ftlA = np.min([np.min(np.where(np.abs(fal) > 0.)[0]) for fal in [ftlA, felA, fblA]])
                    lmin_ftlB = np.min([np.min(np.where(np.abs(fal) > 0.)[0]) for fal in [ftlB, felB, fblB]])
                    n1L = n1f.n1(Ls, cl_kind, kA, kB, k_ind, self.cltt, self.clte, self.clee,
                                 clttfid, cltefid, cleefid,  ftlA, felA, fblA, ftlB, felB, fblB,
                                  lmin_ftlA, lmin_ftlB,  self.dL, self.lps)

                ret = np.zeros(Lmax + 1)
                ret[1:] =  spline(Ls, np.array(n1L) * n1_flat(Ls), s=0., ext='raise', k=3)(np.arange(1, Lmax + 1) * 1.)
                ret[1:] *= cli(n1_flat(np.arange(1, Lmax + 1) * 1.))
                self.npdb.add(idx, ret)
                return ret
            return self.npdb.get(idx)

        if (kA in estimator_keys_derived) and (kB in estimator_keys_derived):
            ret = 0.
            for (tk1, cl1) in _get_est_derived(kA, Lmax):
                for (tk2, cl2) in _get_est_derived(kB, Lmax):
                    tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB,
                                       clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                       kB=tk2, n1_flat=n1_flat, sglLmode=sglLmode)
                    tret *= cl1[:Lmax + 1]
                    tret *= cl2[:Lmax + 1]
                    ret += tret
            return ret
        elif (kA in estimator_keys_derived) and (kB in estimator_keys):
            ret = 0.
            for (tk1, cl1) in _get_est_derived(kA, Lmax):
                tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=kB,
                                   clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                   n1_flat=n1_flat, sglLmode=sglLmode)
                tret *= cl1[:Lmax + 1]
                ret += tret
            return ret
        elif (kA in estimator_keys) and (kB in estimator_keys_derived):
            ret = 0.
            for (tk2, cl2) in _get_est_derived(kB, Lmax):
                tret = self.get_n1(kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=tk2,
                                   clttfid=clttfid, cltefid=cltefid, cleefid=cleefid,
                                   n1_flat=n1_flat, sglLmode=sglLmode)
                tret *= cl2[:Lmax + 1]
                ret += tret
            return ret
        assert 0

    def _get_n1_L(self, L, kA, kB, k_ind, cl_kind, ftlA, felA, fblA, ftlB, felB, fblB, clttfid, cltefid, cleefid, remove_only=False):
        if kB is None: kB = kA
        assert kA in estimator_keys and kB in estimator_keys
        assert len(cl_kind) > self.lmaxphi
        if kA in estimator_keys and kB in estimator_keys:
            if kA < kB:
                return self._get_n1_L(L, kB, kA, k_ind, cl_kind, ftlB, felB, fblB, ftlA, felA, fblA, clttfid, cltefid, cleefid)
            else:
                lmin_ftlA = np.min([np.where(np.abs(fal) > 0.)[0][0] for fal in [ftlA, felA, fblA]])
                lmin_ftlB = np.min([np.where(np.abs(fal) > 0.)[0][0] for fal in [ftlB, felB, fblB]])
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

                n1_L = self.fldb.get(idx)

                if n1_L is None:
                    if remove_only:
                        return 0.
                    n1_L = n1f.n1l(L, cl_kind, kA, kB, k_ind,
                                  self.cltt, self.clte, self.clee, clttfid, cltefid, cleefid,
                                  ftlA, felA, fblA, ftlB, felB, fblB,
                                  lmin_ftlA, lmin_ftlB,  self.dL, self.lps)
                    self.fldb.add(idx, n1_L)
                    return n1_L
                else:
                    if remove_only:
                        self.fldb.remove(idx)
                        return 0.
                    return n1_L
        assert 0

    def get_n1_jtp(self, kA, k_ind, cl_kind, fAlmat, Lmax, kB=None, fBlmat=None,
            clttfid=None, cltefid=None, cleefid=None, n1_flat=lambda ell: np.ones(len(ell), dtype=float)):

        if kB is None: kB = kA
        # FIXME:
        if kA[0] == 's' or kB[0] == 's':
            assert kA[0] == kB[0], 'point source implented following DH gradient convention, you wd probably need to pick a sign there'
        if fBlmat is None: fBlmat = fAlmat

        clttfid = self.cltt if clttfid is None else clttfid
        cltefid = self.clte if cltefid is None else cltefid
        cleefid = self.clee if cleefid is None else cleefid


        if kA in estimator_keys and kB in estimator_keys:
            if kA < kB:
                return self.get_n1_jtp(kB, k_ind, cl_kind, fBlmat, Lmax, fBlmat=fAlmat, kB=kA,
                                   clttfid=clttfid, cltefid=cltefid, cleefid=cleefid, n1_flat=n1_flat)


            X, Y = kA[1:]
            I, J = kB[1:]
            assert np.all(i in ['t', 'e', 'b'] for i in [X, Y, I, J]),  [X, Y, I, J]
            ret = 0.
            for Xp in ['t', 'e', 'b']:
                FXXp = fAlmat.get(X + Xp, fAlmat.get(Xp + X, [0.]))
                if np.any(FXXp):
                    for Yp in ['t', 'e', 'b']:
                        FYYp = fAlmat.get(Y + Yp, fAlmat.get(Yp + Y, [0.]))
                        if np.any(FYYp):
                            for Ip in ['t', 'e', 'b']:
                                FIIp = fBlmat.get(I + Ip, fBlmat.get(Ip + I, [0.]))
                                if np.any(FIIp):
                                    for Jp in ['t', 'e', 'b']:
                                        FJJp = fBlmat.get(J + Jp, fBlmat.get(Jp + J, [0.]))
                                        if np.any(FJJp):
                                            idx = 'splined_' + X + Xp + Y + Yp + I + Ip + J + Jp
                                            idx += '_clpp' + clhash(cl_kind)
                                            idx += '_fXXp' + clhash(FXXp)
                                            idx += '_fYYp' + clhash(FYYp)
                                            idx += '_fIIp' + clhash(FIIp)
                                            idx += '_fJJp' + clhash(FJJp)
                                            idx += '_clttfid' + clhash(clttfid)
                                            idx += '_cltefid' + clhash(cltefid)
                                            idx += '_cleefid' + clhash(cleefid)
                                            idx += '_Lmax%s' % Lmax

                                            if self.npdb.get(idx) is None:
                                                Ls = np.unique(np.concatenate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.arange(1, Lmax + 1)[::20], [Lmax]]))
                                                n1L = np.zeros(len(Ls), dtype=float)
                                                for i, L in enumerate(Ls):
                                                    print("n1: doing L %s kA %s kB %s kind %s " % (L, kA, kB, k_ind)  + Xp + Yp + Ip + Jp)
                                                    n1L[i] = (self._get_n1_L_jtp(L, kA, kB, k_ind, cl_kind, Xp, Yp, Ip, Jp, fAlmat, fBlmat, clttfid, cltefid, cleefid))
                                                ret = np.zeros(Lmax + 1)
                                                ret[1:] =  spline(Ls, np.array(n1L) * n1_flat(Ls), s=0., ext='raise', k=3)(np.arange(1, Lmax + 1) * 1.)
                                                ret[1:] *= cli(n1_flat(np.arange(1, Lmax + 1) * 1.))
                                                self.npdb.add(idx, ret)
                                            ret = ret +  self.npdb.get(idx)
            return ret
        if (kA in estimator_keys_derived) or (kB in estimator_keys_derived):
            ret = 0.
            for (tk1, cl1) in _get_est_derived(kA, Lmax):
                for (tk2, cl2) in _get_est_derived(kB, Lmax):
                    tret = self.get_n1_jtp(tk1, k_ind, cl_kind, fAlmat, Lmax, kB=tk2, fBlmat=fBlmat,
                                    clttfid=clttfid, cltefid=cltefid, cleefid=cleefid, n1_flat=n1_flat)
                    ret = ret + tret * cl1[:Lmax + 1] * cl2[:Lmax + 1]
            return ret
        assert 0

    def _get_n1_L_jtp(self, L, kA, kB, k_ind, cl_kind, Xp, Yp, Ip, Jp, fAlmat, fBlmat, clttfid, cltefid, cleefid):
            if kB is None: kB = kA
            if kA in estimator_keys and kB in estimator_keys:
                if kA < kB:
                    assert 0, 'fix this'
                else:
                    X, Y = kA[1:]
                    I, J = kB[1:]
                    FXXp = fAlmat.get(X + Xp, fAlmat.get(Xp + X, None))
                    if FXXp is None: return 0.

                    FYYp = fAlmat.get(Y + Yp, fAlmat.get(Yp + Y, None))
                    if FYYp is None: return 0.

                    FIIp = fBlmat.get(I + Ip, fBlmat.get(Ip + I, None))
                    if FIIp is None: return 0.

                    FJJp = fBlmat.get(J + Jp, fBlmat.get(Jp + J, None))
                    if FJJp is None: return 0.

                    lmax_ftl = np.max([FXXp.size, FYYp.size, FIIp.size, FJJp.size]) - 1
                    lmin_ftlA = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [FXXp, FYYp]])
                    lmin_ftlB = np.min([np.where(np.abs(fal) > 0.)[0] for fal in [FIIp, FJJp]])
                    assert len(clttfid) > lmax_ftl and len(self.cltt) > lmax_ftl
                    assert len(cltefid) > lmax_ftl and len(self.clte) > lmax_ftl
                    assert len(cleefid) > lmax_ftl and len(self.clee) > lmax_ftl
                    assert (FXXp.size == FYYp.size) and (FIIp.size == FJJp.size)
                    assert len(cl_kind) > self.lmaxphi

                    idx = str(L)  + X + Xp + Y + Yp + I + Ip + J + Jp
                    idx += '_clpp' + clhash(cl_kind)
                    idx += '_fXXp' + clhash(FXXp)
                    idx += '_fYYp' + clhash(FYYp)
                    idx += '_fIIp' + clhash(FIIp)
                    idx += '_fJJp' + clhash(FJJp)
                    idx += '_clttfid' + clhash(clttfid)
                    idx += '_cltefid' + clhash(cltefid)
                    idx += '_cleefid' + clhash(cleefid)
                    # n1L_jtp(L, cl_kI, kA, kB, XpIp, YpJp, kI, cltt, clte, clee, clttfid, cltefid,
                    #        cleefid, &
                    # fXXp, fYYp, fIIp, fJJp, lminA, lmaxA, lminB, lmaxB, lmaxI, &
                    # lmaxtt, lmaxte, lmaxee, lmaxttfid, lmaxtefid, lmaxeefid, dL, lps, nlps)
                    if self.fldb.get(idx) is None:
                        n1_L = n1f.n1l_jtp(L, cl_kind, kA, kB, Xp, Yp, Ip, Jp, k_ind,
                                       self.cltt, self.clte, self.clee, clttfid, cltefid, cleefid,
                                       FXXp, FYYp, FIIp, FJJp,
                                       lmin_ftlA, lmin_ftlB, self.dL, self.lps)
                        self.fldb.add(idx, n1_L)
                        return n1_L
                    return self.fldb.get(idx)
            assert 0
