"""$N^{(1)}_L$ quadratic estimator bias calculation module.

This modules contains:
  * the $N^{(1)}_L$ bias calculation scripts (for lensing and other quadratic estimators)
  * the $N^{(1)}_{LL'}$ matrix derivative calculations with respect to the CMB power spectra.

All calculations are performed using the flat-sky approximation.

"""
from __future__ import print_function

import os

import numpy as np
import sympy
import pickle as pk
from scipy.interpolate import UnivariateSpline as spline
import weave

from plancklens2018.utils import hash_check, clhash
from plancklens2018.sql import npdb

#FIXME: a lot to clean in this file

estimator_keys = ['ptt', 'pte', 'pet', 'pee', 'peb', 'pbe', 'ptb', 'pbt',
                  'stt', 'mtt', 'ntt','ftt','dtt','xtt', 'xte', 'xet', 'xee', 'xeb', 'xbe', 'xtb', 'xbt']
estimator_keys_derived = ['p', 'p_te', 'p_eb', 'p_tb', 'p_teb', 'p_p', 'p_tp',
                          'f', 'f_te', 'f_eb', 'f_tb', 'f_teb', 'f_p', 'f_tp','ptt_bh_d',
                          'ptt_bh_m', 'ptt_bh_n', 'ptt_bh_s', 'ptt_bh_mn', 'ptt_bh_f', 'ftt_bh_p',
                          'x', 'x_te', 'x_eb', 'x_tb', 'x_teb', 'x_p', 'x_tp']


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
    elif k in ['p_p', 'x_p' , 'x_p']:
        g = k[0]
        ret = [('%see' % g, clo),
               ('%seb' % g, 2. * clo)]
    elif k in ['p_te', 'x_te', 'p_tb', 'x_tb', 'p_eb', 'x_eb']:
        ret = [(k[0] + k[2] + k[3], 0.5 * clo),(k[0] + k[3] + k[2], 0.5 * clo)]
    else:
        assert 0, k
    return ret


class library_n1:
    """
     Extension of Duncan N1 derivative matrix, allowing for larger Lphi, important at high L.
     This can be used to produce N1 for any cosmo spectra but fixed weiucial qe weights and fiters
     after precompuation of binned N1 matrices.
     """

    def __init__(self, lib_dir, cltt, clte, clee, lmaxphi=4096, dl=10, dL=10, lps=None):
        """
        lps is actually the range of lensing potential multipole in the integration,
        while Ls is the l1 integration in Eq. A.25.

        ftl: 1/ (cmbcl + noise/bl^2) etc.
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

        dlps = [(lps[1] - lps[0])]
        for i in range(1, len(lps) - 1):
            dlps.append(0.5 * (lps[i + 1] - lps[i - 1]))
        dlps.append(lps[-1] - lps[-2])

        self.dl = dl
        self.dL = dL

        self.lps = lps
        self.dlps = np.array(dlps)

        self.cltt = np.copy(cltt[:], order='C')
        self.clte = np.copy(clte[:], order='C')
        self.clee = np.copy(clee[:], order='C')

        self.lmaxphi = lmaxphi

        self.n1 = {}
        self.lmaxiip = None
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        if not os.path.exists(lib_dir + '/n1_hash.pk'):
            pk.dump(self.hashdict(), open(lib_dir + '/n1_hash.pk', 'w'))
        hash_check(self.hashdict(), pk.load(open(lib_dir + '/n1_hash.pk', 'r')))
        self.npdb = npdb(lib_dir + '/npdb.db')
        self.lib_dir = lib_dir
        self.zerokeys = ['pbb', 'xbb', 'sbb', 'sbe', 'seb']

    def hashdict(self):
        #FIXME: cldtt
        return {'cltt': clhash(self.cltt), 'clte': clhash(self.clte), 'clee': clhash(self.clee), 'dl': self.dl,
                'dL': self.dL, 'lmaxphi': self.lmaxphi}

    def AisB(self):
        return np.all([np.all(self.ftlA == self.ftlB), np.all(self.felA == self.felB), np.all(self.fblA == self.fblB)])

    def get_fal(self, id, a):
        assert id in ['A', 'B']
        assert a in ['t', 'e', 'b'], a
        if id == 'A':
            if a == 't': return self.ftlA
            if a == 'e': return self.felA
            if a == 'b': return self.fblA
        if id == 'B':
            if a == 't': return self.ftlB
            if a == 'e': return self.felB
            if a == 'b': return self.fblB
        assert 0

    def get_n1(self, kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, kB=None, ftlB=None, felB=None, fblB=None, ttl=None,
               clttfid = None,cltefid = None,cleefid = None,n1_flat=lambda ell: np.ones(len(ell)) * 1.):
        """

        Args:
            k_ind (str): anisotropy source key.
            kA (str): quadratic estimator key (first leg).
            kB (str): quadratic estimator key (second leg). (Optional, defaults to kA.)
        """
        if kB is None: kB = kA
        #FIXME:
        if kA[0] == 's' or kB[0] == 's':
            assert kA[0] == kB[0],'point source implented following DH gradient convention, you wd probably need to pick a sign there'
        if ftlB is None: ftlB = ftlA
        if felB is None: felB = felA
        if fblB is None: fblB = fblA


        self.clttfid = self.cltt if clttfid is None else np.copy(clttfid,order = 'C')
        self.cltefid = self.clte if cltefid is None else np.copy(cltefid,order = 'C')
        self.cleefid = self.clee if cleefid is None else np.copy(cleefid,order = 'C')

        lmin_ftl = np.min([np.where(fal > 0.)[0] for fal in [ftlA, felA, fblA, ftlB, felB, fblB]])
        lmax_ftl = np.max([np.where(fal > 0.)[-1] for fal in [ftlA, felA, fblA, ftlB, felB, fblB]])
        dl = self.dl
        dL = self.dL

        self.Ls = np.concatenate([np.arange(lmin_ftl, dL), np.arange(dL, 2 * dL, 2), np.arange(2 * dL, lmax_ftl, dL)])
        if self.Ls[-1] != lmax_ftl: self.Ls = np.concatenate([self.Ls, [lmax_ftl]])
        self.lmax_ftl = lmax_ftl
        self.lmin_ftl = lmin_ftl
        self.ftlA = np.copy(ftlA[:lmax_ftl + 1], order='C')
        self.felA = np.copy(felA[:lmax_ftl + 1], order='C')
        self.fblA = np.copy(fblA[:lmax_ftl + 1], order='C')

        self.ftlB = np.copy(ftlB[:lmax_ftl + 1], order='C')
        self.felB = np.copy(felB[:lmax_ftl + 1], order='C')
        self.fblB = np.copy(fblB[:lmax_ftl + 1], order='C')

        self.ttl = np.copy(ttl[:lmax_ftl + 1], order='C') if ttl is not None else None

        assert len(self.clttfid) > self.lmax_ftl
        assert len(self.cltefid) > self.lmax_ftl
        assert len(self.cleefid) > self.lmax_ftl
        assert self.ttl is None or len(self.ttl) > self.lmax_ftl

        if kA in estimator_keys and kB in estimator_keys:
            if kA in self.zerokeys or kB in self.zerokeys:
                return np.zeros(Lmax + 1, dtype=float)
            if kA < kB:
                return self.get_n1(kB, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=kA,
                                   clttfid=clttfid,cltefid=cltefid,cleefid=cleefid, ttl=ttl)

            id = 'splined_kA' + kA + '_kB' + kB + '_ind' + k_ind
            id += '_clpp' + clhash(cl_kind[:])
            id += '_ftlA' + clhash(ftlA)
            id += '_felA' + clhash(felA)
            id += '_fblA' + clhash(fblA)
            id += '_ftlB' + clhash(ftlB)
            id += '_felB' + clhash(felB)
            id += '_fblB' + clhash(fblB)
            if clttfid is not None: id += '_clttfid' + clhash(clttfid)
            if cltefid is not None: id += '_cltefid' + clhash(cltefid)
            if cleefid is not None: id += '_cleefid' + clhash(cleefid)
            if ttl is not None: id += '_ttl'+ clhash(self.ttl)
            if Lmax != 2048:
                id += '_Lmax%s'%Lmax

            if self.npdb.get(id) is None:
                ret = []
                ells = np.unique(
                    np.concatenate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], np.arange(1, Lmax + 1)[::10], [Lmax]]))
                for L in ells:
                    print("n1: doing L %s kA %s kB %s kind %s" % (L, kA, kB, k_ind))
                    ret.append(self._get_n1_L(L, cl_kind, kA=kA, kB=kB, k_ind=k_ind))
                    # self.npdb.add(id, np.array(ret))
                _ret = np.zeros(Lmax + 1)
                _ret[1:] = spline(ells, np.array(ret) * n1_flat(ells), s=0., ext='raise', k=3)(np.arange(1, Lmax + 1) * 1.) / n1_flat(np.arange(1, Lmax + 1) * 1.)
                self.npdb.add(id,_ret)
            return self.npdb.get(id)
            # return self.npdb.get(id)

        assert self.AisB(), 'not sure the est. breakdown is OK for non-identical legs'
        if (kA in estimator_keys_derived) and (kB in estimator_keys_derived):
            ret = 0.
            for (tk1, cl1) in _get_est_derived(kA, Lmax):
                for (tk2, cl2) in _get_est_derived(kB,  Lmax):
                    # tret = self.get_n1(L,clpp,tk1, kB=tk2,k_ind=k_ind)
                    tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB,clttfid=clttfid,cltefid=cltefid,cleefid=cleefid,
                                       kB=tk2, n1_flat=n1_flat, ttl=ttl)
                    tret *= cl1[:Lmax + 1]
                    tret *= cl2[:Lmax + 1]
                    ret += tret
            return ret
        elif (kA in estimator_keys_derived) and (kB in estimator_keys):
            ret = 0.
            for (tk1, cl1) in _get_est_derived(kA,  Lmax):
                # tret = self.get_n1(L,clpp,tk1, kB=kB,k_ind=k_ind)
                tret = self.get_n1(tk1, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=kB,clttfid=clttfid,cltefid=cltefid,cleefid=cleefid,
                                   n1_flat=n1_flat, ttl=ttl)
                tret *= cl1[:Lmax + 1]
                ret += tret
            return ret
        elif (kA in estimator_keys) and (kB in estimator_keys_derived):
            ret = 0.
            for (tk2, cl2) in _get_est_derived(kB,  Lmax):
                tret = self.get_n1(kA, k_ind, cl_kind, ftlA, felA, fblA, Lmax, ftlB=ftlB, felB=felB, fblB=fblB, kB=tk2,clttfid=clttfid,cltefid=cltefid,cleefid=cleefid,
                                   n1_flat=n1_flat, ttl=ttl)
                # tret = self.get_n1(L,clpp,kA, kB=tk2,k_ind=k_ind)
                tret *= cl2[:Lmax + 1]
                ret += tret
            return ret
        assert 0


    def _get_n1_L(self, L, cl_kind, kA='ptt', kB=None, k_ind='p'):
        """
        Anisotropy source
        """
        if kB is None: kB = kA
        assert kA not in self.zerokeys and kB not in self.zerokeys
        assert kA in estimator_keys and kB in estimator_keys
        assert len(cl_kind) > self.lmaxphi
        if kA in estimator_keys and kB in estimator_keys:
            if kA < kB:
                return self._get_n1_L(L, cl_kind, kA=kB, kB=kA, k_ind=k_ind)
            else:
                #FIXME: hack

                id = str(L) + 'kA' + kA + '_kB' + kB + '_ind' + k_ind
                id += '_clpp' + clhash(cl_kind[:])
                id += '_ftlA' + clhash(self.ftlA)
                id += '_felA' + clhash(self.felA)
                id += '_fblA' + clhash(self.fblA)
                id += '_ftlB' + clhash(self.ftlB)
                id += '_felB' + clhash(self.felB)
                id += '_fblB' + clhash(self.fblB)
                if self.ttl is not None: id += '_ttl' + clhash(self.ttl)
                if not np.all(self.clttfid == self.cltt): id += '_clttfid' + clhash(self.clttfid)
                if not np.all(self.cltefid == self.clte): id += '_cltefid' + clhash(self.cltefid)
                if not np.all(self.cleefid == self.clee): id += '_cleefid' + clhash(self.cleefid)

                if self.npdb.get(id) is None:
                    a1 = kA[-2]
                    a2 = kA[-1]
                    a3 = kB[-2]
                    a4 = kB[-1]
                    L1x = sympy.symbols('L1x')
                    L2x = sympy.symbols('L2x')
                    L3x = sympy.symbols('L3x')
                    L4x = sympy.symbols('L4x')
                    L1y = sympy.symbols('L1y')
                    L2y = sympy.symbols('L2y')
                    L3y = sympy.symbols('L3y')
                    L4y = sympy.symbols('L4y')
                    L1int = sympy.symbols('L1int')
                    L2int = sympy.symbols('L2int')
                    L3int = sympy.symbols('L3int')
                    L4int = sympy.symbols('L4int')
                    cos = sympy.symbols('cos')
                    sin = sympy.symbols('sin')
                    atan2 = sympy.symbols('atan2')

                    cltt = {}
                    cltt[L1int] = sympy.symbols('cltt[L1int]')
                    cltt[L2int] = sympy.symbols('cltt[L2int]')
                    cltt[L3int] = sympy.symbols('cltt[L3int]')
                    cltt[L4int] = sympy.symbols('cltt[L4int]')

                    cldtt = {}
                    cldtt[L1int] = sympy.symbols('cldtt[L1int]')
                    cldtt[L2int] = sympy.symbols('cldtt[L2int]')
                    cldtt[L3int] = sympy.symbols('cldtt[L3int]')
                    cldtt[L4int] = sympy.symbols('cldtt[L4int]')

                    clttfid = {}
                    clttfid[L1int] = sympy.symbols('clttfid[L1int]')
                    clttfid[L2int] = sympy.symbols('clttfid[L2int]')
                    clttfid[L3int] = sympy.symbols('clttfid[L3int]')
                    clttfid[L4int] = sympy.symbols('clttfid[L4int]')

                    cldttfid = {}
                    cldttfid[L1int] = sympy.symbols('cldttfid[L1int]')
                    cldttfid[L2int] = sympy.symbols('cldttfid[L2int]')
                    cldttfid[L3int] = sympy.symbols('cldttfid[L3int]')
                    cldttfid[L4int] = sympy.symbols('cldttfid[L4int]')

                    clte = {}
                    clte[L1int] = sympy.symbols('clte[L1int]')
                    clte[L2int] = sympy.symbols('clte[L2int]')
                    clte[L3int] = sympy.symbols('clte[L3int]')
                    clte[L4int] = sympy.symbols('clte[L4int]')

                    cltefid = {}
                    cltefid[L1int] = sympy.symbols('cltefid[L1int]')
                    cltefid[L2int] = sympy.symbols('cltefid[L2int]')
                    cltefid[L3int] = sympy.symbols('cltefid[L3int]')
                    cltefid[L4int] = sympy.symbols('cltefid[L4int]')

                    clee = {}
                    clee[L1int] = sympy.symbols('clee[L1int]')
                    clee[L2int] = sympy.symbols('clee[L2int]')
                    clee[L3int] = sympy.symbols('clee[L3int]')
                    clee[L4int] = sympy.symbols('clee[L4int]')

                    cleefid = {}
                    cleefid[L1int] = sympy.symbols('cleefid[L1int]')
                    cleefid[L2int] = sympy.symbols('cleefid[L2int]')
                    cleefid[L3int] = sympy.symbols('cleefid[L3int]')
                    cleefid[L4int] = sympy.symbols('cleefid[L4int]')

                    fal1 = {}
                    fal1[L1int] = sympy.symbols("fal1[L1int]")
                    fal2 = {}
                    fal2[L2int] = sympy.symbols("fal2[L2int]")
                    fal3 = {}
                    fal3[L3int] = sympy.symbols("fal3[L3int]")
                    fal4 = {}
                    fal4[L4int] = sympy.symbols("fal4[L4int]")

                    ttl = {}
                    ttl[L1int] = sympy.symbols("ttl[L1int]")
                    ttl[L2int] = sympy.symbols("ttl[L2int]")
                    ttl[L3int] = sympy.symbols("ttl[L3int]")
                    ttl[L4int] = sympy.symbols("ttl[L4int]")

                    def wt_ptt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltt[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y) + cltt[l2abs] * (
                            (l1x + l2x) * l2x + (l1y + l2y) * l2y))

                    def wt_pte(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l1abs] * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x))) * (
                            (l1x + l2x) * l1x + (l1y + l2y) * l1y) + clte[l2abs] * (
                                    (l1x + l2x) * l2x + (l1y + l2y) * l2y))

                    def wt_pet(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l2abs] * cos(2. * (atan2(l1y, l1x) - atan2(l2y, l2x))) * (
                            (l2x + l1x) * l2x + (l2y + l1y) * l2y) + clte[l1abs] * (
                                    (l2x + l1x) * l1x + (l2y + l1y) * l1y))

                    # def wt_ptb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                    #    return (clte[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_ptb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return ((clte[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x))))

                    def wt_pbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l2abs] * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wt_pee(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y) + clee[l2abs] * (
                        (l1x + l2x) * l2x + (l1y + l2y) * l2y)) * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_peb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_pbe(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l2abs] * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wt_pbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_xtt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltt[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) + cltt[l2abs] * (
                        -(l1y + l2y) * l2x + (l1x + l2x) * l2y))

                    def wt_xte(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l1abs] * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x))) * (
                        -(l1y + l2y) * l1x + (l1x + l2x) * l1y) + clte[l2abs] * (
                                -(l1y + l2y) * l2x + (l1x + l2x) * l2y))

                    def wt_xet(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l2abs] * cos(2. * (atan2(l1y, l1x) - atan2(l2y, l2x))) * (
                        -(l2y + l1y) * l2x + (l2x + l1x) * l2y) + clte[l1abs] * (
                                -(l2y + l1y) * l1x + (l2x + l1x) * l1y))

                    def wt_xtb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_xbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clte[l2abs] * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wt_xee(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) + clee[l2abs] * (
                        -(l1y + l2y) * l2x + (l1x + l2x) * l2y)) * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_xeb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wt_xbe(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clee[l2abs] * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wt_ftt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return cltt[l1abs] + cltt[l2abs]

                    def wt_dtt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return cldtt[l1abs] + cldtt[l2abs]

                    def wt_ntt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return ttl[l1abs] * ttl[l2abs]

                    def wt_xbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_stt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 1.0

                    def wt_ste(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_set(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_see(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_stb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_sbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wt_sbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    #------
                    def wtfid_ptt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clttfid[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y) + clttfid[l2abs] * (
                            (l1x + l2x) * l2x + (l1y + l2y) * l2y))

                    def wtfid_ftt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return clttfid[l1abs] + clttfid[l2abs]

                    def wtfid_ntt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return ttl[l1abs] * ttl[l2abs]

                    def wtfid_dtt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return cldttfid[l1abs] + cldttfid[l2abs]

                    def wtfid_pte(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l1abs] * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x))) * (
                            (l1x + l2x) * l1x + (l1y + l2y) * l1y) + cltefid[l2abs] * (
                                    (l1x + l2x) * l2x + (l1y + l2y) * l2y))

                    def wtfid_pet(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l2abs] * cos(2. * (atan2(l1y, l1x) - atan2(l2y, l2x))) * (
                            (l2x + l1x) * l2x + (l2y + l1y) * l2y) + cltefid[l1abs] * (
                                    (l2x + l1x) * l1x + (l2y + l1y) * l1y))

                    # def wt_ptb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                    #    return (clte[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_ptb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return ((cltefid[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x))))

                    def wtfid_pbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l2abs] * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wtfid_pee(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y) + cleefid[l2abs] * (
                        (l1x + l2x) * l2x + (l1y + l2y) * l2y)) * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_peb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l1abs] * ((l1x + l2x) * l1x + (l1y + l2y) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_pbe(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l2abs] * ((l2x + l1x) * l2x + (l2y + l1y) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wtfid_pbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_xtt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (clttfid[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) + clttfid[l2abs] * (
                        -(l1y + l2y) * l2x + (l1x + l2x) * l2y))

                    def wtfid_xte(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l1abs] * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x))) * (
                        -(l1y + l2y) * l1x + (l1x + l2x) * l1y) + cltefid[l2abs] * (
                                -(l1y + l2y) * l2x + (l1x + l2x) * l2y))

                    def wtfid_xet(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l2abs] * cos(2. * (atan2(l1y, l1x) - atan2(l2y, l2x))) * (
                        -(l2y + l1y) * l2x + (l2x + l1x) * l2y) + cltefid[l1abs] * (
                                -(l2y + l1y) * l1x + (l2x + l1x) * l1y))

                    def wtfid_xtb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_xbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cltefid[l2abs] * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wtfid_xee(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y) + cleefid[l2abs] * (
                        -(l1y + l2y) * l2x + (l1x + l2x) * l2y)) * cos(2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_xeb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l1abs] * (-(l1y + l2y) * l1x + (l1x + l2x) * l1y)) * sin(
                            2. * (atan2(l2y, l2x) - atan2(l1y, l1x)))

                    def wtfid_xbe(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return (cleefid[l2abs] * (-(l2y + l1y) * l2x + (l2x + l1x) * l2y)) * sin(
                            2. * (atan2(l1y, l1x) - atan2(l2y, l2x)))

                    def wtfid_xbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_stt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 1.0

                    def wtfid_ste(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_set(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_see(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_stb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_sbt(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    def wtfid_sbb(l1x, l2x, l1y, l2y, l1abs, l2abs):
                        return 0.0

                    pd = {'ptt': wt_ptt,'ftt':wt_ftt,'dtt':wt_dtt, 'ntt':wt_ntt,
                          'pte': wt_pte, 'pet': wt_pet,
                          'ptb': wt_ptb, 'pbt': wt_pbt,
                          'pee': wt_pee,
                          'peb': wt_peb, 'pbe': wt_pbe,
                          'pbb': wt_pbb, 'stt': wt_stt,
                          'xtt': wt_xtt,
                          'xte': wt_xte, 'xet': wt_xet,
                          'xtb': wt_xtb, 'xbt': wt_xbt,
                          'xee': wt_xee,
                          'xeb': wt_xeb, 'xbe': wt_xbe,
                          'xbb': wt_xbb}
                    pdfid = {'ptt': wtfid_ptt,'ftt': wtfid_ftt,'dtt':wtfid_dtt, 'ntt': wtfid_ntt,
                          'pte': wtfid_pte, 'pet': wtfid_pet,
                          'ptb': wtfid_ptb, 'pbt': wtfid_pbt,
                          'pee': wtfid_pee,
                          'peb': wtfid_peb, 'pbe': wtfid_pbe,
                          'pbb': wtfid_pbb, 'stt': wtfid_stt,
                          'xtt': wtfid_xtt,
                          'xte': wtfid_xte, 'xet': wtfid_xet,
                          'xtb': wtfid_xtb, 'xbt': wtfid_xbt,
                          'xee': wtfid_xee,
                          'xeb': wtfid_xeb, 'xbe': wtfid_xbe,
                          'xbb': wtfid_xbb}
                    wtA = pdfid[kA]
                    wtB = pdfid[kB]
                    wt13 = pd[k_ind + (a1 + a3).lower()]
                    wt14 = pd[k_ind + (a1 + a4).lower()]
                    wt23 = pd[k_ind + (a2 + a3).lower()]
                    wt24 = pd[k_ind + (a2 + a4).lower()]
                    # JC: wA, wB fiducial weight functions, not differentiated.
                    # Regarding Eq A.2: uses symmetry of weight functions -l1 -,l2 -> l1 l2 etc
                    # and L_1 = l1, L_2 = L - l1, L_3 = l_phi - l1, L_4 = - L - L_3 =  -L -l+phi + l1
                    term1 = (wtA(L1x, L2x, L1y, L2y, L1int, L2int) * wtB(L3x, L4x, L3y, L4y, L3int, L4int) *
                             wt13(L1x, L3x, L1y, L3y, L1int, L3int) * wt24(L2x, L4x, L2y, L4y, L2int, L4int) *
                             fal1[L1int] * fal2[L2int] * fal3[L3int] * fal4[L4int])
                    term2 = (wtA(L1x, L2x, L1y, L2y, L1int, L2int) * wtB(L4x, L3x, L4y, L3y, L4int, L3int) *
                             wt14(L1x, L3x, L1y, L3y, L1int, L3int) * wt23(L2x, L4x, L2y, L4y, L2int, L4int) *
                             fal1[L1int] * fal2[L2int] * fal3[L3int] * fal4[L4int])
                    terms = term1 + term2

                    def pow_to_mul(expr):
                        """ Convert integer powers in an expression to Muls, like a**2 => a*a.

                        http://stackoverflow.com/questions/14264431/expanding-algebraic-powers-in-python-sympy
                        """
                        pows = list(expr.atoms(sympy.Pow))
                        if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
                            raise ValueError("A power contains a non-integer exponent")
                        repl = zip(pows, (sympy.Mul(*[b] * e, evaluate=False) for b, e in
                                          (i.as_base_exp() for i in pows)))
                        # FIXME : for some reason exp.subs does not replace the Clte[L1int] ** 2 Clte[L4int] ** 2 ???
                        return expr.subs(repl), repl

                    terms_str, repl = pow_to_mul(terms)
                    terms_str = str(terms_str)
                    for old, new in repl:
                        terms_str = terms_str.replace(str(old), str(new))
                    support_code = """
                    #define max(a,b) (a>b ? a : b)"""

                    c_code = """
                    // input arguments.
                    //double *cltt, *clte, *clee, *clpp, *tal1, *tal2, *tal3, *tal4, *ret;
                    int bigL = int(L);

                    // temporary arguments.
                    double Lx, Ly, L1, L1x, L1y, L2, L2x, L2y, L3x, L3y, L3, L4x, L4y, L4, PhiLx, PhiLy, PhiL, dPh;
                    int L1int, L2int, L3int, L4int, PhiL_nphi, PhiL_nphi_ix;
                    int nphi, phiIx, PhiLix,PhiLi;
                    double phi, dphi, PhiL_phi_dphi, PhiL_phi, fac;
                    double r = 0.;
                    //printf("library_n1_dmat::get_n1_mats evaluating for L = %f", bigL);

                    Lx = bigL;
                    Ly = 0;
                    for (L1 = max(lmin_ftl,dL/2); L1<=lmax_ftl; L1 += int(dL)) {
                        //printf ("L1 = %d ", L1);
                        L1int = round(L1);
                        nphi = (2.*L1+1.);
                        if (L1>(3*dL)) {
                            nphi=2*round(0.5*L1/dL)+1;
                        }
                        dphi = 2.*M_PI/nphi;

                        for (phiIx=0; phiIx <= (nphi-1)/2; phiIx++) {
                            phi = dphi*phiIx;
                            L1x = L1*cos(phi);
                            L1y = L1*sin(phi);

                            L2x = Lx - L1x;
                            L2y = Ly - L1y;
                            L2  = sqrt(L2x*L2x + L2y*L2y);

                            //if ((L2<lmin_ftl) || (L2>lmax_ftl)) { continue; }
                            if ((L2>=lmin_ftl) && (L2<=lmax_ftl)) {
                                L2int = round(L2);
                                //integral over (Lphi,Lphi_angle) according to lps grid.
                                //PhiL is l1 - l1' in A.25 (?)
                                for (PhiLix = 0; PhiLix < int(nps); PhiLix++) {
                                    PhiL = lps[PhiLix];
                                    PhiLi = int(PhiL);
                                    dPh  = dlps[PhiLix];
    
                                    PhiL_nphi = (2*PhiL+1);
                                    if (PhiL>20) {
                                        PhiL_nphi = 2*round(0.5*PhiL_nphi/dPh)+1;
                                    }
                                    PhiL_phi_dphi = 2.*M_PI/PhiL_nphi;
    
                                    fac  = (PhiL_phi_dphi * PhiL * dPh) * (dphi * L1 * dL) / pow(2.*M_PI,4.) / 4.;
                                    if (phiIx != 0) {
                                        fac=fac*2; //integrate 0-Pi for phi_L1
                                    }
    
                                    for (PhiL_nphi_ix=-(PhiL_nphi-1)/2; PhiL_nphi_ix <= (PhiL_nphi-1)/2; PhiL_nphi_ix++) {
                                        PhiL_phi = PhiL_phi_dphi * PhiL_nphi_ix;
                                        PhiLx = PhiL*cos(PhiL_phi);
                                        PhiLy = PhiL*sin(PhiL_phi);
                                        // L3 is l1' Eq. A.25
                                        L3x = PhiLx - L1x;
                                        L3y = PhiLy - L1y;
                                        L3 = sqrt(L3x*L3x + L3y*L3y);
    
                                        if ((L3>=lmin_ftl) && (L3<=lmax_ftl)) {
                                            L3int = round(L3);
                                            // L4 should be l2' Eq. A.25
                                            L4x = -Lx - L3x;
                                            L4y = -Ly - L3y;
                                            L4  = sqrt(L4x*L4x + L4y*L4y);
    
                                            if ((L4>=lmin_ftl) && (L4<=lmax_ftl)) {
                                                L4int = round(L4);
                                                r += (""" + terms_str + """)*fac*clpp[PhiLi];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ret[0] = r;
                    """
                    fal1 = self.get_fal('A', kA[-2])
                    fal2 = self.get_fal('A', kA[-1])
                    fal3 = self.get_fal('B', kB[-2])
                    fal4 = self.get_fal('B', kB[-1])
                    nps = int(len(self.lps))
                    lps = self.lps
                    dlps = self.dlps
                    dL = self.dL
                    lmaxphi = int(self.lmaxphi)
                    lmax_ftl = int(self.lmax_ftl)
                    lmin_ftl = int(self.lmin_ftl)

                    clttfid = self.clttfid
                    cltefid = self.cltefid
                    cleefid = self.cleefid
                    cltt = self.cltt
                    clte = self.clte
                    clee = self.clee
                    clpp = np.copy(cl_kind[:lmaxphi + 1], order='C')
                    ret = np.copy(np.array([0.]), order='C')

                    if kA[0] == 'd' or kB[0] == 'd':
                        #FIXME:
                        import lpipe as lp
                        cldtt = np.copy(lp.cldust_2013(4096)[:], order='C')
                        cldttfid = np.copy(lp.cldust_2013(4096)[:], order='C')
                        weave.inline(c_code, ['L', 'lmaxphi', 'lmax_ftl', 'lmin_ftl', 'cltt', 'clte', 'clee',
                                          'clttfid', 'cltefid', 'cleefid', 'clpp','cldtt','cldttfid',
                                          'fal1', 'fal2', 'fal3', 'fal4',
                                          'ret', 'nps', 'lps', 'dlps', 'dL'], support_code=support_code)
                    elif kA[0] == 'n' or kB[0] == 'n':  # FIXME:
                        assert self.ttl is not None
                        ttl = np.copy(self.ttl, order='C')
                        weave.inline(c_code, ['L', 'lmaxphi', 'lmax_ftl', 'lmin_ftl', 'cltt', 'clte', 'clee',
                                          'clttfid', 'cltefid', 'cleefid', 'clpp',
                                          'fal1', 'fal2', 'fal3', 'fal4','ttl',
                                          'ret', 'nps', 'lps', 'dlps', 'dL'], support_code=support_code)

                    else:
                        weave.inline(c_code, ['L', 'lmaxphi', 'lmax_ftl', 'lmin_ftl', 'cltt', 'clte', 'clee',
                                          'clttfid', 'cltefid', 'cleefid', 'clpp',
                                          'fal1', 'fal2', 'fal3', 'fal4',
                                          'ret', 'nps', 'lps', 'dlps', 'dL'], support_code=support_code)
                    self.npdb.add(id, ret)
                    return ret[0]
                return self.npdb.get(id)[0]
        assert 0