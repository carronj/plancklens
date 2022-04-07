"""Quadratic estimation implementation module.


"""
from __future__ import print_function
from __future__ import absolute_import
import healpy as hp
import numpy as np
import os
import pickle as pk
import collections

from plancklens import utils as ut, utils_qe as uqe
from plancklens.helpers import mpi
from plancklens import qresp

_write_alm = lambda fn, alm : hp.write_alm(fn, alm, overwrite=True)

def eval_qe(qe_key, lmax_ivf, cls_weight, get_alm, nside, lmax_qlm, verbose=True):
    """Evaluates a quadratic estimator gradient and curl terms.

        (see 'library' below for QE estimation coupled to CMB inverse-variance filtered simulation libraries,
        whose implementation can be faster for some estimators.)

        Args:
            qe_key: QE key defining the estimator (as defined in the qresp module), e.g. 'ptt' for lensing TT estimator
            lmax_ivf: CMB multipoles up to lmax are used in the QE
            cls_weight: set of CMB spectra entering the QE estimator weights
            get_alm: callable with 't', 'e', 'b' arguments, returning the corresponding inverse-variance filtered CMB map
            nside: the estimator are calculated in position space at healpy resolution nside.
            lmax_qlm: gradient and curl terms are obtained up to multipole lmax_qlm.

        Returns:
            glm and clm healpy arrays (gradient and curl terms of the QE estimate)

    """
    qe_list = qresp.get_qes(qe_key, lmax_ivf, cls_weight)
    return uqe.qe_eval(qe_list, nside, get_alm, lmax_qlm, verbose=verbose)


def library_jtTP(lib_dir, ivfs1, ivfs2, nside, lmax_qlm=None, resplib=None):
    return library(lib_dir, ivfs1, ivfs2, nside, lmax_qlm=lmax_qlm, resplib=resplib)


def library_sepTP(lib_dir, ivfs1, ivfs2, clte, nside, lmax_qlm=None, resplib=None):
    return library(lib_dir, ivfs1, ivfs2, nside, clte=clte, lmax_qlm=lmax_qlm, resplib=resplib)


class library:
    r"""From two inverse-variance filtered CMB simulation libraries returns a QE evaluation library.

        Args:
            lib_dir: QE estimates will be stored there.
            ivfs1: inverse-variance filtering instance of the QE 1st leg.
            ivfs2: inverse-variance filtering instance of the QE 2nd leg.
            nside: QE estimates are obtained from real-space healpy maps at resolution nside
            clte(optional): TE CMB spectrum weight. If set this is used to build :math:`X^{\rm WF}` from :math:`\bar X`.
                            Defaults to None, which is adequate for the MV estimator if T and P maps are jointly filtered.
            lmax_qlm(optional): QE estimates are computed up to multipole lmax_qlm (defaults to 3 * nside -1).
            resplib(optional): response library with *get_response* methods. Only used for bias_hardened estimators.

    """
    def __init__(self, lib_dir, ivfs1, ivfs2, nside, clte=None, lmax_qlm=None, resplib=None):
        if lmax_qlm is None:
            lmax_qlm = 3 * nside -1
        self.lib_dir = lib_dir
        self.prefix = lib_dir
        self.lmax_qlm = {'T': lmax_qlm, 'P': lmax_qlm, 'PS': lmax_qlm}
        if clte is None:
            self.f2map1 = lib_filt2map(ivfs1, nside)
            self.f2map2 = lib_filt2map(ivfs2, nside)
        else:
            self.f2map1 = lib_filt2map_sepTP(ivfs1, nside, clte)
            self.f2map2 = lib_filt2map_sepTP(ivfs2, nside, clte)
        assert self.lmax_qlm['T'] == self.lmax_qlm['P'], 'implement this'
        fnhash = os.path.join(self.lib_dir, "qe_sim_hash.pk")
        if (mpi.rank == 0) and (not os.path.exists(fnhash)):
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
            pk.dump(self.hashdict(), open(fnhash, 'wb'), protocol=2)
        mpi.barrier()

        ut.hash_check(pk.load(open(fnhash, 'rb')), self.hashdict())
        if mpi.rank == 0:
            if not os.path.exists(os.path.join(lib_dir, 'fskies.dat')):
                print("Caching sky fractions...")
                ms = {1: self.get_mask(1), 2: self.get_mask(2)}
                fskies = {}
                for i in [1, 2]:
                    for j in [1, 2][i - 1:]:
                        fskies[10 * i + j] = np.mean(ms[i] * ms[j])
                with open(os.path.join(lib_dir, 'fskies.dat'), 'w') as f:
                    for lab, _f in zip(np.sort(list(fskies.keys())),
                                       np.array(list(fskies.values()))[np.argsort(list(fskies.keys()))]):
                        f.write('%4s %.5f \n' % (lab, _f))
        mpi.barrier()
        fskies = {}
        with open(os.path.join(lib_dir, 'fskies.dat')) as f:
            for line in f:
                (key, val) = line.split()
                fskies[int(key)] = float(val)
        self.fskies = fskies
        self.fsky11 = fskies[11]
        self.fsky12 = fskies[12]
        self.fsky22 = fskies[22]

        self.resplib = resplib

        self.keys_fund = ['ptt', 'xtt', 'p_p', 'x_p', 'p', 'x', 'stt', 'ftt','f_p', 'f','dtt', 'ntt', 'a_p',
                          'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                          'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']
        self.keys = self.keys_fund + ['p_tp', 'x_tp', 'p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb', 'ptt_bh_n',
                                      'ptt_bh_s', 'ptt_bh_f', 'ptt_bh_d', 'dtt_bh_p', 'stt_bh_p', 'ftt_bh_d']

    def hashdict(self):
        return {'f2map1': self.f2map1.hashdict(),
                'f2map2': self.f2map2.hashdict()}

    def get_fundkeys(self, k_list):
        if not isinstance(k_list, list):
            _klist = [k_list]
        else:
            _klist = k_list
        ret = []
        for k in _klist:
            assert k in self.keys, (k, self.keys)
            if k in self.keys_fund:
                ret.append(k)
            elif '_tp' in k:
                ret.append(k[0] + 'tt')
                ret.append(k[0] + '_p')
            elif 'tt_bh_' in k:
                l, f = k.split('_bh_')
                ret.append(l)
                ret.append(f + 'tt')
            elif k in ['p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb']:
                ret.append(k[0] + k[2] + k[3])
                ret.append(k[0] + k[3] + k[2])
        return list(collections.OrderedDict.fromkeys(ret))

    def get_fsky(self, id):
        assert id in [11, 22, 12], id
        return self.fskies[id]

    def get_lmax_qlm(self, k):
        assert self.lmax_qlm['T'] == self.lmax_qlm['P']
        return self.lmax_qlm['T']

    def get_mask(self, leg):
        assert leg in [1, 2]
        return self.f2map1.ivfs.get_fmask() if leg == 1 else self.f2map2.ivfs.get_fmask()

    def get_sim_qlm(self, k, idx, lmax=None):
        """Returns a QE estimate, by computing and caching it if not done previously.

            Args:
                k: quadratic estimator key
                idx: simulation index
                lmax: optionally reduces the lmax of the output healpy array.

        """
        assert k in self.keys, (k, self.keys)
        if lmax is None :
            lmax = self.get_lmax_qlm(k)
        assert lmax <= self.get_lmax_qlm(k)
        if k in ['p_tp', 'x_tp', 'f_tp', 's_tp']:
            return self.get_sim_qlm('%stt' % k[0], idx, lmax=lmax) + self.get_sim_qlm('%s_p' % k[0], idx, lmax=lmax)
        if k in ['p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb']:
            return self.get_sim_qlm(k[0]+k[2]+k[3], idx, lmax=lmax) + self.get_sim_qlm(k[0]+k[3]+k[2], idx, lmax=lmax)

        if '_bh_' in k: # Bias-hardening
            assert self.resplib is not None, 'resplib arg necessary for this'
            kQE, ksource = k.split('_bh_')
            assert len(ksource) == 1 and ksource + kQE[1:] in self.keys, (ksource, kQE)
            assert self.get_lmax_qlm(kQE) == self.get_lmax_qlm(ksource + kQE[1:]), 'fix this (easy)'
            lmax = self.get_lmax_qlm(kQE)
            wL = self.resplib.get_response(kQE, ksource) * ut.cli(self.resplib.get_response(ksource + kQE[1:], ksource))
            ret = self.get_sim_qlm(kQE, idx, lmax=lmax)
            return ret- hp.almxfl(self.get_sim_qlm(ksource + kQE[1:], idx, lmax=lmax), wL)

        assert k in self.keys_fund, (k, self.keys_fund)
        fname = os.path.join(self.lib_dir, 'sim_%s_%04d.fits'%(k, idx) if idx != -1 else 'dat_%s.fits'%k)
        if not os.path.exists(fname):
            if k in ['ptt', 'xtt']: self._build_sim_Tgclm(idx)
            elif k in ['p_p', 'x_p']: self._build_sim_Pgclm(idx)
            elif k in ['p', 'x']: self._build_sim_MVgclm(idx)
            elif k in ['f']: self._build_sim_f(idx)
            elif k in ['stt']: self._build_sim_stt(idx)
            elif k in ['ftt']: self._build_sim_ftt(idx)
            elif k in ['f_p']: self._build_sim_f_p(idx)
            elif k in ['ntt']: self._build_sim_ntt(idx)
            elif k in ['a_p']: self._build_sim_a_p(idx)
            elif k in ['ptt', 'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                       'xtt', 'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']:
                self._build_sim_xfiltMVgclm(idx, k)
            else:
                assert 0, k

        return  ut.alm_copy(hp.read_alm(fname), lmax=lmax)

    def get_dat_qlm(self, k, **kwargs):
        return self.get_sim_qlm(k, -1, **kwargs)

    def get_sim_qlm_mf(self, k, mc_sims, lmax=None):
        """Returns a QE mean-field estimate, by averaging QE estimates from a set simulations (caches the result).

            Args:
                k: quadratic estimator key
                mc_sims: simulation indices to use for the estimate.
                lmax: optionally reduces the lmax of the output healpy array.

        """
        if lmax is None:
            lmax = self.get_lmax_qlm(k)
        assert lmax <= self.get_lmax_qlm(k)
        if k in ['p_tp', 'x_tp']:
            return (self.get_sim_qlm_mf('%stt' % k[0], mc_sims, lmax=lmax)
                    + self.get_sim_qlm_mf('%s_p' % k[0], mc_sims, lmax=lmax))
        if k in ['p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb']:
            return  self.get_sim_qlm_mf(k[0] + k[2] + k[3], mc_sims, lmax=lmax)  \
                    + self.get_sim_qlm_mf(k[0] + k[3] + k[2], mc_sims, lmax=lmax)
        if '_bh_' in k: # Bias-hardening
            assert self.resplib is not None, 'resplib arg necessary for this'
            kQE, ksource = k.split('_bh_')
            assert len(ksource) == 1 and ksource + kQE[1:] in self.keys, (ksource, kQE)
            assert self.get_lmax_qlm(kQE) == self.get_lmax_qlm(ksource + kQE[1:]), 'fix this (easy)'
            lmax = self.get_lmax_qlm(kQE)
            wL = self.resplib.get_response(kQE, ksource) * ut.cli(self.resplib.get_response(ksource + kQE[1:], ksource))
            ret = self.get_sim_qlm_mf(kQE, mc_sims, lmax=lmax)
            return ret- hp.almxfl(self.get_sim_qlm_mf(ksource + kQE[1:], mc_sims, lmax=lmax), wL)

        assert k in self.keys_fund, (k, self.keys_fund)
        fname = os.path.join(self.lib_dir, 'simMF_k1%s_%s.fits' % (k, ut.mchash(mc_sims)))
        if not os.path.exists(fname):
            this_mcs = np.unique(mc_sims)
            MF = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
            if len(this_mcs) == 0: return MF
            for i, idx in ut.enumerate_progress(this_mcs, label='calculating %s MF' % k):
                MF += self.get_sim_qlm(k, idx, lmax=lmax)
            MF /= len(this_mcs)
            _write_alm(fname, MF)
            print("Cached ", fname)
        return ut.alm_copy(hp.read_alm(fname), lmax=lmax)

    def _get_sim_Tgclm(self, idx, k, swapped=False, xfilt1=None, xfilt2=None):
        """ T only lensing potentials estimators """
        f2map1 = self.f2map1 if not swapped else self.f2map2
        f2map2 = self.f2map2 if not swapped else self.f2map1
        xftl1 = xfilt1 if not swapped else xfilt2
        xftl2 = xfilt2 if not swapped else xfilt1
        tmap = f2map1.get_irestmap(idx, xfilt=xftl1)  # healpy map
        G, C = f2map2.get_gtmap(idx, k=k, xfilt=xftl2)   # 2 healpy maps
        G *= tmap
        C *= tmap
        del tmap
        G, C = hp.map2alm_spin([G, C], 1, lmax=self.lmax_qlm['T'])
        fl = - np.sqrt(np.arange(self.lmax_qlm['T'] + 1, dtype=float) * np.arange(1, self.lmax_qlm['T'] + 2))
        hp.almxfl(G, fl, inplace=True)
        hp.almxfl(C, fl, inplace=True)
        return G, C

    def _get_sim_Pgclm(self, idx, k, swapped=False, xfilt1=None, xfilt2=None):
        """
        Pol. only lensing potentials estimators
        """
        f2map1 = self.f2map1 if not swapped else self.f2map2
        f2map2 = self.f2map2 if not swapped else self.f2map1
        xftl1 = xfilt1 if not swapped else xfilt2
        xftl2 = xfilt2 if not swapped else xfilt1
        repmap, impmap = f2map1.get_irespmap(idx, xfilt=xftl1)
        # complex spin 2 healpy maps
        Gs, Cs = f2map2.get_gpmap(idx, 3, k=k, xfilt=xftl2)  # 2 healpy maps
        GC = (repmap - 1j * impmap) * (Gs + 1j * Cs)  # (-2 , +3)
        Gs, Cs = f2map2.get_gpmap(idx, 1, k=k, xfilt=xftl2)
        GC -= (repmap + 1j * impmap) * (Gs - 1j * Cs)  # (+2 , -1)
        del repmap, impmap, Gs, Cs
        G, C = hp.map2alm_spin([GC.real, GC.imag], 1, lmax=self.lmax_qlm['P'])
        del GC
        fl = - np.sqrt(np.arange(self.lmax_qlm['P'] + 1, dtype=float) * np.arange(1, self.lmax_qlm['P'] + 2))
        hp.almxfl(G, fl, inplace=True)
        hp.almxfl(C, fl, inplace=True)
        return G, C

    def _get_sim_stt(self, idx, swapped=False):
        """Point source estimator """
        tmap1 = self.f2map1.get_irestmap(idx) if not swapped else self.f2map2.get_irestmap(idx)  # healpy map
        tmap1 *= (self.f2map2.get_irestmap(idx) if not swapped else self.f2map1.get_irestmap(idx))  # healpy map
        return -0.5 * hp.map2alm(tmap1, lmax=self.get_lmax_qlm('PS'), iter=0)

    def _get_sim_ntt(self, idx, swapped=False):
        """ Noise inhomogeneity estimator (same as point-source estimator but acting on beam-deconvolved maps) """
        f1 = self.f2map1 if not swapped else self.f2map2
        f2 = self.f2map2 if not swapped else self.f2map1
        tmap1 = f1.get_wirestmap(idx, f1.ivfs.get_tal('t')[:]) * f2.get_wirestmap(idx, f2.ivfs.get_tal('t')[:])
        return -0.5 * hp.map2alm(tmap1, lmax=self.get_lmax_qlm('T'), iter=0)

    def _get_sim_ftt(self, idx, joint=False, swapped=False):
        """Modulation estimator, temperature only."""
        tmap1 = self.f2map1.get_irestmap(idx) if not swapped else self.f2map2.get_irestmap(idx)  # healpy map
        tmap1 *= (self.f2map2.get_tmap(idx, joint=joint) if not swapped else self.f2map1.get_tmap(idx, joint=joint))  # healpy map
        return -hp.map2alm(tmap1, lmax=self.get_lmax_qlm('T'), iter=0)

    def _get_sim_f_p(self, idx, joint=False, swapped=False):
        """Modulation estimator, polarization only. """
        Q1, U1 = self.f2map1.get_irespmap(idx) if not swapped else self.f2map2.get_irespmap(idx)
        Q2, U2 = (self.f2map2.get_pmap(idx, joint=joint) if not swapped else self.f2map1.get_pmap(idx, joint=joint))
        return -2 * hp.map2alm(Q1 * Q2 + U1 * U2 , lmax=self.get_lmax_qlm('P'), iter=0)

    def _get_sim_a_p(self, idx, joint=False, swapped=False):
        """Polarization rotation estimator. """
        Q1, U1 = self.f2map1.get_irespmap(idx) if not swapped else self.f2map2.get_irespmap(idx)
        Q2, U2 = (self.f2map2.get_pmap(idx, joint=joint) if not swapped else self.f2map1.get_pmap(idx, joint=joint))
        return -4. * hp.map2alm(Q1 * U2 -  U1 * Q2 , lmax=self.get_lmax_qlm('P'), iter=0)

    def _get_sim_MVgclm(self, idx, k, swapped=False):
        assert k == 'p'
        GP, CP = self._get_sim_Pgclm(idx, 'p', swapped=swapped)
        GT, CT = self._get_sim_Tgclm(idx, 'p', swapped=swapped)
        return GP + GT, CP + CT

    def _build_sim_Tgclm(self, idx):
        """ T only lensing potentials estimators """
        G, C = self._get_sim_Tgclm(idx, 'ptt')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Tgclm(idx, 'ptt', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        _write_alm(os.path.join(self.lib_dir, 'sim_ptt_%04d.fits'%idx if idx != -1 else 'dat_ptt.fits'), G)
        _write_alm(os.path.join(self.lib_dir, 'sim_xtt_%04d.fits'%idx if idx != -1 else 'dat_xtt.fits'), C)


    def _build_sim_Pgclm(self, idx):
        """ Pol. only lensing potentials estimators """
        G, C = self._get_sim_Pgclm(idx, 'p_p')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Pgclm(idx, 'p_p', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        _write_alm(os.path.join(self.lib_dir, 'sim_p_p_%04d.fits'%idx if idx != -1 else 'dat_p_p.fits'), G)
        _write_alm(os.path.join(self.lib_dir, 'sim_x_p_%04d.fits'%idx if idx != -1 else 'dat_x_p.fits'), C)


    def _build_sim_MVgclm(self, idx):
        """ MV. lensing potentials estimators """
        G, C = self._get_sim_MVgclm(idx, 'p')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_MVgclm(idx, 'p', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        _write_alm(os.path.join(self.lib_dir, 'sim_p_%04d.fits'%idx if idx != -1 else 'dat_p.fits'), G)
        _write_alm(os.path.join(self.lib_dir, 'sim_x_%04d.fits'%idx if idx != -1 else 'dat_x.fits'), C)

    def _build_sim_f(self, idx):
        """ MV. modulation estimators. """
        G= self._get_sim_f_p(idx, joint=True)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            G = 0.5 * (G + self._get_sim_f_p(idx, joint=True, swapped=True))
        GT  = self._get_sim_ftt(idx, joint=True)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            GT = 0.5 * (GT + self._get_sim_ftt(idx, joint=True, swapped=True))
        _write_alm(os.path.join(self.lib_dir, 'sim_f_%04d.fits'%idx if idx != -1 else 'dat_f.fits'), G + GT)

    def _build_sim_xfiltMVgclm(self, idx, k):
        """
        Quick and dirty way to get the full set of estimators
        V X_1 W Y_2, or 1/2 (V X_1 W Y_2 + V Y_1 W X_2 ) if X_1 != Y_2; e.g. 1/2 (V E_1 W B_2 or V B_1 W E_2)
        """

        assert k in ['ptt', 'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                     'xtt', 'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']
        xfilt1 = {f: (k[-2] == f) * np.ones(10000) for f in ['t', 'e', 'b']}
        xfilt2 = {f: (k[-1] == f) * np.ones(10000) for f in ['t', 'e', 'b']}

        G, C = self._get_sim_Pgclm(idx, 'p', xfilt1=xfilt1, xfilt2=xfilt2)
        if not self.f2map1.ivfs == self.f2map2.ivfs or k[-1] != k[-2]:
            _G, _C = self._get_sim_Pgclm(idx, 'p', xfilt1=xfilt1, xfilt2=xfilt2, swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        GT, CT = self._get_sim_Tgclm(idx, 'p', xfilt1=xfilt1, xfilt2=xfilt2)
        if not self.f2map1.ivfs == self.f2map2.ivfs or k[-1] != k[-2]:
            _G, _C = self._get_sim_Tgclm(idx, 'p', xfilt1=xfilt1, xfilt2=xfilt2, swapped=True)
            GT = 0.5 * (GT + _G)
            del _G
            CT = 0.5 * (CT + _C)
            del _C
        fnameG = os.path.join(self.lib_dir, 'sim_p%s_%04d.fits' % (k[1:], idx) if idx != -1 else 'dat_p%s.fits'%k[1:])
        fnameC = os.path.join(self.lib_dir, 'sim_x%s_%04d.fits' % (k[1:], idx) if idx != -1 else 'dat_x%s.fits'%k[1:])
        _write_alm(fnameG, G + GT)
        _write_alm(fnameC, C + CT)

    def _build_sim_stt(self, idx):
        sLM = self._get_sim_stt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            pass  # No need to swap, this thing is symmetric anyways
        _write_alm(os.path.join(self.lib_dir, 'sim_stt_%04d.fits'%idx if idx != -1 else 'dat_stt.fits'), sLM)

    def _build_sim_ntt(self, idx):
        sLM = self._get_sim_ntt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            pass  # No need to swap, this thing is symmetric anyways
        _write_alm(os.path.join(self.lib_dir, 'sim_ntt_%04d.fits'%idx if idx != -1 else 'dat_ntt.fits'), sLM)

    def _build_sim_ftt(self, idx):
        fLM = self._get_sim_ftt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _fLM = self._get_sim_ftt(idx,swapped=True)
            fLM = 0.5 * (fLM + _fLM)
            del _fLM
        _write_alm(os.path.join(self.lib_dir, 'sim_ftt_%04d.fits'%idx if idx != -1 else 'dat_ftt.fits'), fLM)

    def _build_sim_f_p(self, idx):
        fLM = self._get_sim_f_p(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _fLM = self._get_sim_f_p(idx,swapped=True)
            fLM = 0.5 * (fLM + _fLM)
            del _fLM
        _write_alm(os.path.join(self.lib_dir, 'sim_f_p_%04d.fits'%idx if idx != -1 else 'dat_f_p.fits'), fLM)

    def _build_sim_a_p(self, idx):
        fLM = self._get_sim_a_p(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _fLM = self._get_sim_f_p(idx,swapped=True)
            fLM = 0.5 * (fLM + _fLM)
            del _fLM
        _write_alm(os.path.join(self.lib_dir, 'sim_a_p_%04d.fits'%idx if idx != -1 else 'dat_a_p.fits'), fLM)


class lib_filt2map(object):
    """
    Turns filtered maps into gradients and residual maps required for the qest.
    """

    def __init__(self, ivfs, nside):
        self.ivfs = ivfs
        self.nside = nside

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(), 'nside': self.nside}

    def get_gtmap(self, idx, k=None, xfilt=None):
        """
        \sum_{lm} MAP_talm sqrt(l (l + 1)) _1 Ylm(n).
        Spin 1 transform with zero curl comp.
        Recall healpy sign convention for which Glm = - Tlm.
        Output is list with real and imaginary part of the spin 1 transform.
        """
        assert xfilt is None, 'not implemented'
        mliktlm = self.ivfs.get_sim_tmliklm(idx)
        lmax = hp.Alm.getlmax(mliktlm.size)
        Glm = hp.almxfl(mliktlm, -np.sqrt(np.arange(lmax + 1, dtype=float) * (np.arange(1, lmax + 2))))
        return hp.alm2map_spin([Glm, np.zeros_like(Glm)], self.nside, 1, lmax)

    def get_tmap(self, idx):
        """Real-space Wiener filtered tmap.

        \sum_{lm} MAP_talm _0 Ylm(n).
        """
        return hp.alm2map(self.ivfs.get_sim_tmliklm(idx),self.nside)

    def get_pmap(self, idx):
        """Real-space Wiener filtered polarization.

        """
        Glm = self.ivfs.get_sim_emliklm(idx)
        Clm = self.ivfs.get_sim_bmliklm(idx)
        return hp.alm2map_spin([Glm, Clm], self.nside, 2, hp.Alm.getlmax(Glm.size))

    def get_gpmap(self, idx, spin, k=None, xfilt=None):
        """
        \sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n).
                                sqrt(l-2 (l+3)) _3 Ylm(n).
        Output is list with real and imaginary part of the spin 1 or 3 transforms.
        """
        assert spin in [1, 3]
        assert xfilt is None, 'not implemented'
        Glm = self.ivfs.get_sim_emliklm(idx)
        Clm = self.ivfs.get_sim_bmliklm(idx)

        assert Glm.size == Clm.size, (Clm.size, Clm.size)
        lmax = hp.Alm.getlmax(Glm.size)
        if spin == 1:
            fl = np.arange(2, lmax + 3, dtype=float) * (np.arange(-1, lmax))
        elif spin == 3:
            fl = np.arange(-2, lmax - 1, dtype=float) * (np.arange(3, lmax + 4))
        else:
            assert 0
        fl[:spin] *= 0.
        fl = np.sqrt(fl)
        hp.almxfl(Glm, fl, inplace=True)
        hp.almxfl(Clm, fl, inplace=True)
        return hp.alm2map_spin([Glm, Clm], self.nside, spin, lmax)

    def get_irestmap(self, idx, xfilt=None):
        if xfilt is not None:
            assert isinstance(xfilt, dict) and 't' in xfilt.keys()
            if not np.any(xfilt['t']):
                return np.zeros(hp.nside2npix(self.nside), dtype=float)
        reslm = self.ivfs.get_sim_tlm(idx)
        if xfilt is not None:
            reslm = hp.almxfl(reslm, xfilt['t'], inplace=True)
        return hp.alm2map(reslm, self.nside, lmax=hp.Alm.getlmax(reslm.size))

    def get_wirestmap(self, idx, wl):
        """ weighted res map w_l res_l"""
        reslm = self.ivfs.get_sim_tlm(idx)
        return hp.alm2map(hp.almxfl(reslm,wl), self.nside, lmax=hp.Alm.getlmax(reslm.size))

    def get_irespmap(self, idx, xfilt=None):
        reselm = self.ivfs.get_sim_elm(idx)
        resblm = self.ivfs.get_sim_blm(idx)
        assert hp.Alm.getlmax(reselm.size) == hp.Alm.getlmax(resblm.size)
        if xfilt is not None:
            assert isinstance(xfilt, dict) and 'e' in xfilt.keys() and 'b' in xfilt.keys()
            hp.almxfl(reselm, xfilt['e'], inplace=True)
            hp.almxfl(resblm, xfilt['b'], inplace=True)
        fac = 0.5
        return hp.alm2map_spin([reselm * fac, resblm * fac], self.nside, 2, hp.Alm.getlmax(reselm.size))


class lib_filt2map_sepTP(lib_filt2map):
    """
    Same as above but seprately filtered maps.
    """

    def __init__(self, ivfs, nside, clte):
        super(lib_filt2map_sepTP, self).__init__(ivfs, nside)
        self.clte = clte

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(), 'nside': self.nside,
                'clte': ut.clhash(self.clte)}

    def get_tmap(self, idx, joint=False):
        """Real-space Wiener filtered tmap.

        \sum_{lm} MAP_talm _0 Ylm(n).
        """
        tlm = self.ivfs.get_sim_tmliklm(idx)
        if joint:
            tlm += hp.almxfl(self.ivfs.get_sim_elm(idx), self.clte)
        return hp.alm2map(tlm, self.nside)

    def get_pmap(self, idx, joint=False):
        """Real-space Wiener filtered polarization.

        """
        Glm = self.ivfs.get_sim_emliklm(idx)
        Clm = self.ivfs.get_sim_bmliklm(idx)
        if joint:
            Glm += hp.almxfl(self.ivfs.get_sim_tlm(idx), self.clte)
        return hp.alm2map_spin([Glm, Clm], self.nside, 2, hp.Alm.getlmax(Glm.size))

    def get_gtmap(self, idx, k=None, xfilt=None):
        """
        \sum_{lm} MAP_talm sqrt(l (l + 1)) _1 Ylm(n).
        Spin 1 transform with zero curl comp.
        Recall healpy sign convention for which Glm = - Tlm.
        Output is list with real and imaginary part of the spin 1 transform.
        """
        assert k in ['ptt', 'p'], k
        if xfilt is not None:
            assert isinstance(xfilt, dict) and 't' in xfilt.keys()
            if k in ['p']:
                assert 'e' in xfilt.keys()
        need_t = (xfilt is None) or np.any(xfilt['t']) # want to avoid T-calc if unnecessary
        mliktlm = self.ivfs.get_sim_tmliklm(idx) if need_t else 0.
        if xfilt is not None and need_t:
            hp.almxfl(mliktlm, xfilt['t'], inplace=True)
        if k == 'p':
            need_e = (xfilt is None) or np.any(xfilt['e'])
            telm = hp.almxfl(self.ivfs.get_sim_elm(idx), self.clte) if need_e else 0.
            if xfilt is not None and need_e:
                assert 'e' in xfilt.keys()
                hp.almxfl(telm, xfilt['e'], inplace=True)
            mliktlm = mliktlm + telm
            del telm
        if np.any(mliktlm):
            lmax = hp.Alm.getlmax(mliktlm.size)
            Glm = hp.almxfl(mliktlm, -np.sqrt(np.arange(lmax + 1, dtype=float) * (np.arange(1, lmax + 2))))
            return hp.alm2map_spin([Glm, np.zeros_like(Glm)], self.nside, 1, lmax)
        else:
            return np.zeros(hp.nside2npix(self.nside), dtype=float), np.zeros(hp.nside2npix(self.nside), dtype=float)

    def get_gpmap(self, idx, spin, k=None, xfilt=None):
        """
        \sum_{lm} (Elm +- iBlm) sqrt(l+2 (l-1)) _1 Ylm(n).
                                sqrt(l-2 (l+3)) _3 Ylm(n).
        Output is list with real and imaginary part of the spin 1 or 3 transforms.
        """
        assert k in ['p_p', 'p'], k
        assert spin in [1, 3]
        if xfilt is not None:
            assert isinstance(xfilt, dict) and 'e' in xfilt.keys() and 'b' in xfilt.keys() and 't' in xfilt.keys()

        need_p = (xfilt is None) or (np.any(xfilt['e']) or np.any(xfilt['b']))
        Glm, Clm = (self.ivfs.get_sim_emliklm(idx), self.ivfs.get_sim_bmliklm(idx)) if need_p else (0., 0.)
        if xfilt is not None and need_p:
            hp.almxfl(Glm, xfilt['e'], inplace=True)
            hp.almxfl(Clm, xfilt['b'], inplace=True)
        if k == 'p':
            need_t = (xfilt is None) or np.any(xfilt['t'])
            G_tlm = hp.almxfl(self.ivfs.get_sim_tlm(idx), self.clte) if need_t else 0.
            if xfilt is not None and need_t:
                hp.almxfl(G_tlm, xfilt['t'], inplace=True)
            Glm = Glm + G_tlm
            del G_tlm
        if np.any(Glm) or np.any(Clm):
            lmax = hp.Alm.getlmax(Glm.size)
            if spin == 1:
                fl = np.arange(2, lmax + 3, dtype=float) * (np.arange(-1, lmax))
            elif spin == 3:
                fl = np.arange(-2, lmax - 1, dtype=float) * (np.arange(3, lmax + 4))
            else:
                assert 0
            fl[:spin] *= 0.
            fl = np.sqrt(fl)
            hp.almxfl(Glm, fl, inplace=True)
            if np.any(Clm):
                hp.almxfl(Clm, fl, inplace=True)
            if np.isscalar(Clm):
                return hp.alm2map_spin([Glm, Glm * 0.], self.nside, spin, lmax)
            else:
                return hp.alm2map_spin([Glm, Clm], self.nside, spin, lmax)
        else:
            return np.zeros(hp.nside2npix(self.nside), dtype=float), np.zeros(hp.nside2npix(self.nside), dtype=float)

