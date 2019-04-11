from __future__ import print_function
import healpy as hp
import numpy as np
import os
import pickle as pk
import collections

from . import utils
from . import mpi

#FIXME lmax_qlm's

def library_jtTP(lib_dir, ivfs1, ivfs2, nside, lmax_qlm=None, resplib=None):
    if lmax_qlm is None: lmax_qlm={'T': 4096, 'P': 4096, 'PS': 4096}
    return library(lib_dir, ivfs1, ivfs2, nside, lmax_qlm=lmax_qlm, resplib=resplib)


def library_sepTP(lib_dir, ivfs1, ivfs2, clte, nside, lmax_qlm=None, resplib=None):
    if lmax_qlm is None: lmax_qlm={'T': 4096, 'P': 4096, 'PS': 4096}
    return library(lib_dir, ivfs1, ivfs2, nside, clte=clte, lmax_qlm=lmax_qlm, resplib=resplib)


class library:
    """
    If clte is set, this assume separately filtered T and P maps.
    Else, maps are jointly filtered.
    For jointly filtered T/P library 'p_tp' and 'x_tp' becomes the true MV estimator.
    For separatly filtered T/P library, the gmaps need to be modified according to clte.
    """

    # FIXME: could differentiate the two libraries. For jtl filt there is no need to make 'p','x'
    # FIXME a fundamental key

    def __init__(self, lib_dir, ivfs1, ivfs2, nside, clte=None, lmax_qlm=None, resplib=None):
        if lmax_qlm is None : lmax_qlm = {'T': 4096, 'P': 4096, 'PS': 4096}
        self.lib_dir = lib_dir
        self.prefix = lib_dir
        self.lmax_qlm = lmax_qlm
        if clte is None:
            self.f2map1 = lib_filt2map(ivfs1, nside)
            self.f2map2 = lib_filt2map(ivfs2, nside)
        else:
            self.f2map1 = lib_filt2map_sepTP(ivfs1, nside, clte)
            self.f2map2 = lib_filt2map_sepTP(ivfs2, nside, clte)
        assert lmax_qlm['T'] == lmax_qlm['P'], 'implement this'
        if (mpi.rank == 0) and (not os.path.exists(self.lib_dir + "/qe_sim_hash.pk")):
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
            pk.dump(self.hashdict(), open(self.lib_dir + "/qe_sim_hash.pk", 'wb'))
        mpi.barrier()

        utils.hash_check(pk.load(open(self.lib_dir + "/qe_sim_hash.pk", 'rb')), self.hashdict())
        if mpi.rank == 0:
            if not os.path.exists(lib_dir + '/fskies.dat'):
                print("Caching sky fractions...")
                ms = {1: self.get_mask(1), 2: self.get_mask(2)}
                fskies = {}
                for i in [1, 2]:
                    for j in [1, 2][i - 1:]:
                        fskies[10 * i + j] = np.mean(ms[i] * ms[j])
                with open(lib_dir + '/fskies.dat', 'wb') as f:
                    for lab, _f in zip(np.sort(list(fskies.keys())), np.array(list(fskies.values()))[np.argsort(list(fskies.keys()))]):
                        f.write('%4s %.5f \n' % (lab, _f))
        mpi.barrier()
        fskies = {}
        with open(lib_dir + '/fskies.dat') as f:
            for line in f:
                (key, val) = line.split()
                fskies[int(key)] = float(val)
        self.fskies = fskies
        self.fsky11 = fskies[11]
        self.fsky12 = fskies[12]
        self.fsky22 = fskies[22]

        self.resplib = resplib
        self.keys_fund = ['ptt', 'xtt', 'p_p', 'x_p', 'p', 'x', 'stt', 'ftt', 'dtt', 'ntt',
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
        assert k in self.keys, (k, self.keys)
        if lmax is None :
            lmax = self.get_lmax_qlm(k)
        assert lmax <= self.get_lmax_qlm(k)
        if k in ['p_tp', 'x_tp', 'f_tp', 's_tp']:
            return self.get_sim_qlm('%stt' % k[0], idx, lmax=lmax) + self.get_sim_qlm('%s_p' % k[0], idx, lmax=lmax)
        if k in ['p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb']:
            return 0.5 * (self.get_sim_qlm(k[0]+k[2]+k[3], idx, lmax=lmax) + self.get_sim_qlm(k[0]+k[3]+k[2], idx, lmax=lmax))
        if 'tt_bh_'in k:
            _k,f = k.split('_bh_')
            assert self.get_lmax_qlm(_k) == self.get_lmax_qlm(f + 'tt'),'fix this (easy)'
            lmax = self.get_lmax_qlm(_k)
            wL = self.resplib.get_response(_k,f + 'tt') * self.resplib.get_response(f + 'tt',f + 'tt').inverse()
            rethp = self.get_sim_qlm(_k, idx, lmax=lmax)
            rethp -= hp.almxfl(self.get_sim_qlm(f + 'tt', idx, lmax=lmax),wL[:])
            return rethp

        assert k in self.keys_fund, (k, self.keys_fund)
        fname = self.lib_dir + '/sim_%s_%04d.fits' % (k, idx) if idx != -1 else self.lib_dir + '/dat_%s.fits' % k
        if not os.path.exists(fname):
            if k in ['ptt', 'xtt']: self._build_sim_Tgclm(idx)
            elif k in ['p_p', 'x_p']: self._build_sim_Pgclm(idx)
            elif k in ['p', 'x']: self._build_sim_MVgclm(idx)
            elif k in ['stt']: self._build_sim_stt(idx)
            elif k in ['ftt']: self._build_sim_ftt(idx)
            elif k in ['ntt']: self._build_sim_ntt(idx)
            elif k in ['ptt', 'pte', 'pet', 'ptb', 'pbt', 'pee', 'peb', 'pbe', 'pbb',
                       'xtt', 'xte', 'xet', 'xtb', 'xbt', 'xee', 'xeb', 'xbe', 'xbb']:
                self._build_sim_xfiltMVgclm(idx, k)
            else:
                assert 0, k

        return  utils.alm_copy(hp.read_alm(fname), lmax=lmax)

    def get_dat_qlm(self, k, **kwargs):
        return self.get_sim_qlm(k, -1, **kwargs)

    def get_sim_qlm_mf(self, k, mc_sims, lmax=None):
        if lmax is None:
            lmax = self.get_lmax_qlm(k)
        assert lmax <= self.get_lmax_qlm(k)
        if k in ['p_tp', 'x_tp']:
            return (self.get_sim_qlm_mf('%stt' % k[0], mc_sims, lmax=lmax)
                    + self.get_sim_qlm_mf('%s_p' % k[0], mc_sims, lmax=lmax))
        if k in ['p_te', 'p_tb', 'p_eb', 'x_te', 'x_tb', 'x_eb']:
            return 0.5 * (self.get_sim_qlm_mf(k[0] + k[2] + k[3], mc_sims, lmax=lmax) \
                        + self.get_sim_qlm_mf(k[0] + k[3] + k[2], mc_sims, lmax=lmax))
        elif 'tt_bh_' in k:
            _k,f = k.split('_bh_')
            assert self.get_lmax_qlm(_k) == self.get_lmax_qlm(f + 'tt'),'fix this (easy)'
            lmax = self.get_lmax_qlm(_k)
            wL = self.resplib.get_response(_k,f + 'tt') * self.resplib.get_response(f + 'tt',f + 'tt').inverse()
            rethp = self.get_sim_qlm_mf(_k, mc_sims, lmax=lmax)
            rethp -= hp.almxfl(self.get_sim_qlm_mf(f + 'tt',mc_sims, lmax=lmax),wL[:])
            return rethp
        assert k in self.keys_fund, (k, self.keys_fund)
        fname = self.lib_dir + '/simMF_k1%s_%s.fits' % (k, utils.mchash(mc_sims))
        if not os.path.exists(fname):
            MF = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
            if len(mc_sims) == 0: return MF
            for i, idx in utils.enumerate_progress(mc_sims, label='calculating %s MF' % k):
                MF += self.get_sim_qlm(k, idx, lmax=lmax)
            MF /= len(mc_sims)
            hp.write_alm(fname, MF)
            print("Cached ", fname)
        return utils.alm_copy(hp.read_alm(fname), lmax=lmax)

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
        """ Point source estimator """
        #FIXME: sign consistent with Planck Lensing 2015, but couter-intuitive and should be fixed.
        tmap1 = self.f2map1.get_irestmap(idx) if not swapped else self.f2map2.get_irestmap(idx)  # healpy map
        tmap1 *= (self.f2map2.get_irestmap(idx) if not swapped else self.f2map1.get_irestmap(idx))  # healpy map
        return -0.5 * hp.map2alm(tmap1, lmax=self.get_lmax_qlm('PS'),iter=0)

    def _get_sim_ntt(self, idx, swapped=False):
        """ Noise inhomogeneity estimator (same as point-source estimator but acting on beam-deconvolved maps) """
        #FIXME: sign consistent with Planck Lensing 2015, but couter-intuitive and should be fixed.
        f1 = self.f2map1 if not swapped else self.f2map2
        f2 = self.f2map2 if not swapped else self.f2map1
        tmap1 = f1.get_wirestmap(idx, f1.ivfs.get_tal('t')[:]) * f2.get_wirestmap(idx, f2.ivfs.get_tal('t')[:])
        return -0.5 * hp.map2alm(tmap1, lmax=self.get_lmax_qlm('T'),iter=0)

    def _get_sim_ftt(self, idx, swapped=False):
        """ Modulation estimator """
        tmap1 = self.f2map1.get_irestmap(idx) if not swapped else self.f2map2.get_irestmap(idx)  # healpy map
        tmap1 *= (self.f2map2.get_tmap(idx) if not swapped else self.f2map1.get_tmap(idx))  # healpy map
        return hp.map2alm(tmap1, lmax=self.get_lmax_qlm('T'), iter = 0)

    def _build_sim_Tgclm(self, idx):
        """ T only lensing potentials estimators """
        G, C = self._get_sim_Tgclm(idx, 'ptt')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Tgclm(idx, 'ptt', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        fnameG = self.lib_dir + '/sim_ptt_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_ptt.fits'
        fnameC = self.lib_dir + '/sim_xtt_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_xtt.fits'
        hp.write_alm(fnameG, G)
        hp.write_alm(fnameC, C)

    def _build_sim_Pgclm(self, idx):
        """ Pol. only lensing potentials estimators """
        G, C = self._get_sim_Pgclm(idx, 'p_p')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Pgclm(idx, 'p_p', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        fnameG = self.lib_dir + '/sim_p_p_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_p_p.fits'
        fnameC = self.lib_dir + '/sim_x_p_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_x_p.fits'
        hp.write_alm(fnameG, G)
        hp.write_alm(fnameC, C)

    def _build_sim_MVgclm(self, idx):
        """ MV. lensing potentials estimators """
        G, C = self._get_sim_Pgclm(idx, 'p')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Pgclm(idx, 'p', swapped=True)
            G = 0.5 * (G + _G)
            del _G
            C = 0.5 * (C + _C)
            del _C
        GT, CT = self._get_sim_Tgclm(idx, 'p')
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _G, _C = self._get_sim_Tgclm(idx, 'p', swapped=True)
            GT = 0.5 * (GT + _G)
            del _G
            CT = 0.5 * (CT + _C)
            del _C
        fnameG = self.lib_dir + '/sim_p_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_p.fits'
        fnameC = self.lib_dir + '/sim_x_%04d.fits'%idx if idx != -1 else self.lib_dir + '/dat_x.fits'
        hp.write_alm(fnameG, G + GT)
        hp.write_alm(fnameC, C + CT)

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
        fnameG = self.lib_dir + '/sim_p%s_%04d.fits' % (k[1:], idx) if idx != -1 else self.lib_dir + '/dat_p%s.fits'%k[1:]
        fnameC = self.lib_dir + '/sim_x%s_%04d.fits' % (k[1:], idx) if idx != -1 else self.lib_dir + '/dat_x%s.fits'%k[1:]
        hp.write_alm(fnameG, G + GT)
        hp.write_alm(fnameC, C + CT)

    def _build_sim_stt(self, idx):
        sLM = self._get_sim_stt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            pass  # No need to swap, this thing is symmetric anyways
            # sLM = 0.5 * (sLM + self._get_sim_stt(idx, swapped=True))
        fname = self.lib_dir + '/sim_stt_%04d.fits' % (idx) if idx != -1 else self.lib_dir + '/dat_stt.fits'
        hp.write_alm(fname, sLM)

    def _build_sim_ntt(self, idx):
        sLM = self._get_sim_ntt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            pass  # No need to swap, this thing is symmetric anyways
            # nLM = 0.5 * (nLM + self._get_sim_ntt(idx, swapped=True))
        fname = self.lib_dir + '/sim_ntt_%04d.fits' % (idx) if idx != -1 else self.lib_dir + '/dat_ntt.fits'
        hp.write_alm(fname, sLM)

    def _build_sim_ftt(self, idx):
        fLM = self._get_sim_ftt(idx)
        if not self.f2map1.ivfs == self.f2map2.ivfs:
            _fLM = self._get_sim_ftt(idx,swapped=True)
            fLM = 0.5 * (fLM + _fLM)
            del _fLM
        fname = self.lib_dir + '/sim_ftt_%04d.fits' % (idx) if idx != -1 else self.lib_dir + '/dat_ftt.fits'
        hp.write_alm(fname, fLM)

    def get_response(self, k1, k2, recache=False):
        return self.resplib.get_response(k1, k2, recache=recache)


class lib_filt2map(object):
    """
    Turns filtered maps into gradients and residual maps required for the qest.
    """

    def __init__(self, ivfs, nside):
        self.ivfs = ivfs
        self.nside = nside

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(), 'nside': self.nside}

    def get_gtmap(self, idx, k=None):
        """
        \sum_{lm} MAP_talm sqrt(l (l + 1)) _1 Ylm(n).
        Spin 1 transform with zero curl comp.
        Recall healpy sign convention for which Glm = - Tlm.
        Output is list with real and imaginary part of the spin 1 transform.
        """
        mliktlm = self.ivfs.get_sim_tmliklm(idx)
        lmax = hp.Alm.getlmax(mliktlm.size)
        Glm = hp.almxfl(mliktlm, -np.sqrt(np.arange(lmax + 1, dtype=float) * (np.arange(1, lmax + 2))))
        return hp.alm2map_spin([Glm, np.zeros_like(Glm)], self.nside, 1, lmax)

    def get_tmap(self, idx, k=None):
        """Real-space Wiener filtered tmap.

        \sum_{lm} MAP_talm _0 Ylm(n).
        """
        return hp.alm2map(self.ivfs.get_sim_tmliklm(idx),self.nside)

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
        reslm =self.ivfs.get_sim_tlm(idx)
        if xfilt is not None:
            assert isinstance(xfilt, dict) and 't' in xfilt.keys()
            hp.almxfl(reslm, xfilt['t'], inplace=True)
        return hp.alm2map(reslm, self.nside, lmax=hp.Alm.getlmax(reslm.size), verbose=False)

    def get_wirestmap(self, idx, wl):
        """ weighted res map w_l res_l"""
        reslm = self.ivfs.get_sim_tlm(idx)
        return hp.alm2map(hp.almxfl(reslm,wl), self.nside, lmax=hp.Alm.getlmax(reslm.size), verbose=False)

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
                'clte': utils.clhash(self.clte)}

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
        mliktlm = self.ivfs.get_sim_tmliklm(idx)
        if xfilt is not None:
            hp.almxfl(mliktlm, xfilt['t'], inplace=True)
        if k == 'p':
            telm = hp.almxfl(self.ivfs.get_sim_elm(idx), self.clte)
            if xfilt is not None:
                assert 'e' in xfilt.keys()
                hp.almxfl(telm, xfilt['e'], inplace=True)
            mliktlm += telm
            del telm
        lmax = hp.Alm.getlmax(mliktlm.size)
        Glm = hp.almxfl(mliktlm, -np.sqrt(np.arange(lmax + 1, dtype=float) * (np.arange(1, lmax + 2))))
        return hp.alm2map_spin([Glm, np.zeros_like(Glm)], self.nside, 1, lmax)

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
        Glm = self.ivfs.get_sim_emliklm(idx)
        Clm = self.ivfs.get_sim_bmliklm(idx)
        if xfilt is not None:
            hp.almxfl(Glm, xfilt['e'], inplace=True)
            hp.almxfl(Clm, xfilt['b'], inplace=True)
        if k == 'p':
            G_tlm = hp.almxfl(self.ivfs.get_sim_tlm(idx), self.clte)
            if xfilt is not None:
                hp.almxfl(G_tlm, xfilt['t'], inplace=True)
            Glm += G_tlm
            del G_tlm
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
