"""Calculation of semi-analytical noise biases module.

"""
from __future__ import print_function

import os
import pickle as pk
import numpy as np
import healpy as hp

from plancklens import qresp, utils, utils_spin as uspin
from plancklens.helpers import mpi, sql


def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivf1, lmax_ivf2,
            lmax_out=None, lmax_ivf12=None, lmax_ivf22=None, cls_weights2=None,
            cls_ivfs_bb=None, cls_ivfs_ab=None, cls_ivfs_ba=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

        Args:
            qe_key1: QE key 1
            qe_key2: QE key 2
            cls_weights: dictionary with the CMB spectra entering the QE weights.
                        (expected are 'tt', 'te', 'ee' when/if relevant)
            cls_ivfs: dictionary with the inverse-variance filtered CMB spectra.
                        (expected are 'tt', 'te', 'ee', 'bb', 'tb', 'eb' when/if relevant)
            lmax_ivf1: QE 1 uses CMB multipoles down to lmax_ivf1.
            lmax_ivf2: QE 2 uses CMB multipoles down to lmax_ivf2.
            lmax_out(optional): outputs are calculated down to lmax_out. Defaults to lmax_ivf1 + lmax_ivf2
            cls_weights2(optional): Second QE cls weights, if different from cls_weights

        Outputs:
            4-tuple of gradient (G) and curl (C) mode Gaussian noise co-variances GG, CC, GC, CG.

    """
    if lmax_ivf12 is None: lmax_ivf12 = lmax_ivf1
    if lmax_ivf22 is None: lmax_ivf22 = lmax_ivf2
    if cls_weights2 is None: cls_weights2 = cls_weights
    qes1 = qresp.get_qes(qe_key1, lmax_ivf1, cls_weights, lmax2=lmax_ivf12)
    qes2 = qresp.get_qes(qe_key2, lmax_ivf2, cls_weights2, lmax2=lmax_ivf22)
    if lmax_out is None:
        lmax_out = max(lmax_ivf1, lmax_ivf12) + max(lmax_ivf2, lmax_ivf22)
    return  _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=cls_ivfs_bb, cls_ivfs_ab=cls_ivfs_ab, cls_ivfs_ba=cls_ivfs_ba)

def _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=None, cls_ivfs_ab=None, cls_ivfs_ba=None, ret_terms=False):
    GG_N0 = np.zeros(lmax_out + 1, dtype=float)
    CC_N0 = np.zeros(lmax_out + 1, dtype=float)
    GC_N0 = np.zeros(lmax_out + 1, dtype=float)
    CG_N0 = np.zeros(lmax_out + 1, dtype=float)

    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs if cls_ivfs_ba is None else cls_ivfs_ba
    if ret_terms:
        terms = []
    for qe1 in qes1:
        cL1 = qe1.cL(np.arange(lmax_out + 1))
        for qe2 in qes2:
            cL2 = qe2.cL(np.arange(lmax_out + 1))
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)

            clsu = utils.joincls([qe1.leg_a.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([qe1.leg_b.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(ti, vi, cls_ivfs_bb)])
            R_sutv = utils.joincls([uspin.wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = utils.joincls([qe1.leg_a.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([qe1.leg_b.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(ti, ui, cls_ivfs_ba)])
            R_sutv = R_sutv + utils.joincls([uspin.wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_out), cL1, cL2])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = utils.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = utils.joincls([uspin.wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = utils.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-ti, ui, cls_ivfs_ba)])
            R_msmtuv = R_msmtuv + utils.joincls([uspin.wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_out), cL1, cL2])

            GG_N0 +=  0.5 * R_sutv.real
            GG_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv.real

            CC_N0 += 0.5 * R_sutv.real
            CC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.real

            GC_N0 -= 0.5 * R_sutv.imag
            GC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

            CG_N0 += 0.5 * R_sutv.imag
            CG_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag
            if ret_terms:
                terms += [0.5 * R_sutv, 0.5 * (-1) ** (to + so) * R_msmtuv]
    return (GG_N0, CC_N0, GC_N0, CG_N0) if not ret_terms else (GG_N0, CC_N0, GC_N0, CG_N0, terms)

class nhl_lib_simple:
    """Semi-analytical unnormalized N0 library.

        This version only for 4 identical legs, and with simple 1/fsky spectrum estimator.

        Args:
            lib_dir: outputs will be cached there
            ivfs: inverse-variance filtering library
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            lmax_qlm: noise (co-)variances are calculated up to multipole lmax_qlm
            resplib: only relevant for bias hardened estimators

    """
    def __init__(self, lib_dir, ivfs, cls_weight, lmax_qlm, resplib=None):
        self.lmax_qlm = lmax_qlm
        self.cls_weight = cls_weight
        self.ivfs = ivfs
        fn_hash = os.path.join(lib_dir, 'nhl_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict(), fn=fn_hash)

        self.lib_dir = lib_dir
        self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))
        self.fsky = np.mean(self.ivfs.get_fmask())
        self.resplib = resplib

    def hashdict(self):
        ret = {k: utils.clhash(self.cls_weight[k]) for k in self.cls_weight.keys()}
        ret['ivfs']  = self.ivfs.hashdict()
        ret['lmax_qlm'] = self.lmax_qlm
        return ret

    def _get_qe_derived(self, k):
        if '_bh_' in k:
            kQE, ksource = k.split('_bh_')
            assert len(ksource) == 1.
            wL = self.resplib.get_response(kQE, ksource) * utils.cli(self.resplib.get_response(ksource + kQE[1:], ksource))
            return [(kQE, 1.), (ksource + kQE[1:], -wL)]
        else:
            return [(k, 1.)]

    def get_sim_nhl(self, idx, k1, k2, recache=False):
        """
            Args:
                idx: simulation index
                k1: QE key 1
                k2: QE key 2
        """
        assert idx == -1 or idx >= 0, idx
        k1sw = self._get_qe_derived(k1)
        k2sw = self._get_qe_derived(k2)
        ret = np.zeros(self.lmax_qlm + 1)
        for k1, w1 in k1sw:
            for k2, w2 in k2sw:
                s1, GC1, s1ins, ksp1 = qresp.qe_spin_data(k1)
                s2, GC2, s2ins, ksp2 = qresp.qe_spin_data(k2)
                fn = 'anhl_qe_' + ksp1 + k1[1:] + '_qe_' + ksp2 +  k2[1:] + GC1 + GC2
                suf =  ('sim%04d'%idx) * (int(idx) >= 0) +  'dat' * (idx == -1)
                if self.npdb.get(fn + suf) is None or recache:
                    assert s1 >= 0 and s2 >= 0, (s1, s2)
                    cls_ivfs, lmax_ivf = self._get_cls(idx, np.unique(np.concatenate([s1ins, s2ins])))
                    GG, CC, GC, CG = get_nhl(k1, k2, self.cls_weight, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=self.lmax_qlm)
                    fns = [('G', 'G', GG) ] + [('C', 'G', CG)] * (s1 > 0) + [('G', 'C', GC)] * (s2 > 0) + [('C', 'C', CC)] * (s1 > 0) * (s2 > 0)
                    if recache and self.npdb.get(fn + suf) is not None:
                        for GC1, GC2, N0 in fns:
                            self.npdb.remove('anhl_qe_' + ksp1 +  k1[1:] + '_qe_'+ ksp2 + k2[1:] + GC1 + GC2 + suf)
                    for GC1, GC2, N0 in fns:
                        self.npdb.add('anhl_qe_' + ksp1 + k1[1:] + '_qe_' + ksp2 + k2[1:] + GC1 + GC2 + suf, N0)
                ret += w1 * w2 * self.npdb.get(fn + suf)
        return ret

    def _get_cls(self, idx, spins):
        assert np.all(spins >= 0), spins
        ret = {}
        if 0 in spins:
            ret['tt'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx)) / self.fsky
        if 2 in spins:
            ret['ee'] = hp.alm2cl(self.ivfs.get_sim_elm(idx)) / self.fsky
            ret['bb'] = hp.alm2cl(self.ivfs.get_sim_blm(idx)) / self.fsky
            ret['eb'] = hp.alm2cl(self.ivfs.get_sim_elm(idx), alms2=self.ivfs.get_sim_blm(idx)) / self.fsky
        if 0 in spins and 2 in spins:
            ret['te'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx), alms2=self.ivfs.get_sim_elm(idx)) / self.fsky
            ret['tb'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx), alms2=self.ivfs.get_sim_blm(idx)) / self.fsky
        lmaxs = [len(cl) for cl in ret.values()]
        assert len(np.unique(lmaxs)) == 1, lmaxs
        return ret, lmaxs[0]

def cls2dls(cls):
    """Turns cls dict. into camb cl array format"""
    keys = ['tt', 'ee', 'bb', 'te']
    lmax = np.max([len(cl) for cl in cls.values()]) - 1
    dls = np.zeros((lmax + 1, 4), dtype=float)
    refac = np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float) / (2. * np.pi)
    for i, k in enumerate(keys):
        cl = cls.get(k, np.zeros(lmax + 1, dtype=float))
        sli = slice(0, min(len(cl), lmax + 1))
        dls[sli, i] = cl[sli] * refac[sli]
    cldd = np.copy(cls.get('pp', None))
    if cldd is not None:
        cldd *= np.arange(len(cldd)) ** 2 * np.arange(1, len(cldd) + 1, dtype=float) ** 2 /  (2. * np.pi)
    return dls, cldd

def dls2cls(dls):
    """Inverse operation to cls2dls"""
    assert dls.shape[1] == 4
    lmax = dls.shape[0] - 1
    cls = {}
    refac = 2. * np.pi * utils.cli( np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k] = dls[:, i] * refac
    return cls


def get_N0_iter(qe_key:str, nlev_t:float, nlev_p:float, beam_fwhm:float, cls_unl_fid:dict, lmin_ivf, lmax_ivf, itermax, cls_unl_dat=None,
                lmax_qlm=None, ret_delcls=False, ret_resp=False, datnoise_cls:dict or None=None, unlQE=False, version='1'):
    """Iterative lensing-N0 estimate

        Calculates iteratively partially lensed spectra and lensing noise levels.
        This uses the python camb package to get the partially lensed spectra.

        This makes no assumption on response =  1 / noise hence is about twice as slow as it could be in standard cases.

        Args:
            qe_key: QE estimator key
            nlev_t: temperature noise level (in :math:`\mu `K-arcmin)
            nlev_p: polarisation noise level (in :math:`\mu `K-arcmin)
            beam_fwhm: Gaussian beam full width half maximum in arcmin
            cls_unl_fid(dict): unlensed CMB power spectra
            lmin_ivf: minimal CMB multipole used in the QE
            lmax_ivf: maximal CMB multipole used in the QE
            itermax: number of iterations to perform
            lmax_qlm(optional): maximum lensing multipole to consider. Defaults to :math:`2 lmax_ivf`
            ret_delcls(optional): returns the partially delensed CMB cls as well if set
            ret_resp(optional): returns the iterative response
            datnoise_cls(optional): feeds in custom noise spectra to the data. The nlevs and beam only apply to the filtering in this case

        Returns
            Array of shape (itermax + 1, lmax_qlm + 1) with all iterated N0s. First entry is standard N0.


        Note: This assumes the unlensed spectra are known

    #FIXME: this is requiring the full camb python package for the lensed spectra calc.

     """
    assert qe_key in ['p_p', 'p', 'ptt'], qe_key
    try:
        from camb.correlations import lensed_cls
    except ImportError:
        assert 0, "could not import camb.correlations.lensed_cls"


    if lmax_qlm is None:
        lmax_qlm = 2 * lmax_ivf
    lmax_qlm = min(lmax_qlm, 2 * lmax_ivf)
    lmin_ivf = max(lmin_ivf, 1)
    transfi2 = utils.cli(hp.gauss_beam(beam_fwhm / 180. / 60. * np.pi, lmax=lmax_ivf)) ** 2
    llp2 = np.arange(lmax_qlm + 1, dtype=float) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2 / (2. * np.pi)
    if datnoise_cls is None:
        datnoise_cls = dict()
        if qe_key in ['ptt', 'p']:
            datnoise_cls['tt'] = (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
        if qe_key in ['p_p', 'p']:
            datnoise_cls['ee'] = (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            datnoise_cls['bb'] = (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
    N0s_biased = []
    N0s_unbiased = []
    N1s_biased = []
    N1s_unbiased = []
    delcls_fid = []
    delcls_true = []

    N0_unbiased = np.inf
    N1_unbiased = np.inf
    dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
    cls_len_fid= dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
    if cls_unl_dat is None:
        cls_unl_dat = cls_unl_fid
        cls_len_true= cls_len_fid
    else:
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        cls_len_true= dls2cls(lensed_cls(dls_unl_true, cldd_true))
    cls_plen_true = cls_len_true
    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = 0.
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            rho_sqd_phi[:lmax_qlm +1] =   cldd_true[:lmax_qlm + 1] * utils.cli(cldd_true[:lmax_qlm + 1] + llp2 * (N0_unbiased[:lmax_qlm+1] + N1_unbiased[:lmax_qlm + 1]))

        if 'wE' in version:
            assert qe_key in ['p_p']
            if it == 0:
                print('including imperfect knowledge of E in iterations')
            slic = slice(lmin_ivf, lmax_ivf + 1)
            rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
            # rho_sqd_E[slic] = cls_unl_dat['ee'][slic] * utils.cli(cls_plen_true['ee'][slic] + datnoise_cls['ee'][slic])
            rho_sqd_E[slic] = cls_len_fid['ee'][slic] * utils.cli(cls_len_fid['ee'][slic] + datnoise_cls['ee'][slic]) # Assuming that the difference between lensed and unlensed EE can be neglected
            dls_unl_fid[:, 1] *= rho_sqd_E
            dls_unl_true[:, 1] *= rho_sqd_E
            cldd_fid *= rho_sqd_phi
            cldd_true *= rho_sqd_phi

            cls_plen_fid_resolved = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true_resolved = dls2cls(lensed_cls(dls_unl_true, cldd_true))
            cls_plen_fid =  {ck: cls_len_fid[ck] - (cls_plen_fid_resolved[ck] - cls_unl_fid[ck][:len(cls_len_fid[ck])]) for ck in cls_len_fid.keys()}
            cls_plen_true = {ck: cls_len_true[ck] -(cls_plen_true_resolved[ck] - cls_unl_dat[ck][:len(cls_len_true[ck])]) for ck in cls_len_true.keys()}

        else:
            cldd_true *= (1. - rho_sqd_phi)  # The true residual lensing spec.
            cldd_fid *= (1. - rho_sqd_phi)  # What I think the residual lensing spec is
            cls_plen_fid  = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))

        cls_filt = cls_plen_fid if not unlQE else cls_unl_fid
        cls_w = cls_plen_fid if not unlQE else cls_unl_fid
        cls_f = cls_plen_true
        fal = {}
        dat_delcls = {}
        if qe_key in ['ptt', 'p']:
            fal['tt'] = cls_filt['tt'][:lmax_ivf + 1] + (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['tt'] = cls_plen_true['tt'][:lmax_ivf + 1] + datnoise_cls['tt']
        if qe_key in ['p_p', 'p']:
            fal['ee'] = cls_filt['ee'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            fal['bb'] = cls_filt['bb'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['ee'] = cls_plen_true['ee'][:lmax_ivf + 1] + datnoise_cls['ee']
            dat_delcls['bb'] = cls_plen_true['bb'][:lmax_ivf + 1] + datnoise_cls['bb']
        if qe_key in ['p']:
            fal['te'] = np.copy(cls_filt['te'][:lmax_ivf + 1])
            dat_delcls['te'] = np.copy(cls_plen_true['te'][:lmax_ivf + 1])
        fal = utils.cl_inverse(fal)
        # TODO: Should update if we use different lmin_ivf for T, E and B ?
        for cl in fal.values():
            cl[:lmin_ivf] *= 0.
        for cl in dat_delcls.values():
            cl[:lmin_ivf] *= 0.
        cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
        cls_ivfs = dict()
        for i, a in enumerate(['t', 'e', 'b']):
            for j, b in enumerate(['t', 'e', 'b'][i:]):
                if np.any(cls_ivfs_arr[i, j + i]):
                    cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]

        n_gg = get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
        r_gg_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0]
        r_gg_fid = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_w, fal, lmax_qlm=lmax_qlm)[0] if cls_f is not cls_w else r_gg_true
        N0_biased = n_gg * utils.cli(r_gg_fid ** 2) # N0 of possibly biased (by Rtrue / Rfid) QE estimator
        N0_unbiased = n_gg * utils.cli(r_gg_true ** 2) # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased
        N0s_biased.append(N0_biased)
        N0s_unbiased.append(N0_unbiased)

        cls_plen_true['pp'] =  cldd_true *utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 /  (2. * np.pi))
        cls_plen_fid['pp'] =  cldd_fid *utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 /  (2. * np.pi))
        
        if 'wE' and it>0:
            # In case we have E noise in the iterations, then the rho is defined differently. 
            cls_plen_true['pp'] =  cls_plen_true['pp'] *utils.cli( rho_sqd_phi) * (1. - rho_sqd_phi) 
            cls_plen_fid['pp'] =  cls_plen_fid['pp'] *utils.cli( rho_sqd_phi) * (1. - rho_sqd_phi) 
        elif 'wE' and it ==0:
            cls_plen_true['pp'] =  cls2dls(cls_unl_dat)[1] * utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 /  (2. * np.pi))
            cls_plen_fid['pp'] =  cls2dls(cls_unl_fid)[1] * utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 /  (2. * np.pi))
        
        if 'wN1' in version:
            if it == 0: print('Adding n1 in iterations')
            from lensitbiases import n1_fft
            from scipy.interpolate import UnivariateSpline as spl
            lib = n1_fft.n1_fft(fal, cls_w, cls_f, np.copy(cls_plen_true['pp']), lminbox=50, lmaxbox=5000, k2l=None)
            n1_Ls = np.arange(50, lmax_qlm+1, 50)
            if lmax_qlm not in n1_Ls:  n1_Ls = np.append(n1_Ls, lmax_qlm)
            n1 = np.array([lib.get_n1(qe_key, L, do_n1mat=False) for L in n1_Ls])
            N1_biased  = spl(n1_Ls, n1_Ls ** 2 * (n1_Ls * 1. + 1) ** 2 * n1 / r_gg_fid[n1_Ls] ** 2, k=2, s=0, ext='zeros')(np.arange(len(N0_unbiased)))
            N1_biased *= utils.cli(np.arange(lmax_qlm + 1) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2)
            N1_unbiased = N1_biased * (r_gg_fid * utils.cli(r_gg_true)) ** 2
        else:
            N1_biased = np.zeros(lmax_qlm + 1, dtype=float)
            N1_unbiased = np.zeros(lmax_qlm + 1, dtype=float)


        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)

        N1s_biased.append(N1_biased)
        N1s_unbiased.append(N1_unbiased)
    ret = (np.array(N0s_biased), np.array(N0s_unbiased))
    if ret_delcls:
        ret += (delcls_fid, delcls_true)
    if ret_resp:
        ret += (r_gg_fid, r_gg_true)
    if 'wN1' in version:
        ret+= (N1s_biased, N1s_unbiased)
    return ret
