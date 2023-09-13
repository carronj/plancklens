"""This module provides convenience functions to calculate curved-sky responses and reconstruction noise curve for lensing or other estimators
    In plancklens a QE is often described by a short string.
    For example 'ptt' stands for lensing (or lensing gradient mode) from temperature x temperature.
    Anisotropy source keys are a one-letter string including
        'p' (lensing gradient)
        'x' (lensing curl)
        's' (point sources)
        'f' (modulation field)
        'a' (polarization rotation)
    Typical keys include then:
        'ptt', 'xtt', 'stt', 'ftt' for the corresponding QEs from temperature only
        'p_p', 'x_p', 'f_p', 'a_p' for the corresponding QEs from polarization only (combining EE EB and BB if relevant)
        'p', 'x', 'f', 'a', 'f' ... for the MV (or GMV) combination
        'p_eb', ... for the EB estimator (this is the symmetrized version  ('peb' + 'pbe') / 2  so that E and B appear each once on the gradient and inverse-variance filtered leg)
    Bias-hardening can be included by inserting '_bh_'.
        E.g. 'ptt_bh_s' is the lensing TT QE bias-hardened against point source contamination using the 'stt' estimator
    Responses method takes as input the QE weights (typically the lensed CMB spectra) and the filtering cls ('fals')
    which describes the filtering applied to the maps (the :math:`(C + N)^{-1}` operation)
    *get_N0_iter* calculates an estimate of the N0s for iterative lensing estimator beyond the QE
"""

import os
import healpy as hp
import numpy as np
import plancklens
from plancklens import utils, qresp, nhl
from copy import deepcopy


def get_N0(beam_fwhm=1.4, nlev_t: float or np.ndarray = 5., nlev_p: np.array = None, lmax_CMB: dict or int = 3000,
           lmin_CMB=100, lmax_out=None,
           cls_filt: dict or None =None,
           cls_len: dict or None = None,
           cls_weight: dict or None = None,
           cls_sky: dict or None = None,
           joint_TP=True, ksource='p',wfleg_Tcut=None):
    r"""Example function to calculates reconstruction noise levels for a bunch of quadratic estimators


        Args:
            beam_fwhm: beam fwhm in arcmin, assuming gaussian shape.
            nlev_t: T white noise level in uK-arcmin (an array of size lmax_CMB can be passed for scale-dependent noise level)
            nlev_p: P white noise level in uK-arcmin (defaults to root(2) nlevt, if None) (can also be a (non-white) array of shape (lmax_CMB), or with either polarization or E and B noise separately, in which case the array is of shape (2,lmax_CMB)), and E noise expected at first
            lmax_CMB: max. CMB multipole used in the QE (use a dict with 't' 'e' 'b' keys instead of int to set different CMB lmaxes)
            lmin_CMB: min. CMB multipole used in the QE
            lmax_out: max lensing 'L' multipole calculated
            cls_filt: set of spectra used for the filtering, defaults to FFP10 lensed CMB spectra
            cls_len: CMB spectra entering the sky response to the anisotropy
                     (defaults to FFP10 lensed CMB spectra, in general should be the lensed gradient spectra)
            cls_sky: actual spectra of the measured CMB (without noise); defaults to FFP10 lensed CMB spectra
            cls_weight: CMB spectra entering the QE weights (defaults to FFP10 lensed CMB spectra; optimally, gradient spectra)
            joint_TP: if True include calculation of the N0s for the GMV estimator (incl. joint T and P filtering)
            ksource: anisotropy source to consider (defaults to 'p', lensing)
            wfleg_Tcut: high-l cut on the T gradient leg if set


        Returns:
            N0s array for the lensing gradient and curl modes for  the T-only, P-onl and (G)MV estimators


        Prompted by AL
    """
    if nlev_p is None:
        nlev_p = nlev_t * np.sqrt(2)
    if not isinstance(lmax_CMB, dict):
        lmaxs_CMB = {s: lmax_CMB for s in ['t', 'e', 'b']}
    else:
        lmaxs_CMB = lmax_CMB
        print("Seeing CMB lmax's:")
        for s in lmaxs_CMB.keys():
            print(s + ': ' + str(lmaxs_CMB[s]))

    # If nlev_p is arraylike
    if isinstance(nlev_p, (np.ndarray, list)):
        if isinstance(nlev_p, list):
            nlev_p = np.array(nlev_p)

        # e = b (scale-dependent) (catching shape=(1,lmax) and shape=(1,2))
        if nlev_p.shape[0] == 1:
            nlev_e = nlev_p[0]
            nlev_b = nlev_p[0]
        # e =/= b (scale-dependent) (catching shape=(2,lmax) and shape=(2,2))
        elif nlev_p.shape[0] == 2:
            nlev_e = nlev_p[0]
            nlev_b = nlev_p[1]
        # e = b scale-dependent noise (catching shape=lmax and shape=1)
        else:
            nlev_e = nlev_p
            nlev_b = nlev_p

    # If nlev_p is single number
    elif isinstance(nlev_p, (float, int, np.float, np.int)):
        nlev_e = nlev_p
        nlev_b = nlev_p
    else:
        print("Not sure about the datatype of your nlev_p: {}".format(type(nlev_p)))

    lmax_ivf = np.max(list(lmaxs_CMB.values()))
    if isinstance(lmin_CMB, dict):
        lmins_ivf = lmin_CMB
        print("Seeing lmin's:")
        for s in lmins_ivf.keys():
            print(s + ': ' + str(lmins_ivf[s]))
    else:
        lmins_ivf = {s: max(lmin_CMB, 1) for s in ['t', 'e', 'b']}

    lmax_qlm = lmax_out or lmax_ivf
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_len = cls_len or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_weight = cls_weight or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_sky = cls_sky or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_filt = cls_filt or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

    # We consider here TT, Pol-only and the GMV comb if joint_TP is set
    qe_keys = [ksource + 'tt', ksource + '_p']
    if not joint_TP:
        qe_keys.append(ksource)

    # Simple noise model. Can feed here something more fancy if desired
    transf = hp.gauss_beam(beam_fwhm / 60. / 180. * np.pi, lmax=lmax_ivf)
    Noise_L_T = (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2
    Noise_L_E = (nlev_e / 60. / 180. * np.pi) ** 2 / transf ** 2
    Noise_L_B = (nlev_b / 60. / 180. * np.pi) ** 2 / transf ** 2

    cls_dat = {}
    cls_filter = {}
    for cls, source in ((cls_dat, cls_sky), (cls_filter, cls_filt)):
        # Data power spectra
        cls.update({
            'tt': (source['tt'][:lmax_ivf + 1] + Noise_L_T),
            'ee': (source['ee'][:lmax_ivf + 1] + Noise_L_E),
            'bb': (source['bb'][:lmax_ivf + 1] + Noise_L_B),
            'te': np.copy(source['te'][:lmax_ivf + 1])})

        for s in cls.keys():
            cls[s][min(lmaxs_CMB[s[0]], lmaxs_CMB[s[1]]) + 1:] *= 0.
            cls[s][:max(lmins_ivf[s[0]], lmins_ivf[s[1]])] *= 0.

    # (C+N)^{-1} filter spectra
    # For independent T and P filtering, this is really just 1/ (C+ N), diagonal in T, E, B space
    fal_sepTP = {spec: utils.cli(cls_filter[spec]) for spec in ['tt', 'ee', 'bb']}
    # Spectra of the inverse-variance filtered maps
    # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
    cls_ivfs_sepTP = utils.cls_dot([fal_sepTP, cls_dat, fal_sepTP], ret_dict=True)



    # For joint TP filtering, fals is matrix inverse
    fal_jtTP = utils.cl_inverse(cls_filter)
    # When cls_dat = fals, then the filtered map spectra cls_ivfs is the same as fals.
    # However, if the data spectra do not match the filter, this becomes:
    cls_ivfs_jtTP = utils.cls_dot([fal_jtTP, cls_dat, fal_jtTP], ret_dict=True)
    if wfleg_Tcut is not None and wfleg_Tcut < lmaxs_CMB['t']: # Applying high-l cut on T Wiener-filtered leg
        fal_sepTP_b = deepcopy(fal_sepTP)
        fal_sepTP_b['tt'][wfleg_Tcut + 1:] *= 0
        cls_temp = deepcopy(cls_dat)
        for k in cls_temp:
            if 't' in k:
                cls_temp[k][wfleg_Tcut+1:] *= 0

        fal_jtTP_b = utils.cl_inverse(cls_temp)
        cls_ivfs_sepTP_ab = utils.cls_dot([fal_sepTP, cls_dat, fal_sepTP_b], ret_dict=True)
        cls_ivfs_sepTP_ba = utils.cls_dot([fal_sepTP_b, cls_dat, fal_sepTP], ret_dict=True)
        cls_ivfs_sepTP_bb = utils.cls_dot([fal_sepTP_b, cls_dat, fal_sepTP_b], ret_dict=True)
        cls_ivfs_jtTP_ab = utils.cls_dot([fal_jtTP, cls_dat, fal_jtTP_b], ret_dict=True)
        cls_ivfs_jtTP_ba = utils.cls_dot([fal_jtTP_b, cls_dat, fal_jtTP], ret_dict=True)
        cls_ivfs_jtTP_bb = utils.cls_dot([fal_jtTP_b, cls_dat, fal_jtTP_b], ret_dict=True)

    else:
        fal_sepTP_b, fal_jtTP_b = fal_sepTP, fal_jtTP
        cls_ivfs_sepTP_ab, cls_ivfs_jtTP_ab = cls_ivfs_sepTP, cls_ivfs_jtTP
        cls_ivfs_sepTP_ba, cls_ivfs_jtTP_ba = cls_ivfs_sepTP, cls_ivfs_jtTP
        cls_ivfs_sepTP_bb, cls_ivfs_jtTP_bb = cls_ivfs_sepTP, cls_ivfs_jtTP

    for cls in [fal_sepTP, fal_jtTP, fal_sepTP_b, fal_jtTP_b,
                cls_ivfs_sepTP, cls_ivfs_jtTP,
                cls_ivfs_sepTP_ab, cls_ivfs_jtTP_ab,
                cls_ivfs_sepTP_ba, cls_ivfs_jtTP_ba,
                cls_ivfs_sepTP_bb, cls_ivfs_jtTP_bb]:
        for cl_key, cl_val in cls.items():
            cls[cl_key][:max(1, lmins_ivf[cl_key[0]], lmins_ivf[cl_key[1]])] *= 0.

    N0s = {}
    N0_curls = {}
    for qe_key in qe_keys:
        # This calculates the unormalized QE gradient (G), curl (C) variances and covariances:
        # (GC and CG is zero for most estimators)
        NG, NC, NGC, NCG = nhl.get_nhl(qe_key, qe_key, cls_weight, cls_ivfs_sepTP, lmax_ivf, lmax_ivf,
                                       lmax_out=lmax_qlm, cls_ivfs_ab=cls_ivfs_sepTP_ab, cls_ivfs_bb=cls_ivfs_sepTP_bb, cls_ivfs_ba=cls_ivfs_sepTP_ba)
        # Calculation of the G to G, C to C, G to C and C to G QE responses (again, cross-terms are typically zero)
        RG, RC, RGC, RCG = qresp.get_response(qe_key, lmax_ivf, ksource, cls_weight, cls_glen, fal_sepTP,
                                              lmax_qlm=lmax_qlm, fal_leg2=fal_sepTP_b)

        # Gradient and curl noise terms
        N0s[qe_key] = utils.cli(RG ** 2) * NG
        N0_curls[qe_key] = utils.cli(RC ** 2) * NC

    if joint_TP:
        NG, NC, NGC, NCG = nhl.get_nhl(ksource, ksource, cls_weight, cls_ivfs_jtTP, lmax_ivf, lmax_ivf,
                                       lmax_out=lmax_qlm, cls_ivfs_ab=cls_ivfs_jtTP_ab, cls_ivfs_bb=cls_ivfs_jtTP_bb, cls_ivfs_ba=cls_ivfs_jtTP_ba)
        RG, RC, RGC, RCG = qresp.get_response(ksource, lmax_ivf, ksource, cls_weight, cls_len, fal_jtTP,
                                              lmax_qlm=lmax_qlm, fal_leg2=fal_jtTP_b)
        N0s[ksource] = utils.cli(RG ** 2) * NG
        N0_curls[ksource] = utils.cli(RC ** 2) * NC

    return N0s, N0_curls


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
        cldd *= np.arange(len(cldd)) ** 2 * np.arange(1, len(cldd) + 1, dtype=float) ** 2 / (2. * np.pi)
    return dls, cldd


def dls2cls(dls):
    """Inverse operation to cls2dls"""
    assert dls.shape[1] == 4
    lmax = dls.shape[0] - 1
    cls = {}
    refac = 2. * np.pi * utils.cli(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float))
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k] = dls[:, i] * refac
    return cls


def get_N0_iter(qe_key: str, nlev_t: float or np.ndarray, nlev_p: float or np.ndarray, beam_fwhm: float,
                cls_unl_fid: dict, lmin_cmb: int or dict, lmax_cmb: int or dict, itermax, cls_unl_dat=None,
                lmax_qlm=None, ret_delcls=False, datnoise_cls: dict or None = None, ret_curl=False,
                rho_sqd_ext: float or np.ndarray = 0, filter_E=False):
    r"""Iterative lensing-N0 estimate

        This calculates iteratively partially lensed spectra and lensing noise levels.
        At each iteration this takes out the resolved part of the lenses (ignoring N1) and recomputes a N0

        Args:
            qe_key: QE estimator key
            nlev_t: temperature noise level (in :math:`\mu `K-arcmin) (an array can be passed for scale-dependent noise level)
            nlev_p: polarisation noise level (in :math:`\mu `K-arcmin)(an array can be passed for scale-dependent noise level) can also be for E and B noise separately, in which case the array is of shape (2,lmax_CMB)), and E noise expected at first
            beam_fwhm: Gaussian beam full width half maximum in arcmin
            cls_unl_fid(dict): unlensed CMB power spectra
            lmin_cmb: minimal CMB multipole used in the QE
            lmax_cmb: maximal CMB multipole used in the QE
            itermax: number of iterations to perform
            lmax_qlm(optional): maximum lensing multipole to consider. Defaults to 2 lmax_ivf
            ret_delcls(optional): returns the partially delensed CMB cls as well if set
            datnoise_cls(optional): feeds in custom noise spectra to the data. The nlevs and beam only apply to the filtering in this case
            ret_curl(optional): also returns lensing curl N0s
            rho_sqd_ext (optional scalar or array): option cross-correlation coefficent squared of external tracer to use for partial delensing
            filter_E (optional): do linear delensing by substractng B estimate constructed by partial lensing of Wiener-filtered unlensed E
        Returns
            tuple of arrays of shape (itermax + 1, lmax_qlm + 1) with all iterated N0s. First entry is standard N0. See code below.


        Note:
            this requires camb python package for the lensed spectra calc.

     """
    assert qe_key in ['p_p', 'p', 'ptt'], qe_key
    try:
        from camb.correlations import lensed_cls
    except ImportError:
        assert 0, "could not import camb.correlations.lensed_cls"

    if isinstance(lmax_cmb, dict):
        lmaxs_ivf = lmax_cmb
        print("Seeing lmax's:")
        for s in lmaxs_ivf.keys():
            print(s + ': ' + str(lmaxs_ivf[s]))
    else:
        lmaxs_ivf = {s: lmax_cmb for s in ['t', 'e', 'b']}

    if isinstance(lmin_cmb, dict):
        lmins_ivf = lmin_cmb
        print("Seeing lmin's:")
        for s in lmins_ivf.keys():
            print(s + ': ' + str(lmins_ivf[s]))
    else:
        lmins_ivf = {s: max(lmin_cmb, 1) for s in ['t', 'e', 'b']}

    lmax_ivf = np.max(list(lmaxs_ivf.values()))
    if lmax_qlm is None:
        lmax_qlm = 2 * lmax_ivf
    lmax_qlm = min(lmax_qlm, 2 * lmax_ivf)
    transfi2 = utils.cli(hp.gauss_beam(beam_fwhm / 180. / 60. * np.pi, lmax=lmax_ivf)) ** 2
    llp2 = np.arange(lmax_qlm + 1, dtype=float) ** 2 * np.arange(1, lmax_qlm + 2, dtype=float) ** 2 / (2. * np.pi)

    # If nlev_p is arraylike
    if isinstance(nlev_p, (np.ndarray, list)):
        if isinstance(nlev_p, list):
            nlev_p = np.array(nlev_p)

        # e = b (scale-dependent) (catching shape=(1,lmax) and shape=(1,2))
        if nlev_p.shape[0] == 1:
            nlev_e = nlev_p[0]
            nlev_b = nlev_p[0]
        # e =/= b (scale-dependent) (catching shape=(2,lmax) and shape=(2,2))
        elif nlev_p.shape[0] == 2:
            nlev_e = nlev_p[0]
            nlev_b = nlev_p[1]
        # e = b (scale-dependent) noise (catching shape=lmax and shape=1)
        else:
            nlev_e = nlev_p
            nlev_b = nlev_p

    # If nlev_p is single number
    elif isinstance(nlev_p, (float, int, np.float, np.int)):
        nlev_e = nlev_p
        nlev_b = nlev_p
    else:
        print("Not sure about the datatype of your nlev_p: {}".format(type(nlev_p)))

    if not np.isscalar(rho_sqd_ext):
        rho_sqd_ext = rho_sqd_ext[:lmax_qlm + 1]

    if datnoise_cls is None:
        datnoise_cls = dict()
        if qe_key in ['ptt', 'p']:
            datnoise_cls['tt'] = (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
        if qe_key in ['p_p', 'p']:
            datnoise_cls['ee'] = (nlev_e * np.pi / 180. / 60.) ** 2 * transfi2
            datnoise_cls['bb'] = (nlev_b * np.pi / 180. / 60.) ** 2 * transfi2
    N0s_biased = []
    N0s_unbiased = []
    N0s_biased_cc = []
    N0s_unbiased_cc = []
    delcls_fid = []
    delcls_true = []

    N0_unbiased = np.inf
    dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
    if cls_unl_dat is None:
        cls_unl_dat = cls_unl_fid

    if filter_E:
        cls_len_fid = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
        if cls_unl_dat is None:
            cls_len_true = cls_len_fid
        else:
            dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
            cls_len_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))
        cls_plen_true = cls_len_true
    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = rho_sqd_ext
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            N0_now = llp2 * N0_unbiased[:lmax_qlm + 1]
            rho_sqd_phi[:lmax_qlm + 1] = ((1 - rho_sqd_ext) * cldd_true[:lmax_qlm + 1] + rho_sqd_ext * N0_now) * \
                                         utils.cli((1 - rho_sqd_ext) * cldd_true[:lmax_qlm + 1] + N0_now)

        if filter_E:
            assert qe_key in ['p_p']
            if it == 0:
                print('including imperfect knowledge of E in iterations')
            slic = slice(lmins_ivf['e'], lmaxs_ivf['e'] + 1)
            rho_sqd_E = np.zeros(len(dls_unl_true[:, 1]))
            rho_sqd_E[slic] = cls_unl_dat['ee'][slic] * utils.cli(cls_plen_true['ee'][slic] + datnoise_cls['ee'][slic])
            dls_unl_fid[:, 1] *= rho_sqd_E
            dls_unl_true[:, 1] *= rho_sqd_E
            cldd_fid *= rho_sqd_phi
            cldd_true *= rho_sqd_phi

            cls_plen_fid_delta = dls2cls(lensed_cls(dls_unl_fid, cldd_fid, delta_cls=True))
            cls_plen_true_delta = dls2cls(lensed_cls(dls_unl_true, cldd_true, delta_cls=True))
            cls_plen_fid = {ck: cls_len_fid[ck] - (cls_plen_fid_delta[ck]) for ck in cls_len_fid.keys()}
            cls_plen_true = {ck: cls_len_true[ck] - (cls_plen_true_delta[ck]) for ck in cls_len_true.keys()}
        else:
            cldd_true *= (1. - rho_sqd_phi)  # The true residual lensing spec.
            cldd_fid *= (1. - rho_sqd_phi)  # What I think the residual lensing spec is
            cls_plen_fid = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
            cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))

        cls_filt = cls_plen_fid
        cls_f = cls_plen_true
        fal = {}
        dat_delcls = {}
        if qe_key in ['ptt', 'p']:
            fal['tt'] = cls_filt['tt'][:lmax_ivf + 1] + (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['tt'] = cls_plen_true['tt'][:lmax_ivf + 1] + datnoise_cls['tt'][:lmax_ivf + 1]
        if qe_key in ['p_p', 'p']:
            fal['ee'] = cls_filt['ee'][:lmax_ivf + 1] + (nlev_e * np.pi / 180. / 60.) ** 2 * transfi2
            fal['bb'] = cls_filt['bb'][:lmax_ivf + 1] + (nlev_b * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['ee'] = cls_plen_true['ee'][:lmax_ivf + 1] + datnoise_cls['ee']
            dat_delcls['bb'] = cls_plen_true['bb'][:lmax_ivf + 1] + datnoise_cls['bb']
        if qe_key in ['p']:
            fal['te'] = np.copy(cls_filt['te'][:lmax_ivf + 1])
            dat_delcls['te'] = np.copy(cls_plen_true['te'][:lmax_ivf + 1])
        for spec in fal.keys():
            fal[spec][min(lmaxs_ivf[spec[0]], lmaxs_ivf[spec[1]]) + 1:] *= 0
        for spec in dat_delcls.keys():
            dat_delcls[spec][min(lmaxs_ivf[spec[0]], lmaxs_ivf[spec[1]]) + 1:] *= 0

        fal = utils.cl_inverse(fal)
        for cl_key, cl_val in fal.items():
            fal[cl_key][:max(lmins_ivf[cl_key[0]], lmins_ivf[cl_key[1]])] *= 0.
        for cl_key, cl_val in dat_delcls.items():
            dat_delcls[cl_key][:max(lmins_ivf[cl_key[0]], lmins_ivf[cl_key[1]])] *= 0.
        cls_ivfs = utils.cls_dot([fal, dat_delcls, fal], ret_dict=True)
        cls_w = deepcopy(cls_plen_fid)
        for spec in cls_w.keys():  # in principle not necessary
            cls_w[spec][:max(lmins_ivf[spec[0]], lmins_ivf[spec[1]])] *= 0.
            cls_w[spec][min(lmaxs_ivf[spec[0]], lmaxs_ivf[spec[1]]) + 1:] *= 0

        n_gg, n_cc = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0:2]
        r_gg_true, r_cc_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0:2]
        (r_gg_fid, r_cc_fid) = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_w, fal, lmax_qlm=lmax_qlm)[
                               0:2] if cls_f is not cls_w else (r_gg_true, r_cc_true)
        N0_biased = n_gg * utils.cli(r_gg_fid ** 2)  # N0 of possibly biased (by Rtrue / Rfid) QE estimator
        N0_unbiased = n_gg * utils.cli(
            r_gg_true ** 2)  # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased
        N0_biased_cc = n_cc * utils.cli(r_cc_fid ** 2)  # N0 of possibly biased (by Rtrue / Rfid) QE estimator
        N0_unbiased_cc = n_cc * utils.cli(
            r_cc_true ** 2)  # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased

        N0s_biased.append(N0_biased)
        N0s_unbiased.append(N0_unbiased)
        N0s_biased_cc.append(N0_biased_cc)
        N0s_unbiased_cc.append(N0_unbiased_cc)

        cls_plen_true['pp'] = cldd_true * utils.cli(
            np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 / (2. * np.pi))
        cls_plen_fid['pp'] = cldd_fid * utils.cli(
            np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 / (2. * np.pi))

        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)
    if ret_curl:
        return (np.array(N0s_biased), np.array(N0s_unbiased), np.array(N0s_unbiased_cc),
                np.array(N0s_biased_cc)) if not ret_delcls else ((
            np.array(N0s_biased), np.array(N0s_unbiased), np.array(N0s_unbiased_cc), np.array(N0s_biased_cc),
            delcls_fid,
            delcls_true))
    else:
        return (np.array(N0s_biased), np.array(N0s_unbiased)) if not ret_delcls else (
            (np.array(N0s_biased), np.array(N0s_unbiased), delcls_fid, delcls_true))
