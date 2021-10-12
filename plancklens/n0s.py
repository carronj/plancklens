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


def get_N0(beam_fwhm=1.4, nlev_t=5., nlev_p=None, lmax_CMB=3000,  lmin_CMB=100, lmax_out=None,
           cls_len:dict or None =None, cls_weight:dict or None=None,
           joint_TP=True, ksource='p'):
    r"""Example function to calculates reconstruction noise levels for a bunch of quadratic estimators

        Args:
            beam_fwhm: beam fwhm in arcmin
            nlev_t: T white noise level in uK-arcmin
            nlev_p: P white noise level in uK-arcmin (defaults to root(2) nlevt)
            lmax_CMB: max. CMB multipole used in the QE
            lmin_CMB: min. CMB multipole used in the QE
            lmax_out: max lensing 'L' multipole calculated
            cls_len: CMB spectra entering the sky response to the anisotropy (defaults to FFP10 lensed CMB spectra)
            cls_weight: CMB spectra entering the QE weights (defaults to FFP10 lensed CMB spectra)
            joint_TP: if True include calculation of the N0s for the GMV estimator (incl. joint T and P filtering)
            ksource: anisotropy source to consider (defaults to 'p', lensing)

        Returns:
            N0s array for the lensing gradient and curl modes for  the T-only, P-onl and (G)MV estimators

        Prompted by AL
    """
    if nlev_p is None:
        nlev_p = nlev_t * np.sqrt(2)

    lmax_ivf = lmax_CMB
    lmin_ivf = lmin_CMB
    lmax_qlm = lmax_out or lmax_ivf
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_len = cls_len or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
    cls_weight = cls_weight or utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

    # We consider here TT, Pol-only and the GMV comb if joint_TP is set
    qe_keys = [ksource + 'tt', ksource + '_p']
    if not joint_TP:
        qe_keys.append(ksource)

    # Simple white noise model. Can feed here something more fancy if desired
    transf = hp.gauss_beam(beam_fwhm / 60. / 180. * np.pi, lmax=lmax_ivf)
    Noise_L_T = (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2
    Noise_L_P = (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2

    # Data power spectra
    cls_dat = {
        'tt': (cls_len['tt'][:lmax_ivf + 1] + Noise_L_T),
        'ee': (cls_len['ee'][:lmax_ivf + 1] + Noise_L_P),
        'bb': (cls_len['bb'][:lmax_ivf + 1] + Noise_L_P),
        'te': np.copy(cls_len['te'][:lmax_ivf + 1])}

    # (C+N)^{-1} filter spectra
    # For independent T and P filtering, this is really just 1/ (C+ N), diagonal in T, E, B space
    fal_sepTP = {spec: utils.cli(cls_dat[spec]) for spec in ['tt', 'ee', 'bb']}

    # Spectra of the inverse-variance filtered maps
    # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
    # Here we have assumed data cls match perfectly filtering cls, so that for independent TP filter:
    cls_ivfs_sepTP = {'tt': fal_sepTP['tt'].copy(),
                      'ee': fal_sepTP['ee'].copy(),
                      'bb': fal_sepTP['bb'].copy(),
                      'te': cls_len['te'][:lmax_ivf + 1] * fal_sepTP['tt'] * fal_sepTP['ee']}

    # For joint TP filtering, fals is matrix inverse
    fal_jtTP = utils.cl_inverse(cls_dat)
    # since cls_dat = fals, cls_ivfs = fals. If the data spectra do not match the filter, this must be changed:
    cls_ivfs_jtTP = utils.cl_inverse(cls_dat)

    for cls in [fal_sepTP, fal_jtTP, cls_ivfs_sepTP, cls_ivfs_jtTP]:
        for cl in cls.values():
            cl[:max(1, lmin_ivf)] *= 0.

    N0s = {}
    N0_curls = {}
    for qe_key in qe_keys:
        # This calculates the unormalized QE gradient (G), curl (C) variances and covariances:
        # (GC and CG is zero for most estimators)
        NG, NC, NGC, NCG = nhl.get_nhl(qe_key, qe_key, cls_weight, cls_ivfs_sepTP, lmax_ivf, lmax_ivf,
                                       lmax_out=lmax_qlm)
        # Calculation of the G to G, C to C, G to C and C to G QE responses (again, cross-terms are typically zero)
        RG, RC, RGC, RCG = qresp.get_response(qe_key, lmax_ivf, ksource, cls_weight, cls_len, fal_sepTP,
                                              lmax_qlm=lmax_qlm)

        # Gradient and curl noise terms
        N0s[qe_key] = utils.cli(RG ** 2) * NG
        N0_curls[qe_key] = utils.cli(RC ** 2) * NC

    if joint_TP:
        NG, NC, NGC, NCG = nhl.get_nhl(ksource, ksource, cls_weight, cls_ivfs_jtTP, lmax_ivf, lmax_ivf,
                                       lmax_out=lmax_qlm)
        RG, RC, RGC, RCG = qresp.get_response(ksource, lmax_ivf, ksource, cls_weight, cls_len, fal_jtTP,
                                              lmax_qlm=lmax_qlm)
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
                lmax_qlm=None, ret_delcls=False, datnoise_cls:dict or None=None):
    r"""Iterative lensing-N0 estimate

        Calculates iteratively partially lensed spectra and lensing noise levels.
        This uses the python camb package to get the partially lensed spectra.

        At each iteration this takes out the resolved part of the lenses and recomputes a N0

        Args:
            qe_key: QE estimator key
            nlev_t: temperature noise level (in :math:`\mu `K-arcmin)
            nlev_p: polarisation noise level (in :math:`\mu `K-arcmin)
            beam_fwhm: Gaussian beam full width half maximum in arcmin
            cls_unl_fid(dict): unlensed CMB power spectra
            lmin_ivf: minimal CMB multipole used in the QE
            lmax_ivf: maximal CMB multipole used in the QE
            itermax: number of iterations to perform
            lmax_qlm(optional): maximum lensing multipole to consider. Defaults to 2 lmax_ivf
            ret_delcls(optional): returns the partially delensed CMB cls as well if set
            datnoise_cls(optional): feeds in custom noise spectra to the data. The nlevs and beam only apply to the filtering in this case

        Returns
            Array of shape (itermax + 1, lmax_qlm + 1) with all iterated N0s. First entry is standard N0.


        Note:
            this is requiring camb python package for the lensed spectra calc.

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
    delcls_fid = []
    delcls_true = []

    N0_unbiased = np.inf
    if cls_unl_dat is None:
        cls_unl_dat = cls_unl_fid

    for irr, it in utils.enumerate_progress(range(itermax + 1)):
        dls_unl_true, cldd_true = cls2dls(cls_unl_dat)
        dls_unl_fid, cldd_fid = cls2dls(cls_unl_fid)
        if it == 0:
            rho_sqd_phi = 0.
        else:
            # The cross-correlation coefficient is identical for the Rfid-biased QE or the rescaled one
            rho_sqd_phi = np.zeros(len(cldd_true))
            rho_sqd_phi[:lmax_qlm +1] =   cldd_true[:lmax_qlm + 1] * utils.cli(cldd_true[:lmax_qlm + 1] + llp2 * N0_unbiased[:lmax_qlm+1])

        cldd_true *= (1. - rho_sqd_phi)  # The true residual lensing spec.
        cldd_fid *= (1. - rho_sqd_phi)  # What I think the residual lensing spec is
        cls_plen_fid  = dls2cls(lensed_cls(dls_unl_fid, cldd_fid))
        cls_plen_true = dls2cls(lensed_cls(dls_unl_true, cldd_true))

        cls_filt = cls_plen_fid
        cls_w = cls_plen_fid
        cls_f = cls_plen_true
        fal = {}
        dat_delcls = {}
        if qe_key in ['ptt', 'p']:
            fal['tt'] = cls_filt['tt'][:lmax_ivf + 1] + (nlev_t * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['tt'] = cls_plen_true['tt'][:lmax_ivf + 1] + datnoise_cls['ee']
        if qe_key in ['p_p', 'p']:
            fal['ee'] = cls_filt['ee'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            fal['bb'] = cls_filt['bb'][:lmax_ivf + 1] + (nlev_p * np.pi / 180. / 60.) ** 2 * transfi2
            dat_delcls['ee'] = cls_plen_true['ee'][:lmax_ivf + 1] + datnoise_cls['ee']
            dat_delcls['bb'] = cls_plen_true['bb'][:lmax_ivf + 1] + datnoise_cls['bb']
        if qe_key in ['p']:
            fal['te'] = np.copy(cls_filt['te'][:lmax_ivf + 1])
            dat_delcls['te'] = np.copy(cls_plen_true['te'][:lmax_ivf + 1])
        fal = utils.cl_inverse(fal)
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

        n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
        r_gg_true = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0]
        r_gg_fid = qresp.get_response(qe_key, lmax_ivf, 'p', cls_w, cls_w, fal, lmax_qlm=lmax_qlm)[0] if cls_f is not cls_w else r_gg_true
        N0_biased = n_gg * utils.cli(r_gg_fid ** 2) # N0 of possibly biased (by Rtrue / Rfid) QE estimator
        N0_unbiased = n_gg * utils.cli(r_gg_true ** 2) # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased
        N0s_biased.append(N0_biased)
        N0s_unbiased.append(N0_unbiased)
        cls_plen_true['pp'] =  cldd_true *utils.cli(np.arange(len(cldd_true)) ** 2 * np.arange(1, len(cldd_true) + 1, dtype=float) ** 2 /  (2. * np.pi))
        cls_plen_fid['pp'] =  cldd_fid *utils.cli(np.arange(len(cldd_fid)) ** 2 * np.arange(1, len(cldd_fid) + 1, dtype=float) ** 2 /  (2. * np.pi))

        delcls_fid.append(cls_plen_fid)
        delcls_true.append(cls_plen_true)


    return (np.array(N0s_biased), np.array(N0s_unbiased)) if not ret_delcls else ((np.array(N0s_biased), np.array(N0s_unbiased), delcls_fid, delcls_true))