"""Module for QE response calculations.

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
        etc

    
    Bias-hardening can be included by inserting '_bh_'. 
        E.g. 'ptt_bh_s' is the lensing TT QE bias-hardened against point source contamination using the 'stt' estimator


    Responses method takes as input the QE weights (typically the lensed CMB spectra) and the filtering cls ('fals')
    which describes the filtering applied to the maps (the :math:`(C + N)^{-1}` operation)
    See *examples/get_N0s.py* to see how these are typically calculated for independent or joint temperature and polarization filtering


"""

from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import pickle as pk
from scipy.special import gammaln

from plancklens import utils as ut, utils_spin as uspin, utils_qe as uqe
from plancklens.helpers import mpi, sql

def _clinv(cl):
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1./cl[ii]
    return ret

def get_qes(qe_key, lmax, cls_weight, lmax2=None, transf=None):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_weight (dict): CMB spectra entering the weights (when relevant).
        lmax2 (int, optional): weight on the second leg are built up to lmax2 (default to lmax)

    The weights are defined by their action on the inverse-variance filtered spin-weight $ _{s}\bar X_{lm}$.

    """
    if lmax2 is None: lmax2 = lmax
    if qe_key[0] in ['p', 'x', 'a', 'f', 's']:
        if qe_key in ['ptt', 'xtt', 'att', 'ftt', 'stt']:
            s_lefts= [0]
        elif qe_key in ['p_p', 'x_p', 'a_p', 'f_p']:
            s_lefts= [-2, 2]
        else:
            s_lefts = [0, -2, 2]
        qes = []
        s_rights_in = s_lefts
        for s_left in s_lefts:
            for sin in s_rights_in:
                sout = -s_left
                s_qe, irr1, cl_sosi, cL_out =  get_covresp(qe_key[0], sout, sin, cls_weight, lmax2, transf=transf)
                if np.any(cl_sosi):
                    lega = uqe.qeleg(s_left, s_left, 0.5 *(1. + (s_left == 0)) * np.ones(lmax + 1, dtype=float))
                    legb = uqe.qeleg(sin, sout + s_qe, 0.5 * (1. + (sin == 0)) * 2 * cl_sosi)
                    qes.append(uqe.qe(lega, legb, cL_out))
        if len(qe_key) == 1 or qe_key[1:] in ['tt', '_p']:
            return uqe.qe_simplify(qes)
        elif qe_key[1:] in ['te', 'et', 'tb', 'bt', 'ee', 'eb', 'be', 'bb']:
            return uqe.qe_simplify(uqe.qe_proj(qes, qe_key[1], qe_key[2]))
        elif qe_key[1:] in ['_te', '_tb', '_eb']:
            return uqe.qe_simplify(uqe.qe_proj(qes, qe_key[2], qe_key[3]) + uqe.qe_proj(qes, qe_key[3], qe_key[2]))
        else:
            assert 0, 'qe key %s  not recognized'%qe_key
    elif qe_key in ['ntt']:
        lega = uqe.qeleg(0, 0, 1   * _clinv(transf[:lmax + 1]))
        legb = uqe.qeleg(0, 0, 0.5 * _clinv(transf[:lmax + 1]))  # Weird norm to match PS case for no beam
        qes = [uqe.qe(lega, legb, lambda L: np.ones(len(L), dtype=float))]
        return uqe.qe_simplify(qes)
    elif qe_key in ['ktt']:
        ls = np.arange(1, lmax + 3)
        dlnDldlnl = ls[:-1] * np.diff(np.log(cls_weight['tt'][ls] * ls * (ls + 1)))
        lega = uqe.qeleg(0, 0, np.ones(lmax + 1, dtype=float))
        legb = uqe.qeleg(0, 0, 0.5 * cls_weight['tt'][:lmax+1] * dlnDldlnl)
        qes = [uqe.qe(lega, legb, lambda L: -L * (L + 1.))]
        return uqe.qe_simplify(qes)
    else:
        assert 0, qe_key + ' not implemented'


def get_resp_legs(source, lmax):
    r"""Defines the responses terms for a CMB map anisotropy source.

    Args:
        source (str): anisotropy source (e.g. 'p', 'f', ...).
        lmax (int): responses are given up to lmax.

    Returns:
        4-tuple (r, rR, -rR, cL):  source spin response *r* (positive or zero),
        the harmonic responses for +r and -r (2 1d-arrays), and the scaling between the G/C modes
        and the potentials of interest. (for lensing, cL is given by :math:`L\sqrt{L (L + 1)}`).

    """
    if source in ['p', 'x']:
        # lensing (gradient and curl): _sX -> _sX -  1/2 alpha_1 \eth _sX - 1/2 \alpha_{-1} \bar \eth _sX
        return {s : (1, -0.5 * uspin.get_spin_lower(s, lmax), -0.5 * uspin.get_spin_raise(s, lmax),
                     lambda ell : uspin.get_spin_raise(0, np.max(ell))[ell]) for s in [0, -2, 2]}
    if source == 'f': # Modulation: _sX -> _sX + f _sX.
        return {s : (0, 0.5 * np.ones(lmax + 1, dtype=float), 0.5 * np.ones(lmax + 1, dtype=float),
                        lambda ell: np.ones(len(ell), dtype=float)) for s in [0, -2, 2]}
    if source in ['a', 'a_p']: # Polarisation rotation _\pm 2 X ->  _\pm 2 X + \mp 2 i a _\pm 2 X
        ret = {s: (0,  -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                       -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                        lambda ell: np.ones(len(ell), dtype=float)) for s in [-2, 2]}
        ret[0]=(0, np.zeros(lmax + 1, dtype=float),
                   np.zeros(lmax + 1, dtype=float),
                   lambda ell: np.ones(len(ell), dtype=float))
        return ret

    assert 0, source + ' response legs not implemented'

def get_covresp(source, s1, s2, cls, lmax, transf=None):
    r"""Defines the responses terms for a CMB covariance anisotropy source.

        \delta < s_d(n) _td^*(n')> \equiv
        _r\alpha(n) W^{r, st}_l _{s - r}Y_{lm}(n) _tY^*_{lm}(n') +
        _r\alpha^*(n') W^{r, ts}_l _{s}Y_{lm}(n) _{t-r}Y^*_{lm}(n')

    """
    if source in ['p','x', 'f', 'a', 'a_p']:
        # Lensing, modulation, or pol. rotation field from the field representation
        s_source, prR, mrR, cL_scal = get_resp_legs(source, lmax)[s1]
        coupl = uspin.spin_cls(s1, s2, cls)[:lmax + 1]
        return s_source, prR * coupl, mrR * coupl, cL_scal
    elif source in ['stt', 's']:
        # Point source 'S^2': Cov -> Cov + B delta_nn' S^2(n) B^\dagger on the diagonal.
        # From the def. there are actually 4 identical W terms hence a factor 1/4.
        cond = s1 == 0 and s2 == 0
        s_source = 0
        prR = 0.25 * cond * np.ones(lmax + 1, dtype=float)
        mrR = 0.25 * cond * np.ones(lmax + 1, dtype=float)
        cL_scal =  lambda ell : np.ones(len(ell), dtype=float)
        return s_source, prR, mrR, cL_scal
    elif source in ['ntt', 'n']:
        assert transf is not None
        cL_scal =  lambda ell : np.ones(len(ell), dtype=float)
        assert 0, 'dont think this parametrization works here'
        return s_source, prR, mrR, cL_scal
    else:
        assert 0, 'source ' + source + ' cov. response not implemented'

def qe_spin_data(qe_key):
    """Returns out and in spin-weights of quadratic estimator from its quadratic estimator key.

        Output is an integer >= 0 (spin), a letter 'G' or 'C' if gradient or curl mode estimator, the
        unordered list of unique spins (>= 0) input to the estimator, and the spin-1 qe key.

    """
    if qe_key in ['ntt']:
        return 0, 'G', [0], 'n'
    qes = get_qes(qe_key, 10, {k:np.ones(11 + 4, dtype=float) for k in ['tt', 'te', 'ee', 'bb']}) #Hack
    spins_out = [qe.leg_a.spin_ou + qe.leg_b.spin_ou for qe in qes]
    spins_in = np.unique(np.abs([qe.leg_a.spin_in for qe in qes] + [qe.leg_b.spin_in for qe in qes]))
    assert len(np.unique(spins_out)) == 1, spins_out
    assert spins_out[0] >= 0, spins_out[0]
    if spins_out[0] > 0: assert qe_key[0] in ['x', 'p'], 'non-zero spin anisotropy ' + qe_key +  ' not implemented ?'
    return spins_out[0], 'C' if qe_key[0] == 'x' else 'G', spins_in, 'p' if qe_key[0] == 'x' else qe_key[0]


class resp_lib_simple:
    r"""QE responses calculation library.

            This wraps the *get_response* function and caches the outputs.

        Args:
            lib_dir: outputs are cached in this directory
            lmax_ivf: max. CMB multipole used in the QE
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            cls_cmb(dict): CMB spectra entering the CMB response (in principle lensed spectra, or grad-lensed spectra)
            fal(dict): (isotropic approximation to the) filtering cls.
                       e.g. fal['tt'] :math:`= \frac {1} {C^{TT}_\ell  +  N^{\rm TT}_\ell / b^2_\ell}`
                       for temperature if filtered independently from polarization.
            lmax_qlm(optional): responses are calculated up to this multipole. Defaults to lmax_ivf + lmax_ivf2

    """
    def __init__(self, lib_dir, lmax_ivf, cls_weight, cls_cmb, fal, lmax_qlm, transf=None):
        self.lmax_qe = lmax_ivf
        self.lmax_qlm = lmax_qlm
        self.cls_weight = cls_weight
        self.cls_cmb = cls_cmb
        self.fal = fal
        self.transf = transf
        self.lib_dir = lib_dir

        fn_hash = os.path.join(lib_dir, 'resp_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        ut.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())
        self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))

    def hashdict(self):
        ret = {'lmaxqe':self.lmax_qe, 'lmax_qlm':self.lmax_qlm}
        for k in self.cls_weight.keys():
            ret['clsweight ' + k] = ut.clhash(self.cls_weight[k])
        for k in self.cls_cmb.keys():
            ret['clscmb ' + k] = ut.clhash(self.cls_cmb[k])
        for k in self.fal.keys():
            ret['fal' + k] = ut.clhash(self.fal[k])
        return ret

    def get_response(self, k, ksource, recache=False):
        """
            Args:
                k: QE anisotropy key
                ksource: CMB anisotropy source key

            Returns:
                Response array

        """
        if '_bh_' in k: # bias-hardened estimator
            kQE, bhksource = k.split('_bh_')
            assert len(ksource) == 1, (kQE, ksource)
            wL = self.get_response(kQE, bhksource, recache=recache)
            wL *= ut.cli(self.get_response(bhksource + kQE[1:], bhksource, recache=recache))
            ret = self.get_response(kQE, ksource, recache=recache)
            ret -= wL * self.get_response(bhksource + kQE[1:], ksource, recache=recache)
            return ret
        s, GorC, sins, ksp = qe_spin_data(k)
        assert s >= 0, s
        if s == 0: assert GorC == 'G', (s, GorC)
        fn = 'qe_' + ksp + k[1:] + '_source_%s_'%ksource + GorC + GorC
        if self.npdb.get(fn) is None or recache:
            GG, CC, GC, CG = get_response(k, self.lmax_qe, ksource, self.cls_weight, self.cls_cmb, self.fal,
                                lmax_qlm=self.lmax_qlm, transf=self.transf)
            if np.any(CG) or np.any(GC):
                print("Warning: C-G or G-C responses non-zero but not returned")
                # This may happen only if EB and/or TB are relevant and/or strange estimator mix.

            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_GG')
                if s > 0:
                    self.npdb.remove('qe_'+ ksp  + k[1:] + '_source_%s' % ksource + '_CC')
            self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_GG', GG)
            if s > 0:
                self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_CC', CC)
        return self.npdb.get(fn)


def get_response(qe_key, lmax_ivf, source, cls_weight, cls_cmb, fal, fal_leg2=None, lmax_ivf2=None, lmax_qlm=None, transf=None):
    r"""QE response calculation

        Args:
            qe_key: Quadratic estimator key (see this module docstring for descriptions)
            lmax_ivf: max. CMB multipole used in the QE
            source: anisotropy source key
            cls_weight(dict): fiducial spectra entering the QE weights (numerator in Eq. 2 of https://arxiv.org/abs/1807.06210)
            cls_cmb(dict): CMB spectra entering the CMB response (in principle lensed spectra, or grad-lensed spectra)
            fal(dict): (isotropic approximation to the) filtering cls. e.g. fal['tt'] :math:`= \frac {1} {C^{\rm TT}_\ell  +  N^{\rm TT}_\ell / b^2_\ell}` for temperature if filtered independently from polarization.
            fal_leg2(dict): Same as *fal* but for the second leg, if different.
            lmax_ivf2(optional): max. CMB multipole used in the QE on the second leg (if different to lmax_ivf)
            lmax_qlm(optional): responses are calculated up to this multipole. Defaults to lmax_ivf + lmax_ivf2

        Note:
            The result is *not* symmetrized with respect to the 'fals', if not the same on the two legs.
            In this case you probably want to run this twice swapping the fals in the second run.

    """
    if lmax_ivf2 is None: lmax_ivf2 = lmax_ivf
    if lmax_qlm is None : lmax_qlm = lmax_ivf + lmax_ivf2
    if '_bh_' in qe_key: # Bias-hardened estimators:
        k, hsource = qe_key.split('_bh_')# kQE hardened against hsource
        assert len(hsource) == 1, hsource
        h = hsource[0]
        RGG_ks, RCC_ks, RGC_ks, RCG_ks = get_response(k, lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm, transf=transf)
        RGG_hs, RCC_hs, RGC_hs, RCG_hs = get_response(h + k[1:], lmax_ivf, source, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm, transf=transf)
        RGG_kh, RCC_kh, RGC_kh, RCG_kh = get_response(k, lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm, transf=transf)
        RGG_hh, RCC_hh, RGC_hh, RCG_hh = get_response(h + k[1:], lmax_ivf, h, cls_weight, cls_cmb, fal,
                                                    fal_leg2=fal_leg2, lmax_ivf2=lmax_ivf2, lmax_qlm=lmax_qlm, transf=transf)
        RGG = RGG_ks - (RGG_kh * RGG_hs  * ut.cli(RGG_hh) + RGC_kh * RCG_hs  * ut.cli(RCC_hh))
        RCC = RCC_ks - (RCG_kh * RGC_hs  * ut.cli(RGG_hh) + RCC_kh * RCC_hs  * ut.cli(RCC_hh))
        RGC = RGC_ks - (RGG_kh * RGC_hs  * ut.cli(RGG_hh) + RGC_kh * RCC_hs  * ut.cli(RCC_hh))
        RCG = RCG_ks - (RCG_kh * RGG_hs  * ut.cli(RGG_hh) + RCC_kh * RCG_hs  * ut.cli(RCC_hh))
        return RGG, RCC, RGC, RCG

    qes = get_qes(qe_key, lmax_ivf, cls_weight, lmax2=lmax_ivf2, transf=transf)
    customR =  _get_response_custom(qe_key, qes, source, fal, lmax_qlm, fal_leg2=fal_leg2, transf=transf)
    if customR is None:
        return _get_response(qes, source, cls_cmb, fal, lmax_qlm, fal_leg2=fal_leg2)
    return customR


def _get_response_custom(qe_key, qes, source, fal_leg1, lmax_qlm, fal_leg2=None, transf=None):
    """Customized response code for selected keys """
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    if 'tt' in qe_key and source in ['n', 'ntt']:
        assert transf is not None
        # mask source keys does not fit under original parametrization scheme of plancklens
        # here source has spin 0 and qe can have any spin
        RGG = np.zeros(lmax_qlm + 1, dtype=float)
        RCC = np.zeros(lmax_qlm + 1, dtype=float)
        RGC = np.zeros(lmax_qlm + 1, dtype=float)
        RCG = np.zeros(lmax_qlm + 1, dtype=float)
        Ls = np.arange(lmax_qlm + 1, dtype=int)
        transfi = _clinv(transf)
        for qe in qes:
            si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
            so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
            s_qe  = abs(so + to)
            s_source = 0
            assert (si, ti) == (0, 0)
            s2, t2 = (0, 0) # Temperature only noise maps
            FA = uspin.get_spin_matrix(si, s2, fal_leg1)
            FB = uspin.get_spin_matrix(ti, t2, fal_leg2)
            if np.any(FB) and np.any(FB):
                # qe_spin positive:
                clA = ut.joincls([qe.leg_a.cl, FA, transfi ])
                clB = ut.joincls([qe.leg_b.cl, FB, transfi ])
                Rpr_st = uspin.wignerc(clA, clB, so, s2, to, t2, lmax_out=lmax_qlm)

                # qe_spin negative
                if s_qe > 0:
                    fac = (-1) ** (so + si + to + ti)
                    FA = uspin.get_spin_matrix(-si, s2, fal_leg1)
                    FB = uspin.get_spin_matrix(-ti, t2, fal_leg2)
                    clA = ut.joincls([qe.leg_a.cl.conj(), FA,  transfi])
                    clB = ut.joincls([qe.leg_b.cl.conj(), FB,  transfi])
                    Rmr_st = fac * uspin.wignerc(clA, clB, -so, s2, -to, t2, lmax_out=lmax_qlm)
                else:
                    Rmr_st = Rpr_st
                prefac = 0.5 * qe.cL(Ls)
                RGG += prefac * ( Rpr_st.real + Rmr_st.real * (-1) ** s_qe)
                RCC += prefac * ( Rpr_st.real - Rmr_st.real * (-1) ** s_qe)
                RGC += prefac * (-Rpr_st.imag + Rmr_st.imag * (-1) ** s_qe)
                RCG += prefac * ( Rpr_st.imag + Rmr_st.imag * (-1) ** s_qe)

        return RGG, RCC, RGC, RCG
    else:
        return None


def get_dresponse_dlncl(qe_key, l, cl_key, lmax_ivf, source, cls_weight, cls_cmb, fal_leg1,
                        fal_leg2=None, lmax_ivf2=None, lmax_out=None):
    """QE isotropic response derivative function dR_L / dlnC_l.

    """
    if lmax_ivf2 is None: lmax_ivf2 = lmax_ivf
    if lmax_out is None : lmax_out = lmax_ivf2 + lmax_ivf
    dcls_cmb = {k: np.zeros_like(cls_cmb[k]) for k in cls_cmb.keys()}
    dcls_cmb[cl_key][l] = cls_cmb[cl_key][l]
    qes = get_qes(qe_key, lmax_ivf, cls_weight, lmax2=lmax_ivf2)
    return _get_response(qes, source, dcls_cmb, fal_leg1,lmax_out, fal_leg2=fal_leg2)

def _get_response(qes, source, cls_cmb, fal_leg1, lmax_qlm, fal_leg2=None):
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    RGG = np.zeros(lmax_qlm + 1, dtype=float)
    RCC = np.zeros(lmax_qlm + 1, dtype=float)
    RGC = np.zeros(lmax_qlm + 1, dtype=float)
    RCG = np.zeros(lmax_qlm + 1, dtype=float)
    Ls = np.arange(lmax_qlm + 1, dtype=int)
    for qe in qes:
        si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
        so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
        for s2 in ([0, -2, 2]):
            FA = uspin.get_spin_matrix(si, s2, fal_leg1)
            if np.any(FA):
                for t2 in ([0, -2, 2]):
                    FB = uspin.get_spin_matrix(ti, t2, fal_leg2)
                    if np.any(FB):
                        rW_st, prW_st, mrW_st, s_cL_st = get_covresp(source, -s2, t2, cls_cmb, len(FB) - 1)
                        clA = ut.joincls([qe.leg_a.cl, FA])
                        clB = ut.joincls([qe.leg_b.cl, FB, mrW_st.conj()])
                        Rpr_st = uspin.wignerc(clA, clB, so, s2, to, -s2 + rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                        rW_ts, prW_ts, mrW_ts, s_cL_ts = get_covresp(source, -t2, s2, cls_cmb, len(FA) - 1)
                        clA = ut.joincls([qe.leg_a.cl, FA, mrW_ts.conj()])
                        clB = ut.joincls([qe.leg_b.cl, FB])
                        Rpr_st = Rpr_st + uspin.wignerc(clA, clB, so, -t2 + rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        assert rW_st == rW_ts and rW_st >= 0, (rW_st, rW_ts)
                        if rW_st > 0:
                            clA = ut.joincls([qe.leg_a.cl, FA])
                            clB = ut.joincls([qe.leg_b.cl, FB, prW_st.conj()])
                            Rmr_st = uspin.wignerc(clA, clB, so, s2, to, -s2 - rW_st, lmax_out=lmax_qlm) * s_cL_st(Ls)

                            clA = ut.joincls([qe.leg_a.cl, FA, prW_ts.conj()])
                            clB = ut.joincls([qe.leg_b.cl, FB])
                            Rmr_st = Rmr_st + uspin.wignerc(clA, clB, so, -t2 - rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts(Ls)
                        else:
                            Rmr_st = Rpr_st
                        prefac = qe.cL(Ls)
                        RGG += prefac * ( Rpr_st.real + Rmr_st.real * (-1) ** rW_st)
                        RCC += prefac * ( Rpr_st.real - Rmr_st.real * (-1) ** rW_st)
                        RGC += prefac * (-Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)
                        RCG += prefac * ( Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)

    return RGG, RCC, RGC, RCG


def get_mf_resp(qe_key, cls_cmb, cls_ivfs, lmax_qe, lmax_out, retterms=False):
    """Deflection-induced mean-field response calculation.

    See Carron & Lewis 2019 in prep.
    """
    # This version looks stable enough
    assert qe_key in ['p_p', 'ptt'], qe_key

    GL = np.zeros(lmax_out + 1, dtype=float)
    CL = np.zeros(lmax_out + 1, dtype=float)
    if qe_key == 'ptt':
        lmax_cmb = len(cls_cmb['tt']) - 1
        spins = [0]
    elif qe_key == 'p_p':
        lmax_cmb = min(len(cls_cmb['ee']) - 1, len(cls_cmb['bb'] - 1))
        spins = [-2, 2]
    elif qe_key == 'p':
        lmax_cmb = min(len(cls_cmb['ee']) - 1, len(cls_cmb['bb']) - 1, len(cls_cmb['tt']) - 1, len(cls_cmb['te']) - 1)
        spins = [0, -2, 2]
    else:
        assert 0, qe_key + ' not implemented'
    assert lmax_qe <= lmax_cmb
    if qe_key == 'ptt':
        cl_cmbtoticmb = {'tt': cls_cmb['tt'][:lmax_qe + 1] ** 2 * cls_ivfs['tt'][:lmax_qe + 1]}
        cl_cmbtoti = {'tt': cls_cmb['tt'][:lmax_qe + 1] * cls_ivfs['tt'][:lmax_qe + 1]}
    elif qe_key == 'p_p':
        cl_cmbtoticmb = {'ee': cls_cmb['ee'][:lmax_qe + 1] ** 2 * cls_ivfs['ee'][:lmax_qe + 1],
                         'bb': cls_cmb['bb'][:lmax_qe + 1] ** 2 * cls_ivfs['bb'][:lmax_qe + 1]}
        cl_cmbtoti = {'ee': cls_cmb['ee'][:lmax_qe + 1] * cls_ivfs['ee'][:lmax_qe + 1],
                      'bb': cls_cmb['bb'][:lmax_qe + 1] * cls_ivfs['bb'][:lmax_qe + 1]}
    else:
        assert 0, 'not implemented'
    # Build remaining fisher term II:
    FisherGII = np.zeros(lmax_out + 1, dtype=float)
    FisherCII = np.zeros(lmax_out + 1, dtype=float)
    terms = {'GK':np.zeros(lmax_out + 1, dtype=float), 'GxiK':np.zeros(lmax_out + 1, dtype=float)}
    for s1 in spins: # (xi K xi - xi) ) (K) like terms
        for s2 in spins:
            cl1 = uspin.spin_cls(s1, s2, cls_ivfs)[:lmax_qe + 1] * (0.5 ** (s1 != 0) * 0.5 ** (s2 != 0))
            # These 1/2 factor from the factor 1/2 in each B of B Covi B^dagger, where B maps spin-fields to T E B.
            cl2 = np.copy(uspin.spin_cls(s2, s1, cls_cmb)[:lmax_cmb + 1])
            cl2[:lmax_qe + 1] -= uspin.spin_cls(s2, s1, cl_cmbtoticmb)[:lmax_qe + 1] # must subtract here other unstable
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = uspin.get_spin_lower(s2, lmax_cmb) if a == - 1 else uspin.get_spin_raise(s2, lmax_cmb)
                    for b in [1]: # a, b symmetry
                        aj = uspin.get_spin_lower(-s1, lmax_cmb) if b == 1 else uspin.get_spin_raise(-s1, lmax_cmb)
                        hL = 2 * (-1) ** (s1 + s2) * uspin.wignerc(cl1, cl2 * ai * aj, s2, s1, -s2 - a, -s1 - b, lmax_out=lmax_out)
                        GL += (- a * b) * hL
                        CL += (-1) * hL

    # Build remaining Fisher term II:
    for s1 in spins: # (xi K) (xi K) like terms
        for s2 in spins:
            cl1 = uspin.spin_cls(s2, s1, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s1 != 0))
            cl2 = uspin.spin_cls(s1, s2, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s2 != 0))
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = uspin.get_spin_lower(s2, lmax_qe) if a == -1 else uspin.get_spin_raise(s2, lmax_qe)
                    for b in [1]:
                        aj = uspin.get_spin_lower(s1, lmax_qe) if b == 1 else uspin.get_spin_raise(s1, lmax_qe)
                        hL = 2 * (-1) ** (s1 + s2) * uspin.wignerc(cl1 * ai, cl2 * aj, -s2 - a, -s1, s2, s1 - b, lmax_out=lmax_out)
                        FisherGII += (- a * b) * hL
                        FisherCII += (-1) * hL

    terms['GK'] += GL
    terms['GxiK'] -= FisherGII
    GL -= FisherGII
    CL -= FisherCII
    terms['Gcons'] = -np.ones_like(GL) * CL[1]
    print("CL[1] ",CL[1])
    print("GL[1] (before subtraction) ", GL[1])
    print("GL[1] (after subtraction) ", GL[1] - CL[1])

    GL -= CL[1]
    CL -= CL[1]
    GL *= 0.25 * np.arange(lmax_out + 1) * np.arange(1, lmax_out + 2)
    CL *= 0.25 * np.arange(lmax_out + 1) * np.arange(1, lmax_out + 2)
    for term in terms.values():
        term *= 0.25 * np.arange(lmax_out + 1) * np.arange(1, lmax_out + 2)
    return (GL, CL) if not retterms else (GL, CL, terms)


def w3j_000_j(L:int, j:int, kmax:int, kmin:int=0, lng=None):
    """Squared spin-0 Wigner 3j symbols for zero magnetic moments (integral of Legendre poly)

            :math:`\begin{pmatrix}l_1 & l_2 & L \\ 0 & 0 & 0 \end{pmatrix}^2`

        Args:
            L: multipole
            j: the Wigners are calculated for all entries along the diagonal :math:`l_1 - l_2 = L - j`
               (others are zero)

        Returns:
            array of size kmax + 1 with k'th entry corresponding to :math:`l_1 = k + j, l_2 = L + k - j`

        Note:
            One could feed in l1max, l2max

    """
    assert 0 <= j <= L, (j, L)  # invalid input (or could return zeros)
    if lng is None:
        from scipy.special import gammaln
        lng = gammaln(np.arange(1, 2 * L + 2 * kmax + 3, dtype=float))  # FIXME: cumul. sum of ln n
    k = np.arange(kmin, kmax + 1, dtype=int)
    g = L + k
    w = lng[2 * (L - j)] + lng[2 * j] + lng[2 * k] - lng[2 * g + 1] + 2 * (lng[g] - lng[L - j] - lng[j] - lng[k])
    return np.exp(w)  # FIXME: need the exp ?


lmax2kmax = lambda L, lmax: lmax + L // 2 + L % 2


class spin0_resp:
    def __init__(self, ftl1, ftl2, cl_w:np.ndarray, cl_r:np.ndarray):
        """

            Note:
                We try and follow PL2015 conventions

                Def. of QE weights in Qu et al 2022 (g) differs by a factor 1/2 from PL2015 (W).


        """

        nz1, nz2 = np.nonzero(ftl1)[0], np.nonzero(ftl2)[0]
        l1min, l1max = nz1[0], nz1[-1]
        l2min, l2max = nz2[0], nz2[-1]
        Lmax = l1max + l2max

        self.l2p1_ftl1 = ftl1 *  (2 * np.arange(len(ftl1)) + 1)
        self.l2p1_ftl2 = ftl2 *  (2 * np.arange(len(ftl2)) + 1)

        self.ftl1 = ftl1
        self.ftl2 = ftl2

        self.lmins = [min(l1min, l2min), l1min, l2min]
        self.lmaxs = [max(l1max, l2max), l1max, l2max]

        self.Lmax = Lmax
        self.cl_w = cl_w # spectrum used in QE weights
        self.cl_r = cl_r # CMB sky response cl

        # dln Cl / dln l (only used for shear estimator)
        ls = np.arange(1, self.lmaxs[0] + 3)
        dlnCldlnl = ls[:-1] * np.diff(np.log(cl_w[ls])) # This might give trouble
        self.dlncldlnl = dlnCldlnl

        self.w2s = np.arange(Lmax + 1) * np.arange(1, Lmax + 2, dtype=float)
        self.lng = gammaln(np.arange(1, 2 * Lmax + 3, dtype=float)) # ln n! starting from 0

    def __call__(self, L, qe_key, source_key):
        return self._eval_resp(L, qe_key, source_key)


    def _fl1l2L_ptt(self, L:int, j:int, kmin:int, kmax:int, cl=None):
        """Lensing gradient response kernel in j, k parametrization, exclusive of wigner3j and root 2l+1 factors.

            Args:
                L: multipole
                j: the response if obtained for all pairs of multipoles such that :math:`l_1 - l_2 = L - j`
                   (others are zero by the triangle conditions)

            Returns:
                array of size kmax - kmin + 1 with k'th entry corresponding to :math:`l_1 = kmin + k + j, l_2 = L + kmin + k + - j`

            Note: allowed by triangle conditions are 0 <= j <= L and 0 <= k <= infty

        """
        if cl is None:
            cl = self.cl_r
        l1 = slice(kmin + j, kmax + j + 1)
        l2 = slice(kmin + L - j, kmax + L - j + 1)
        ret  = (self.w2s[L] + self.w2s[l2] - self.w2s[l1]) * cl[l2]
        ret += (self.w2s[L] + self.w2s[l1] - self.w2s[l2]) * cl[l1]
        return 0.5 * ret

    def _qel1l2L_ptt(self, L:int, j:int, kmin:int, kmax:int):
        return self._fl1l2L_ptt(L, j, kmin, kmax, cl=self.cl_w)

    def _qel1l2L_gtt(self, L:int, j:int, kmin:int, kmax:int, scal1=None, scal2=None):
        """Shear estimator from Qu et al 2022 (for which the pos. space implementation seems unstable at low-L)

            Note:
                the weights are defined here acting on filtered maps, unlike that paper
                we identify the inverse filter with the data spectrum

        """
        if L < 2:
            return 0. # Shear spin 2
        l1 = slice(kmin + j, kmax + j + 1)
        l2 = slice(kmin + L - j, kmax + L - j + 1)
        w2dif = self.w2s[L] + self.w2s[l1] - self.w2s[l2]
        cos2tht = w2dif * (w2dif - 2) / (2 * self.w2s[L] * self.w2s[l1]) - 1
        ret1 = self.w2s[L] * self.dlncldlnl[l1] * self.cl_w[l1] * self.ftl1[l1] * cos2tht * _clinv(self.ftl2[l2])

        # symmetrization
        w2dif = self.w2s[L] + self.w2s[l2] - self.w2s[l1]
        cos2tht = w2dif * (w2dif - 2) / (2 * self.w2s[L] * self.w2s[l2]) - 1
        ret2 = self.w2s[L] * self.dlncldlnl[l2] * self.cl_w[l2] * self.ftl1[l2] * cos2tht * _clinv(self.ftl2[l1])
        return  0.5 * (ret1 + ret2)

    def _qel1l2L_gtt2(self, L:int, j:int, kmin:int, kmax:int):
        """Shear estimator from Qu et al 2022 (for which the pos. space implementation seems unstable at low-L)

            Note:
                the weights are defined here acting on filtered maps, unlike that paper
                we identify the inverse filter with the data spectrum

        """
        if L < 2:
            return 0. # Shear spin 2
        l1 = slice(kmin + j, kmax + j + 1)
        l2 = slice(kmin + L - j, kmax + L - j + 1)
        w2dif = self.w2s[L] + self.w2s[l1] - self.w2s[l2]
        cos2tht = w2dif * (w2dif - 2) / (2 * self.w2s[L] * self.w2s[l1]) - 1
        ret1 = self.w2s[L] * self.dlncldlnl[l1] * self.cl_w[l1]  * cos2tht

        # symmetrization
        w2dif = self.w2s[L] + self.w2s[l2] - self.w2s[l1]
        cos2tht = w2dif * (w2dif - 2) / (2 * self.w2s[L] * self.w2s[l2]) - 1
        ret2 = self.w2s[L] * self.dlncldlnl[l2] * self.cl_w[l2]  * cos2tht
        return  0.5 * (ret1 + ret2)

    def _qel1l2L_stt(self, L:int, j:int, kmin:int, kmax:int):
        """Point source anisotropy weights"""
        return self._fl1l2L_stt(L, j, kmin, kmax)

    @staticmethod
    def _fl1l2L_stt(L: int, j: int, kmin: int, kmax: int):
        """Point source anisotropy weights"""
        return np.full(kmax - kmin + 1, -1., dtype=float)

    def _3legP(self, L:int, j:int, kmin:int, kmax:int):
        """Squared wigner 3j for multipoles L, l1 and l2, and zero magnetic moments
            (twice the integral of 3 Legendre polynomials)

            Arguments are such that  l1 = k + j and l2 = L + k - j

            Note: allowed by triangle conditions are 0 <= j <= L and 0 <= k <= infty

            TODO: improve this

        """
        k = np.arange(kmin, kmax + 1, dtype=int)
        g = L + k
        w  = self.lng[2 * (L - j)] + self.lng[2 * j] + self.lng[2 * k] - self.lng[2 * g + 1]
        w += 2 * (self.lng[g] - self.lng[L - j] - self.lng[j] - self.lng[k])
        return np.exp(w)

    def _eval_resp(self, L, qe_key, source_key):
        if L > self.Lmax:
            return 0.
        resp_weights = getattr(self, '_fl1l2L_' + source_key)
        qe12_weights = getattr(self, '_qel1l2L_' + qe_key)
        R = 0.
        for j in range(L + 1):  # could in principle speed that up since only half the loop is needed owing to sym
            kmax = min(self.lmaxs[1] - j, self.lmaxs[2] + j - L)  # l1 = k + j, l2 = L + 2k - l2 = L + k - j
            kmin = max(max(self.lmins[1] - j, self.lmins[2] + j - L), 0)
            if kmax >= kmin:
                l1 = slice(kmin + j, kmax + j + 1)
                l2 = slice(kmin + L - j, kmax + L - j + 1)
                sky_w = resp_weights(L, j, kmin, kmax) # weights exluding of wigner 3j and 2l + 1 / 4pi factors
                if qe12_weights is not resp_weights:
                    qe_w = qe12_weights(L, j, kmin, kmax) # weights exluding of wigner 3j and 2l + 1 / 4pi factors
                else:
                    qe_w = sky_w
                wig = self._3legP(L, j, kmin, kmax)  # wig3j with zero magnetic number squared
                #: 1/2 of the exec time
                R += np.sum(self.l2p1_ftl1[l1] * self.l2p1_ftl2[l2] * sky_w * qe_w * wig)
        return R / (8 * np.pi)  # 1/8 because 1/2 / (4 pi) (See PL2015 app A)

    def _eval_nhl(self, L, qe1_key, qe2_key):
        """Gaussian noise bias, Eq A30 PL2015.

            We use here l1 + l2 + L always even (wigners with zero magnetic moments) and only symmetric QEs

        """
        if L > self.Lmax:
            return 0.
        qe12_weights = getattr(self, '_qel1l2L_' + qe1_key)
        qe34_weights = getattr(self, '_qel1l2L_' + qe2_key)
        R = 0.
        for j in range(L + 1):  # could in principle speed that up since only half the loop is needed owing to sym
            kmax = min(self.lmaxs[1] - j, self.lmaxs[2] + j - L)  # l1 = k + j, l2 = L + 2k - l2 = L + k - j
            kmin = max(max(self.lmins[1] - j, self.lmins[2] + j - L), 0)
            if kmax >= kmin:
                l1 = slice(kmin + j, kmax + j + 1)
                l2 = slice(kmin + L - j, kmax + L - j + 1)
                qe1_w = qe12_weights(L, j, kmin, kmax) # weights exluding of wigner 3j and 2l + 1 / 4pi factors
                if qe12_weights is not qe34_weights:
                    qe2_w = qe34_weights(L, j, kmin, kmax) # weights exluding of wigner 3j and 2l + 1 / 4pi factors
                else:
                    qe2_w = qe1_w
                wig = self._3legP(L, j, kmin, kmax)  # wig3j with zero magnetic number squared
                #: 1/2 of the exec time
                R += np.sum(self.l2p1_ftl1[l1] * self.l2p1_ftl2[l2] * qe1_w * qe2_w * wig)
                # ftl is spectrum of filtered data for fid matching data
        return R / (8 * np.pi)
