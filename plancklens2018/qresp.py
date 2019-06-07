"""Module for QE response calculations.

FIXME: spin-0 QE sign conventions (stt, ftt, ...)
"""

from __future__ import absolute_import
from __future__ import print_function

import healpy as hp
import os
import numpy as np
import pickle as pk

from plancklens2018 import sql
from plancklens2018 import utils
from plancklens2018 import utils_spin as uspin
from plancklens2018.utils import clhash, hash_check, joincls
from plancklens2018 import mpi

class qeleg:
    def __init__(self, spin_in, spin_out, cl):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def __eq__(self, leg):
        if self.spin_in != leg.spin_in or self.spin_ou != leg.spin_ou or self.get_lmax() != self.get_lmax():
            return False
        return np.all(self.cl == leg.cl)

    def get_lmax(self):
        return len(self.cl) - 1


class qeleg_multi:
    def __init__(self, spins_in, spin_out, cls):
        assert isinstance(spins_in, list) and isinstance(cls, list)
        assert len(spins_in) == len(cls)
        self.spins_in = spins_in
        self.cls = cls
        self.spin_ou = spin_out

    def __iadd__(self, qeleg):
        """Adds one spin_in/cl tuple.

        """
        assert qeleg.spin_ou == self.spin_ou, (qeleg.spin_ou, self.spin_ou)
        self.spins_in.append(qeleg.spin_in)
        self.cls.append(np.copy(qeleg.cl))
        return self

    def __call__(self, get_alm, nside):
        """Returns the spin-weighted real-space map of the estimator.

        We first build X_lm in the wanted _{si}X_lm _{so}Y_lm and then convert this alm2map_spin conventions.
        """
        lmax = self.get_lmax()
        glm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
        clm = np.zeros(hp.Alm.getsize(lmax), dtype=complex) # X_{lm} is here glm + i clm
        for i, (si, cl) in enumerate(zip(self.spins_in, self.cls)):
            assert si in [0, -2, 2], str(si) + ' input spin not implemented'
            gclm = [get_alm('e'), get_alm('b')] if abs(si) == 2 else [-get_alm('t'), 0.]
            assert len(gclm) == 2
            sgn_g = -(-1) ** si if si < 0 else -1
            sgn_c = (-1) ** si if si < 0 else -1
            glm += hp.almxfl(utils.alm_copy(gclm[0], lmax), sgn_g * cl)
            if np.any(gclm[1]):
                clm += hp.almxfl(utils.alm_copy(gclm[1], lmax), sgn_c * cl)
        glm *= -1
        if self.spin_ou > 0: clm *= -1
        Red, Imd = uspin.alm2map_spin((glm, clm), nside, abs(self.spin_ou), lmax)
        if self.spin_ou < 0 and self.spin_ou % 2 == 1: Red *= -1
        if self.spin_ou < 0 and self.spin_ou % 2 == 0: Imd *= -1
        return Red + 1j * Imd


    def get_lmax(self):
        return np.max([len(cl) for cl in self.cls]) - 1

class qe:
    def __init__(self, leg_a, leg_b, cL):
        assert leg_a.spin_ou +  leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def get_lmax_a(self):
        return self.leg_a.get_lmax()

    def get_lmax_b(self):
        return self.leg_b.get_lmax()

    def get_lmax_qlm(self):
        return len(self.cL)

def compress_qe(qes, verbose=True):
    skip = []
    qes_compressed = []
    for i, qe in enumerate(qes):
        if i not in skip:
            lega = qe.leg_a
            lega_m=  qeleg_multi([qe.leg_a.spin_in], qe.leg_a.spin_ou, [qe.leg_a.cl])
            legb_m = qeleg_multi([qe.leg_b.spin_in], qe.leg_b.spin_ou, [qe.leg_b.cl])
            for j, qej in enumerate(qes[i + 1:]):
                if qej.leg_a == lega and legb_m.spin_ou == qej.leg_b.spin_ou:
                    legb_m += qej.leg_b
                    skip.append(i + 1 + j)
            qes_compressed.append( (lega_m, legb_m, qe.cL))
    if len(skip) > 0 and verbose:
        print("%s alm2map_spin transforms now required, down from %s"%(2 * (len(qes) -len(skip)) , 2 * len(qes)) )

    return qes_compressed


def eval_qe(qes_list, nside, get_alm, verbose=True):
    qes = compress_qe(qes_list, verbose=verbose) # triple lega (leg_multi) legb, clout
    qe_spin = qes[0][0].spin_ou + qes[0][1].spin_ou
    cL_out = qes[0][-1]
    assert qe_spin >= 0, qe_spin
    for qe in qes[1:]:
        assert np.all(qe[-1] == cL_out)
        assert qe[0].spin_ou + qe[1].spin_ou == qe_spin
    d = np.zeros(hp.nside2npix(nside), dtype=complex)
    for i, qe in enumerate(qes):
        if verbose:
            print("QE %s out of %s :"%(i + 1, len(qes)))
            print("in-spins 1st leg" ,qe[0].spins_in, qe[0].spin_ou)
            print("in-spins 2nd leg", qe[1].spins_in, qe[1].spin_ou)
        d += qe[0](get_alm, nside) * qe[1](get_alm, nside)
    glm, clm = uspin.map2alm_spin((d.real, d.imag),qe_spin, lmax=len(cL_out) - 1)
    hp.almxfl(glm, cL_out, inplace=True)
    if np.any(clm):
        hp.almxfl(clm, cL_out, inplace=True)
    return glm, clm


def get_resp_legs(source, lmax):
    """Defines the responses terms for a CMB map anisotropy source.

    Args:
        source (str): anisotropy source (e.g. 'p', 'f', ...).
        lmax (int): responses are given up to lmax.

    Returns:
        4-tuple (r, rR, -rR, cL):  source spin response *r* (positive or zero),
        the harmonic responses for +r and -r (2 1d-arrays), and the scaling between the G/C modes
        and the potentials of interest (for lensing, \phi_{LM}, \Omega_{LM} = \sqrt{L (L + 1)} G_{LM}, C_{LM}).

    """
    lmax_cL = 2 *  lmax
    if source == 'p': # lensing (gradient and curl): _sX -> _sX -  1/2 alpha_1 \eth _sX - 1/2 \alpha_{-1} \bar \eth _sX
        return {s : (1, -0.5 * uspin.get_spin_lower(s, lmax), -0.5 * uspin.get_spin_raise(s, lmax),
                     uspin.get_spin_raise(0, lmax_cL)) for s in [0, -2, 2]}
    if source == 'f': # Modulation: _sX -> _sX + f _sX.
        return {s : (0, 0.5 * np.ones(lmax + 1, dtype=float), 0.5 * np.ones(lmax + 1, dtype=float),
                        np.ones(lmax_cL + 1, dtype=float)) for s in [0, -2, 2]}
    if source in ['a', 'a_p']: # Polarisation rotation _\pm 2 X ->  _\pm 2 X + \mp 2 i a _\pm 2 X
        ret = {s: (0,  -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                       -np.sign(s) * 1j * np.ones(lmax + 1, dtype=float),
                        np.ones(lmax_cL + 1, dtype=float)) for s in [-2, 2]}
        ret[0]=(0, np.zeros(lmax + 1, dtype=float),np.zeros(lmax + 1, dtype=float),np.ones(lmax_cL + 1, dtype=float))
        return ret

    assert 0, source + ' response legs not implemented'

def get_covresp(source, s1, s2, cls, lmax):
    """Defines the responses terms for a CMB covariance anisotropy source.

        \delta < s_d(n) _td^*(n')> \equiv
        _r\alpha(n) W^{r, st}_l _{s - r}Y_{lm}(n) _tY^*_{lm}(n') +
        _r\alpha^*(n') W^{r, ts}_l _{s}Y_{lm}(n) _{t-r}Y^*_{lm}(n')

    """
    if source in ['p','x', 'f', 'a', 'a_p']:
        # Lensing, modulation, or pol. rotation field from the field representation
        s_source, prR, mrR, cL_scal = get_resp_legs(source, lmax)[s1]
        coupl = uspin.get_spin_coupling(s1, s2, cls)[:lmax + 1]
        return s_source, prR * coupl, mrR * coupl, cL_scal
    elif source in ['stt', 's']:
        # Point source 'S^2': Cov -> Cov + B delta_nn' S^2(n) B^\dagger on the diagonal.
        # From the def. there are actually 4 identical W terms hence a factor 1/4.
        cond = s1 == 0 and s2 == 0
        s_source = 0
        prR = 0.25 * np.ones(lmax + 1, dtype=float) * cond
        mrR = 0.25 * np.ones(lmax + 1, dtype=float) * cond
        cL_scal = np.ones(2 * lmax + 1, dtype=float) * cond
        return s_source, prR, mrR, cL_scal
    else:
        assert 0, 'source ' + source + ' cov. response not implemented'


def get_qes(qe_key, lmax, cls_weight):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_weight (dict): CMB spectra entering the weights

    #FIXME:
        * lmax_A, lmax_B, lmaxout!

    The weights are defined by their action on the inverse-variance filtered $ _{s}\\bar X_{lm}$.

    """
    if qe_key[0] in ['p', 'x', 'a', 'f', 's']:
        if qe_key in ['ptt', 'xtt', 'att', 'ftt', 'stt']:
            s_lefts= [0]
        elif qe_key in ['p_p', 'x_p', 'a_p', 'f_p']:
            s_lefts= [-2, 2]
        elif qe_key in ['p', 'x', 'a', 'f']:
            s_lefts = [0, -2, 2]
        else:
            assert 0, qe_key + ' not implemented'
        qes = []
        s_rights_in = s_lefts
        for s_left in s_lefts:
            for sin in s_rights_in:
                sout = -s_left
                s_qe, irr1, cl_sosi, cL_out =  get_covresp(qe_key[0], sout, sin, cls_weight, lmax)
                if np.any(cl_sosi):
                    lega = qeleg(s_left, s_left, 0.5 *(1. + (s_left == 0)) * np.ones(lmax + 1, dtype=float))
                    legb = qeleg(sin, sout + s_qe, 0.5 * (1. + (sin == 0)) * 2 * cl_sosi)
                    qes.append(qe(lega, legb, cL_out))
        return qes
    else:
        assert 0, qe_key + ' not implemented'


def qe_spin_data(qe_key):
    """Returns out and in spin-weights of quadratic estimator from its quadratic estimator key.

        Output is an integer >= 0 (spin), a letter 'G' or 'C' if gradient or curl mode estimator, the
        unordered list of unique spins (>= 0) input to the estimator, and the spin-1 qe key.

    """
    qes = get_qes(qe_key, 1, {k:np.array([1.]) for k in ['tt', 'te', 'ee', 'tb', 'eb', 'bb']})
    spins_out = [qe.leg_a.spin_ou + qe.leg_b.spin_ou for qe in qes]
    spins_in = np.unique(np.abs([qe.leg_a.spin_in for qe in qes] + [qe.leg_b.spin_in for qe in qes]))
    assert len(np.unique(spins_out)) == 1, spins_out
    assert spins_out[0] >= 0, spins_out[0]
    if spins_out[0] > 0: assert qe_key[0] in ['x', 'p'], 'non-zero spin anisotropy ' + qe_key +  ' not implemented ?'
    return spins_out[0], 'C' if qe_key[0] == 'x' else 'G', spins_in, 'p' if qe_key[0] == 'x' else qe_key[0]


class resp_lib_simple:
    def __init__(self, lib_dir, lmax_qe, cls_weight, cls_cmb, fal, lmax_qlm):
        self.lmax_qe = lmax_qe
        self.lmax_qlm = lmax_qlm
        self.cls_weight = cls_weight
        self.cls_cmb = cls_cmb
        self.fal = fal
        self.lib_dir = lib_dir

        fn_hash = os.path.join(lib_dir, 'resp_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())
        self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))

    def hashdict(self):
        ret = {'lmaxqe':self.lmax_qe, 'lmax_qlm':self.lmax_qlm}
        for k in self.cls_weight.keys():
            ret['clsweight ' + k] = clhash(self.cls_weight[k])
        for k in self.cls_cmb.keys():
            ret['clscmb ' + k] = clhash(self.cls_cmb[k])
        for k in self.fal.keys():
            ret['fal' + k] = clhash(self.fal[k])
        return ret

    def get_response(self, k, ksource, recache=False):
        s, GorC, sins, ksp = qe_spin_data(k)
        assert s >= 0, s
        if s == 0: assert GorC == 'G', (s, GorC)
        fn = 'qe_' + ksp + k[1:] + '_source_%s_'%ksource + GorC + GorC
        if self.npdb.get(fn) is None or recache:
            GG, CC, GC, CG = get_response(k, self.lmax_qe, ksource, self.cls_weight, self.cls_cmb, self.fal,
                                lmax_out=self.lmax_qlm)
            if np.any(CG) or np.any(GC):
                print("Warning: C-G or G-C responses non-zero but not returned")
                # This may happen only if EB and/or TB are relevant.

            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_GG')
                if s > 0:
                    self.npdb.remove('qe_'+ ksp  + k[1:] + '_source_%s' % ksource + '_CC')
            self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_GG', GG)
            if s > 0:
                self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_CC', CC)
        return self.npdb.get(fn)

def get_dresponse_dlncl(qe_key, l, cl_key, lmax_qe, source, cls_weight, cls_cmb, fal_leg1, fal_leg2=None, lmax_out=None):
    """QE isotropic response derivative function dR_L / dlnC_l.

    """
    dcls_cmb = {k: np.zeros_like(cls_cmb[k]) for k in cls_cmb.keys()}
    dcls_cmb[cl_key][l] = cls_cmb[cl_key][l]
    qes = get_qes(qe_key, lmax_qe, cls_weight)
    return _get_response(qes, lmax_qe, source, dcls_cmb, fal_leg1, fal_leg2=fal_leg2, lmax_out=lmax_out)


def get_response(qe_key, lmax_qe, source, cls_weight, cls_cmb, fal_leg1, fal_leg2=None, lmax_out=None):
    """QE isotropic response.

    #FIXME: explain fal here

    """
    qes = get_qes(qe_key, lmax_qe, cls_weight)
    return _get_response(qes, lmax_qe, source, cls_cmb, fal_leg1, fal_leg2=fal_leg2, lmax_out=lmax_out)

get_response_sepTP = get_response # Here only for historical reasons.


def _get_response(qes, lmax_qe, source,  cls_cmb, fal_leg1, fal_leg2=None, lmax_out=None):
    lmax_qlm = min(2 * lmax_qe,  2 * lmax_qe if lmax_out is None else lmax_out)
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    RGG = np.zeros(lmax_qlm + 1, dtype=float)
    RCC = np.zeros(lmax_qlm + 1, dtype=float)
    RGC = np.zeros(lmax_qlm + 1, dtype=float)
    RCG = np.zeros(lmax_qlm + 1, dtype=float)
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
                        clA = joincls([qe.leg_a.cl, FA])
                        clB = joincls([qe.leg_b.cl, FB, mrW_st.conj()])
                        Rpr_st = uspin.wignerc(clA, clB, so, s2, to, -s2 + rW_st, lmax_out=lmax_qlm) * s_cL_st[:lmax_qlm + 1]

                        rW_ts, prW_ts, mrW_ts, s_cL_ts = get_covresp(source, -t2, s2, cls_cmb, len(FA) - 1)
                        clA = joincls([qe.leg_a.cl, FA, mrW_ts.conj()])
                        clB = joincls([qe.leg_b.cl, FB])
                        Rpr_st = Rpr_st + uspin.wignerc(clA, clB, so, -t2 + rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts[:lmax_qlm + 1]
                        assert rW_st == rW_ts and rW_st >= 0, (rW_st, rW_ts)
                        if rW_st > 0:
                            clA = joincls([qe.leg_a.cl, FA])
                            clB = joincls([qe.leg_b.cl, FB, prW_st.conj()])
                            Rmr_st = uspin.wignerc(clA, clB, so, s2, to, -s2 - rW_st, lmax_out=lmax_qlm) * s_cL_st[:lmax_qlm + 1]

                            clA = joincls([qe.leg_a.cl, FA, prW_ts.conj()])
                            clB = joincls([qe.leg_b.cl, FB])
                            Rmr_st = Rmr_st + uspin.wignerc(clA, clB, so, -t2 - rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts[:lmax_qlm + 1]
                        else:
                            Rmr_st = Rpr_st
                        prefac = (-1) ** (so + to + rW_ts) * qe.cL[:lmax_qlm + 1]
                        RGG += prefac * ( Rpr_st.real + Rmr_st.real * (-1) ** rW_st)
                        RCC += prefac * ( Rpr_st.real - Rmr_st.real * (-1) ** rW_st)
                        RGC += prefac * (-Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)
                        RCG += prefac * ( Rpr_st.imag + Rmr_st.imag * (-1) ** rW_st)

    return RGG, RCC, RGC, RCG


def get_mf_resp(qe_key, cls_cmb, cls_ivfs, lmax_qe, lmax_out):
    """Deflection-induced mean-field response calculation.

    See Carron & Lewis 2019 in prep.
    """
    # This version looks stable enough
    assert qe_key in ['p_p', 'ptt'], qe_key
    assert not np.any(cls_cmb.get('tb', 0.)) and not np.any(cls_cmb.get('eb', 0.)), 'version with CMB EB or TB not implemented'
    assert not np.any(cls_ivfs.get('tb', 0.)) and not np.any(cls_ivfs.get('eb', 0.)), 'version with filt EB or TB not implemented'

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
        assert not np.any(cls_cmb['bb']), 'not implemented w. bb weights'
        cl_cmbtoticmb = {'ee': cls_cmb['ee'][:lmax_qe + 1] ** 2 * cls_ivfs['ee'][:lmax_qe + 1],
                         'bb': np.zeros(lmax_qe + 1, dtype=float)}
        cl_cmbtoti = {'ee': cls_cmb['ee'][:lmax_qe + 1] * cls_ivfs['ee'][:lmax_qe + 1],
                      'bb': np.zeros(lmax_qe + 1, dtype=float)}
    else:
        assert 0, 'not implemented'
    # Build remaining fisher term II:
    FisherGII = np.zeros(lmax_out + 1, dtype=float)
    FisherCII = np.zeros(lmax_out + 1, dtype=float)

    for s1 in spins:
        for s2 in spins:
            cl1 = uspin.get_spin_coupling(s1, s2, cls_ivfs)[:lmax_qe + 1] * (0.5 ** (s1 != 0) * 0.5 ** (s2 != 0))
            # These 1/2 factor from the factor 1/2 in each B of B Covi B^dagger, where B maps spin-fields to T E B.
            cl2 = uspin.get_spin_coupling(s2, s1, cls_cmb)[:lmax_cmb + 1]
            cl2[:lmax_qe + 1] -= uspin.get_spin_coupling(s2, s1, cl_cmbtoticmb)[:lmax_qe + 1]
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = uspin.get_spin_lower(s2, lmax_cmb) if a == - 1 else uspin.get_spin_raise(s2, lmax_cmb)
                    for b in [1]: # a, b symmetry
                        aj = uspin.get_spin_lower(-s1, lmax_cmb) if b == 1 else uspin.get_spin_raise(-s1, lmax_cmb)
                        hL = 2 * (-1) ** (s1 + s2) * uspin.wignerc(cl1, cl2 * ai * aj, s2, s1, -s2 - a, -s1 - b, lmax_out=lmax_out)
                        GL += (- a * b) * hL
                        CL += (-1) * hL

    # Build remaining Fisher term II:
    for s1 in spins:
        for s2 in spins:
            cl1 = uspin.get_spin_coupling(s2, s1, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s1 != 0))
            cl2 = uspin.get_spin_coupling(s1, s2, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s2 != 0))
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = uspin.get_spin_lower(s2, lmax_qe) if a == -1 else uspin.get_spin_raise(s2, lmax_qe)
                    for b in [1]:
                        aj = uspin.get_spin_lower(s1, lmax_qe) if b == 1 else uspin.get_spin_raise(s1, lmax_qe)
                        hL = 2 * (-1) ** (s1 + s2) * uspin.wignerc(cl1 * ai, cl2 * aj, -s2 - a, -s1, s2, s1 - b, lmax_out=lmax_out)
                        FisherGII += (- a * b) * hL
                        FisherCII += (-1) * hL
    GL -= FisherGII
    CL -= FisherCII
    print("CL[1] ",CL[1])
    print("GL[1] (before subtraction) ", GL[1])
    print("GL[1] (after subtraction) ", GL[1] - CL[1])

    GL -= CL[1]
    CL -= CL[1]
    GL *= 0.25 * np.arange(lmax_out + 1) * np.arange(1, lmax_out + 2)
    CL *= 0.25 * np.arange(lmax_out + 1) * np.arange(1, lmax_out + 2)
    return GL, CL
