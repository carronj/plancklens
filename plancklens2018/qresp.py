"""Module for QE response calculations.

FIXME: spin-0 QE sign conventions (stt, ftt, ...)
"""

from __future__ import absolute_import

import os
import numpy as np
import pickle as pk


from plancklens2018 import sql
from plancklens2018.utils import clhash, hash_check, joincls
from plancklens2018 import mpi

try:
    from plancklens2018.wigners import wigners  # fortran shared object
    HASWIGNER = True
except:
    print("wigners.so fortran shared object not found")
    print('try f2py -c -m wigners wigners.f90 from the command line in wigners directory')
    print("Falling back on python2 weave implementation")
    HASWIGNER = False
    from plancklens2018.wigners import gaujac, gauleg

verbose = False


class qeleg:
    def __init__(self, spin_in, spin_out, cl):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def get_lmax(self):
        return len(self.cl) - 1

class qe:
    def __init__(self, leg_a, leg_b, cL):
        assert leg_a.spin_ou +  leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def __call__(self, lega_dlm, legb_dlm, nside):
        pass
        # FIXME
    def get_lmax_a(self):
        return self.leg_a.get_lmax()

    def get_lmax_b(self):
        return self.leg_b.get_lmax()

    def get_lmax_qlm(self):
        return len(self.cL)


class resp_leg:
    """ Response instance of a spin-s field to a spin-r anisotropy source.

    Args:
        s (int): spin of the field which responds to the anisotropy  source.
        r (int): spin of the anisotropy source.
        RL (1d array): response coefficients.

    """
    def __init__(self, s, r, RL):
        assert s >= 0, 'do I want this ?'
        self.s = s
        self.r = r
        self.RL = RL

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
        return {s : (1, -0.5 * get_spin_lower(s, lmax),
                     -0.5 * get_spin_raise(s, lmax),
                     np.sqrt(np.arange(lmax_cL + 1) * np.arange(1, lmax_cL + 2, dtype=float))) for s in [0, -2, 2]}
    if source == 'f': # Modulation: _sX -> _sX + f _sX.
        return {s : (0, 0.5 * np.ones(lmax + 1, dtype=float),
                        0.5 * np.ones(lmax + 1, dtype=float),
                        np.ones(lmax_cL + 1, dtype=float)) for s in [0, -2, 2]}
    assert 0, source + ' response legs not implemented'

def get_covresp(source, s1, s2, cls, lmax):
    """Defines the responses terms for a CMB covariance anisotropy source.

        \delta < s_d(n) _td^*(n')> \equiv
        _r\alpha(n) W^{r, st}_l _{s - r}Y_{lm}(n) _tY^*_{lm}(n') +
        _r\alpha^*(n') W^{r, ts}_l _{s}Y_{lm}(n) _{t-r}Y^*_{lm}(n')

    """
    if source in ['p', 'f']:
        # Lensing or modulation field from the field representation
        s_source, prR, mrR, cL_scal = get_resp_legs(source, lmax)[s1]
        coupl = get_spin_coupling(s1, s2, cls)[:lmax + 1]
        return s_source, prR * coupl, mrR * coupl, cL_scal
    elif source == 'stt':
        # Point source 'S^2': Cov -> Cov + B delta_nn' S^2(n) B^\dagger on the diagonal.
        # From the def. there are actually 4 identical W terms hence a factor 1/4.
        cond = s1 == 0 and s2 == 0
        s_source = 0
        prR = 0.25 * np.ones(lmax + 1, dtype=float) * cond
        mrR = 0.25 * np.ones(lmax + 1, dtype=float) * cond
        cL_scal = np.ones(2 * lmax + 1, dtype=float) * cond
        return s_source, prR, mrR, cL_scal
    else:
        assert 0, 'source ' + source + ' not implemented'


def get_qes(qe_key, lmax, cls_weight):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_weight (dict): CMB spectra entering the weights

    #FIXME:
        * lmax_A, lmax_B, lmaxout!

    The weights are defined by their action on the inverse-variance filtered $ _{s}\\bar X_{lm}$.
    (It is useful to remember that by convention  $_{0}X_{lm} = - T_{lm}$)

    """
    if qe_key[0] == 'p' or qe_key[0] == 'x':
        # Lensing estimate (both gradient and curl)
        cL_out = get_spin_raise(0, 2 * lmax)
        if qe_key in ['ptt', 'xtt']:
            cltt = cls_weight.get('tt', np.zeros(lmax + 1))[:lmax + 1]
            lega = qeleg(0, 0,  -np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, 1, get_spin_raise(0, lmax) * cltt)
            return [qe(lega, legb, cL_out)]

        elif qe_key in ['p_p', 'x_p']:
            qes = []
            clee = cls_weight.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            clbb = cls_weight.get('bb', np.zeros(lmax + 1))[:lmax + 1]
            assert np.all(clbb == 0.), 'not implemented (but easy)'
            # E-part. G = -1/2 _{2}P - 1/2 _{-2}P
            lega = qeleg(2, 2, -0.5 * np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, -1, 0.5 * get_spin_raise(-2, lmax) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(2, 2,  -0.5 * np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, -1, 0.5 * get_spin_raise(-2, lmax) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, -0.5 *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, 3, 0.5 * get_spin_raise(2, lmax) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, -0.5 *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, 3, 0.5 * get_spin_raise(2, lmax) * clee)
            qes.append(qe(lega, legb, cL_out))

            return qes
        elif qe_key in ['p', 'x']:
            #Here T^WF contains C_\ell^{TE} \bar E and E^{WF} contains C_\ell^{TE} \bar T
            qes = get_qes('ptt', lmax, cls_weight) + get_qes('p_p', lmax, cls_weight)
            clte = cls_weight.get('te', np.zeros(lmax + 1))[:lmax + 1]
            lega = qeleg( 0, 0,  -np.ones(lmax + 1, dtype=float))
            qes.append(qe(lega, qeleg(2, 1, 0.5 * get_spin_raise(0, lmax) * clte), cL_out))
            qes.append(qe(lega, qeleg(-2, 1, 0.5 * get_spin_raise(0, lmax) * clte), cL_out))

            lega = qeleg(2,  2, -0.5 * np.ones(lmax + 1, dtype=float))
            qes.append(qe(lega, qeleg(0, -1, get_spin_raise(-2, lmax) * clte), cL_out))

            lega = qeleg(-2, -2, -0.5 * np.ones(lmax + 1, dtype=float))
            qes.append(qe(lega, qeleg(0, 3, get_spin_raise(2, lmax) * clte), cL_out))
            return qes

    elif qe_key[0] == 'f':
        cL_out = np.ones(2 * lmax + 1, dtype=float)
        if qe_key == 'ftt':
            lega = qeleg(0, 0, -np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, 0, -cls_weight.get('tt', np.zeros(lmax + 1))[:lmax + 1])
            return [qe(lega, legb, cL_out)]

        elif qe_key == 'f_p':
            qes = []
            clee = cls_weight.get('ee', np.zeros(lmax + 1))[:lmax + 1]
            clbb = cls_weight.get('bb', np.zeros(lmax + 1))[:lmax + 1]
            assert np.all(clbb == 0.), 'not implemented (but easy)'
            # E-part. G = -1/2 _{2}P - 1/2 _{-2}P
            lega = qeleg(2, 2, 0.5 * np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, -2,  0.5 * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(2, 2,  0.5 *np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, -2, 0.5 * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, 0.5 *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, 2, 0.5  * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, 0.5 *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, 2, 0.5 * clee)
            qes.append(qe(lega, legb, cL_out))
            return qes

        elif qe_key == 'f':
            #Here T^WF contains C_\ell^{TE} \bar E and E^{WF} contains C_\ell^{TE} \bar T
            qes = get_qes('ftt', lmax, cls_weight) + get_qes('f_p', lmax, cls_weight)
            clte = cls_weight.get('te', np.zeros(lmax + 1))[:lmax + 1] #: _0X_{lm} convention
            # Here Wiener-filtered T contains c_\ell^{TE} \bar E
            lega = qeleg( 0, 0,  np.ones(lmax + 1, dtype=float))
            legb = qeleg( 2, 0,  0.5  * clte)
            qes.append(qe(lega, legb, cL_out))
            legb = qeleg(-2, 0,  0.5  * clte)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(2,  2, 0.5 * np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, -2, clte)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, 0.5 * np.ones(lmax + 1, dtype=float))
            legb = qeleg( 0,  2, clte)
            qes.append(qe(lega, legb, cL_out))
            return qes

        else:
            assert 0

    elif qe_key[0] == 's':
        if qe_key == 'stt':
            lega = qeleg(0, 0, - np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, 0, -0.5 * np.ones(lmax + 1, dtype=float))
            cL_out = np.ones(2 * lmax + 1, dtype=float)
            return [qe(lega, legb, cL_out)]
        else:
            assert 0
    else:
        assert 0


def qe_spin_data(qe_key):
    """Returns out and in spin-weights of quadratic estimator from its quadratic estimator key.

        Output is an integer >= 0 (spin), a letter 'G' or 'C' if gradient or curl mode estimator, the
        unordered list of unique spins (>= 0) input to the estimator, and the spin-1 qe key.

    """
    qes = get_qes(qe_key, 1, dict())
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
                pk.dump(self.hashdict(), open(fn_hash, 'wb'))
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
        #This uses the implementation with no TB and EB couplings, in which case there is only a GG and CC response.
        s, GC, sins, ksp = qe_spin_data(k)
        assert s >= 0, s
        if s == 0: assert GC == 'G', (s, GC)
        fn = 'qe_' + ksp + k[1:] + '_source_%s_'%ksource + GC
        if self.npdb.get(fn) is None or recache:
            G, C = get_response(k, self.lmax_qe, ksource, self.cls_weight, self.cls_cmb, self.fal,
                                lmax_out=self.lmax_qlm)
            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_G')
                if s > 0: self.npdb.remove('qe_' + k[1:] + '_source_%s' % ksource + '_C')
            self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_G', G)
            if s > 0: self.npdb.add('qe_' + ksp + k[1:] + '_source_%s' % ksource + '_C', C)
        return self.npdb.get(fn)


def get_response(qe_key, lmax_qe, source, cls_weight, cls_cmb, fal_leg1, fal_leg2=None, lmax_out=None):
    """QE isotropic response.

    #FIXME: explain fal here

    """
    qes = get_qes(qe_key, lmax_qe, cls_weight)
    return _get_response(qes, lmax_qe, source, cls_cmb, fal_leg1, fal_leg2=fal_leg2, lmax_out=lmax_out)


get_response_sepTP = get_response # Here for historical reasons.

def _get_response(qes, lmax_qe, source,  cls_cmb, fal_leg1,
                          fal_leg2=None, lmax_out=None):
    tb = cls_cmb.get('tb', None)
    eb = cls_cmb.get('eb', None)
    assert tb is None and eb is None, "This response implementation only for vanishing EB and TB CMB-sky spectra."
    lmax_qlm = min(2 * lmax_qe,  2 * lmax_qe if lmax_out is None else lmax_out)
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    RGG = np.zeros(lmax_qlm + 1, dtype=float)
    RCC = np.zeros(lmax_qlm + 1, dtype=float)

    def get_F(s1, s2, leg):
        # Matrix \bar _{s_1} X = F_{s_1,s_2} _{s_2} X
        assert s1 in [0, -2, 2] and s2 in [0, -2, 2] and leg in [1, 2]
        fal = fal_leg1 if leg == 1 else fal_leg2
        if s1 == 0:
            return fal['t'] if s2 == 0 else (0.5 * fal['te'] if 'te' in fal.keys() else None)
        if s1 in [-2, 2]:
            if s2 == 0: return fal['te'] if 'te' in fal.keys() else None
            return 0.5 * (fal['e'] + fal['b']) if s1 == s2 else 0.5 * (fal['e'] - fal['b'])
        else:
            assert 0

    for qe in qes:  # loop over all quadratic terms in estimator
        si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
        so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
        # We want R^{a, st}  and R^{-a, st}
        for s2 in ([0, -2, 2]):
            FA = get_F(si, s2, 1)
            if FA is not None:
                for t2 in ([0, -2, 2]):
                    FB = get_F(ti, t2, 2)
                    if FB is not None:
                        rW_st, prW_st, mrW_st, s_cL_st = get_covresp(source, -s2, t2, cls_cmb, len(FB) - 1)
                        clA = joincls([qe.leg_a.cl, FA])
                        clB = joincls([qe.leg_b.cl, FB, mrW_st])
                        Rpr_st = wignerc(clA, clB, so, s2, to, -s2 + rW_st, lmax_out=lmax_qlm) * s_cL_st[:lmax_qlm + 1]

                        rW_ts, prW_ts, mrW_ts, s_cL_ts = get_covresp(source, -t2, s2, cls_cmb, len(FA) - 1)
                        clA = joincls([qe.leg_a.cl, FA, mrW_ts])
                        clB = joincls([qe.leg_b.cl, FB])
                        Rpr_st += wignerc(clA, clB, so, -t2 + rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts[:lmax_qlm + 1]
                        assert rW_st == rW_ts and rW_st >= 0, (rW_st, rW_ts)
                        if rW_st > 0:
                            clA = joincls([qe.leg_a.cl, FA])
                            clB = joincls([qe.leg_b.cl, FB, prW_st])
                            Rmr_st = wignerc(clA, clB, so, s2, to, -s2 - rW_st, lmax_out=lmax_qlm) * s_cL_st[:lmax_qlm + 1]

                            clA = joincls([qe.leg_a.cl, FA, prW_ts])
                            clB = joincls([qe.leg_b.cl, FB])
                            Rmr_st += wignerc(clA, clB, so, -t2 - rW_ts, to, t2, lmax_out=lmax_qlm) * s_cL_ts[:lmax_qlm + 1]
                        else:
                            Rmr_st = Rpr_st
                        RGG += (-1) ** (so + to + rW_ts) * (Rpr_st + Rmr_st * (-1) ** rW_st) * qe.cL[:lmax_qlm + 1]
                        RCC += (-1) ** (so + to + rW_ts) * (Rpr_st - Rmr_st * (-1) ** rW_st) * qe.cL[:lmax_qlm + 1]
    return RGG, RCC


def get_mf_resp(qe_key, cls_cmb, cls_ivfs, lmax_qe, lmax_out):
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
            cl1 = get_spin_coupling(s1, s2, cls_ivfs)[:lmax_qe + 1] * (0.5 ** (s1 != 0) * 0.5 ** (s2 != 0))
            # These 1/2 factor from the factor 1/2 in each B of B Covi B^dagger, where B maps spin-fields to T E B.
            cl2 = get_spin_coupling(s2, s1, cls_cmb)[:lmax_cmb + 1]
            cl2[:lmax_qe + 1] -= get_spin_coupling(s2, s1, cl_cmbtoticmb)[:lmax_qe + 1]
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = get_spin_lower(s2, lmax_cmb) if a == - 1 else get_spin_raise(s2, lmax_cmb)
                    for b in [1]: # a, b symmetry
                        aj = get_spin_lower(-s1, lmax_cmb) if b == 1 else get_spin_raise(-s1, lmax_cmb)
                        hL = 2 * (-1) ** (s1 + s2) * wignerc(cl1, cl2 * ai * aj, s2, s1, -s2 - a, -s1 - b, lmax_out=lmax_out)
                        GL += (- a * b) * hL
                        CL += (-1) * hL

    # Build remaining Fisher term II:
    for s1 in spins:
        for s2 in spins:
            cl1 = get_spin_coupling(s2, s1, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s1 != 0))
            cl2 = get_spin_coupling(s1, s2, cl_cmbtoti)[:lmax_qe + 1] * (0.5 ** (s2 != 0))
            if np.any(cl1) and np.any(cl2):
                for a in [-1, 1]:
                    ai = get_spin_lower(s2, lmax_qe) if a == -1 else get_spin_raise(s2, lmax_qe)
                    for b in [1]:
                        aj = get_spin_lower(s1, lmax_qe) if b == 1 else get_spin_raise(s1, lmax_qe)
                        hL = 2 * (-1) ** (s1 + s2) * wignerc(cl1 * ai, cl2 * aj, -s2 - a, -s1, s2, s1 - b, lmax_out=lmax_out)
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

GL_cache = {}

def wignerc(cl1, cl2, sp1, s1, sp2, s2, lmax_out=None):
    """Legendre coeff. of $ (\\xi_{sp1,s1} * \\xi_{sp2,s2})(\\cos \\theta)$ from their harmonic series.

        The integrand is always a polynomial, of max. degree lmax1 + lmax2 + lmax_out.
        We use Gauss-Legendre integration to solve this exactly.
    """
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmax_out = lmax1 + lmax2 if lmax_out is None else lmax_out
    lmaxtot = lmax1 + lmax2 + lmax_out
    N = (lmaxtot + 2 - lmaxtot % 2) // 2
    if not 'xg wg %s' % N in GL_cache.keys():
        GL_cache['xg wg %s' % N] = wigners.get_xgwg(-1., 1., N) if HASWIGNER else gauleg.get_xgwg(N)
    xg, wg = GL_cache['xg wg %s' % N]

    if HASWIGNER:
        if np.iscomplexobj(cl1):
            xi1 = wigners.wignerpos(np.real(cl1), xg, sp1, s1) + 1j * wigners.wignerpos(np.imag(cl1), xg, sp1, s1)
        else:
            xi1 = wigners.wignerpos(cl1, xg, sp1, s1)
        if np.iscomplexobj(cl2):
            xi2 = wigners.wignerpos(np.real(cl2), xg, sp2, s2) + 1j * wigners.wignerpos(np.imag(cl2), xg, sp2, s2)
        else:
            xi2 = wigners.wignerpos(cl2, xg, sp2, s2)
        xi1xi2w = xi1 * xi2 * wg
        if np.iscomplexobj(xi1xi2w):
            ret = wigners.wignercoeff(np.real(xi1xi2w), xg, sp1 + sp2, s1 + s2, lmax_out)
            ret = ret + 1j * wigners.wignercoeff(np.imag(xi1xi2w), xg, sp1 + sp2, s1 + s2, lmax_out)
            return ret
        else:
            return wigners.wignercoeff(xi1xi2w, xg, sp1 + sp2, s1 + s2, lmax_out)
    else:
        xi1 = gaujac.get_rspace(cl1, xg, sp1, s1)
        xi2 = gaujac.get_rspace(cl2, xg, sp2, s2)
        return 2. * np.pi * np.dot(gaujac.get_wignerd(lmax_out, xg, sp1 + sp2, s1 + s2), wg * xi1 * xi2)


def get_spin_raise(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin raising operator.

        +\sqrt{ (l - s) (l + s + 1) } for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin lowering operator.

        -\sqrt{ (l + s) (l - s + 1) } for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def get_spin_coupling(s1, s2, cls):
    """Spin-weighted power spectrum <_{s1}X_{lm} _{s2}X^*{lm}>

    Note:
        The output is real unless necessary. This uses the spin-field conventions where _0X_{lm} = -T_{lm}.

    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(get_spin_coupling(-s1, -s2, cls))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, 'not implemented')
    if s1 == 0:
        if s2 == 0:
            return cls['tt']
        tb = cls.get('tb', None)
        return  cls['te'] if tb is None else  cls['te'] - 1j * np.sign(s2) * tb
    elif s1 == 2:
        if s2 == 0:
            tb = cls.get('tb', None)
            return  cls['te'] if tb is None else  cls['te'] + 1j * tb
        elif s2 == 2:
            return cls['ee'] + cls['bb']
        elif s2 == -2:
            eb = cls.get('eb', None)
            return  cls['ee'] - cls['bb'] if eb is None else  cls['ee'] - cls['bb'] + 2j * eb
        else:
            assert 0