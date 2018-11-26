from __future__ import absolute_import

import os
import numpy as np

from plancklens2018.wigners import gaujac
from plancklens2018.wigners import gauleg
from plancklens2018 import sql


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
        # FIXME: finish this
        #m = hp.alm2map_spin(lega_dlm, nside, self.leg_a.spin_ou, self.leg_a.get_lmax())
        #m *= hp.alm2map_spin(legb_dlm, nside, self.leg_b.spin_ou, self.leg_b.get_lmax())

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
    """ Defines the responses terms for an anisotropy source.

    Args:
        source (str): anisotropy source (e.g. 'p', 'f', 's', ...).
        lmax (int): responses are given up to lmax.

    Returns:
        4-tuple (r, rR, -rR, cL):  source spin response *r* (positive or zero),
        the harmonic responses for +r and -r (2 1d-arrays), and the scaling between the G/C modes
        and the potentials of interest (for lensing, \phi_{LM}, \Omega_{LM} = \sqrt{L (L + 1)} G_{LM}, C_{LM}).

    """
    lmax_cL = 2 *  lmax
    if source == 'p': # lensing (gradient and curl): _sX -> _sX -  1/2 alpha_1 \eth _sX - 1/2 \alpha_{-1} \bar \eth _sX
        return {s : (1, -0.5 * get_alpha_lower(s, lmax),
                        -0.5 * get_alpha_raise(s, lmax),
                        np.sqrt(np.arange(lmax_cL + 1) * np.arange(1, lmax_cL + 2, dtype=float))) for s in [0, -2, 2]}
    if source == 'f': # Modulation: _sX -> _sX + f _sX.
        return {s : (0, 0.5 * np.ones(lmax + 1, dtype=float),
                        0.5 * np.ones(lmax + 1, dtype=float),
                        np.ones(lmax_cL + 1, dtype=float)) for s in [0, -2, 2]}
    assert 0, source + ' response legs not implemented'

def get_qe_sepTP(qe_key, lmax, cls_weight):
    """ Defines the quadratic estimator weights for quadratic estimator key.

    Args:
        qe_key (str): quadratic estimator key (e.g., ptt, p_p, ... )
        lmax (int): weights are built up to lmax.
        cls_len (dict): CMB spectra entering the weights

    #FIXME:
        * lmax_A, lmax_B, lmaxout!

    The weights are defined by their action on the inverse-variance filtered $ _{s}\\bar X_{lm}$.
    (It is useful to remember that by convention  $_{0}X_{lm} = - T_{lm}$)

    """
    def _sqrt(cl):
        ret = np.zeros(len(cl), dtype=float)
        ret[np.where(cl > 0)] = np.sqrt(cl[np.where(cl > 0)])
        return ret

    if qe_key[0] == 'p' or qe_key[0] == 'x':
        # Lensing estimate (both gradient and curl)
        if qe_key in ['ptt', 'xtt']:
            cL_out = -np.sqrt(np.arange(2 * lmax + 1) * np.arange(1, 2 * lmax + 2, dtype=float) )

            cltt = cls_weight['tt'][:lmax + 1]
            lega = qeleg(0, 0,  np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, 1,  np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)) * cltt)

            return [qe(lega, legb, cL_out)]

        elif qe_key in ['p_p', 'x_p']:
            qes = []
            cL_out = -np.sqrt(np.arange(2 * lmax + 1) * np.arange(1, 2 * lmax + 2, dtype=float) )
            clee = cls_weight['ee'][:lmax + 1]
            clbb = cls_weight['bb'][:lmax + 1]
            assert np.all(clbb == 0.), 'not implemented (but easy)'
            #FIXME: where does this 0.5 comes about??
            fac = 0.5
            # E-part. G = -1/2 _{2}P - 1/2 _{-2}P
            lega = qeleg(2, 2, fac * np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, -1,  0.5 * _sqrt(np.arange(2, lmax + 3) * np.arange(-1, lmax, dtype=float)) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(2, 2,  fac *np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, -1, 0.5 * _sqrt(np.arange(2, lmax + 3) * np.arange(-1, lmax, dtype=float)) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, fac *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(2, 3,0.5 * _sqrt(np.arange(-2, lmax - 1) * np.arange(3, lmax + 4, dtype=float)) * clee)
            qes.append(qe(lega, legb, cL_out))

            lega = qeleg(-2, -2, fac *  np.ones(lmax + 1, dtype=float))
            legb = qeleg(-2, 3, 0.5 * _sqrt(np.arange(-2, lmax - 1) * np.arange(3, lmax + 4, dtype=float)) * clee)
            qes.append(qe(lega, legb, cL_out))

            return qes
        elif qe_key in ['p', 'x']:
            #FIXME: signs...
            cL_out = -np.sqrt(np.arange(2 * lmax + 1) * np.arange(1, 2 * lmax + 2, dtype=float) )
            clte = cls_weight['te'][:lmax + 1] #: _0X_{lm} convention

            qes = get_qe_sepTP('ptt', lmax, cls_weight) + get_qe_sepTP('p_p', lmax, cls_weight)

            # Here used Wiener-filtered T contains c_\ell^{TE} \bar E
            lega = qeleg( 0, 0,  np.ones(lmax + 1, dtype=float))
            legb = qeleg( 2, 1,  -0.5 * np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)) * clte)
            qes.append(qe(lega, legb, cL_out))
            legb = qeleg(-2, 1,  -0.5 * np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2, dtype=float)) * clte)
            qes.append(qe(lega, legb, cL_out))

            # E-mode contains C_\ell^{te} \bar T
            lega = qeleg(2,  2, np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, -1, -_sqrt(np.arange(2, lmax + 3) * np.arange(-1, lmax, dtype=float)) * clte)
            qes.append(qe(lega, legb, cL_out))


            lega = qeleg(-2, -2, np.ones(lmax + 1, dtype=float))
            legb = qeleg( 0,  3,-_sqrt(np.arange(-2, lmax - 1) * np.arange(3, lmax + 4, dtype=float)) * clte)
            qes.append(qe(lega, legb, cL_out))

            return qes

    elif qe_key[0] == 'f':
        if qe_key == 'ftt':
            lega = qeleg(0, 0, -np.ones(lmax + 1, dtype=float))
            legb = qeleg(0, 0, -cls_weight['tt'][:lmax + 1])
            cL_out = np.ones(2 * lmax + 1, dtype=float)
            return [qe(lega, legb, cL_out)]
        else:
            assert 0
    else:
        assert 0

class resp_lib_simple:
    def __init__(self, lib_dir, lmax_qe, cls_weight, cls_cmb, fal):
        self.lmax_qe = lmax_qe
        self.cls_weight = cls_weight
        self.cls_cmb = cls_cmb
        self.fal = fal
        self.lib_dir = lib_dir
        self.npdb = sql.npdb(os.path.join(lib_dir))
        #FIXME: hashdict

    def get_response(self, k, source, recache=False):
        #FIXME: GC
        assert k[0] in ['p', 'x'], 'FIXME'
        fn = 'qe_' + k[1:] + '_source_%s'%source + ('_G' if k[0] != 'x' else '_C')
        if self.npdb.get(fn) is None or recache:
            G, C = get_response_sepTP(k, self.lmax_qe, source, self.cls_weight, self.cls_cmb, self.fal)
            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('qe_' + k[1:] + '_source_%s' % source + '_G')
                self.npdb.remove('qe_' + k[1:] + '_source_%s' % source + '_C')
            self.npdb.add('qe_' + k[1:] + '_source_%s' % source + '_G', G)
            self.npdb.add('qe_' + k[1:] + '_source_%s' % source + '_C', C)
        return self.npdb.get(fn)

class nhl_lib_simple:
    """Analytical unnormalized N0 library. """
    def __init__(self, lib_dir, lmax_qe, cls_weight, cls_ivfs):
        self.lmax_qe = lmax_qe
        self.cls_weight = cls_weight
        self.cls_ivfs = cls_ivfs
        self.lib_dir = lib_dir
        self.npdb = sql.npdb(os.path.join(lib_dir))
        #FIXME: hashdict

    def get_nhl(self, k1, k2, recache=False):
        #FIXME: GC
        assert k1[0] in ['p', 'x'] and k2[0] in ['p', 'x'], 'FIXME'
        if k1[0] != k2[0]: return np.zeros(2 * self.lmax_qe + 1, dtype=float)
        fn = 'anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + ('_G' if k1[0] != 'x' else '_C')
        if self.npdb.get(fn) is None or recache:
            G, C = get_nhl(k1, k2, self.cls_weight, self.cls_ivfs, self.lmax_qe)
            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_G')
                self.npdb.remove('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_C')
            self.npdb.add('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_G', G)
            self.npdb.add('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_C', C)
        return self.npdb.get(fn)



def get_response_sepTP(qe_key, lmax_qe, source, cls_weight, cls_cmb, fal_leg1, fal_leg2=None, ret_terms=False):
    lmax_source = lmax_qe # I think that's fine as long as we the same lmax on both legs.
    qes = get_qe_sepTP(qe_key, lmax_qe, cls_weight)
    resps = get_resp_legs(source, lmax_source)
    lmax_qlm = 2 * lmax_qe
    fal_leg2 = fal_leg1 if fal_leg2 is None else fal_leg2
    #FIXME: fix all lmaxs etc
    Rggcc = np.zeros((2, lmax_qlm + 1), dtype=float)
    terms = []
    for qe in qes: # loop over all quadratic terms in estimator

        si, ti = (qe.leg_a.spin_in, qe.leg_b.spin_in)
        so, to = (qe.leg_a.spin_ou, qe.leg_b.spin_ou)
        print(si, ti)

        # Rst,r involves R^r, -ti}
        def add(si, ti, so, to, fla, flb):
            if np.all(fla == 0.) or np.all(flb == 0.):
                return np.zeros((2, lmax_qlm + 1), dtype=float)
            print 'si ti', si, ti
            si = si * -1
            ti = ti * -1 # FIXME: This seems works for Pol, but why ??? (exc. for fac of 2 in qest file)
            cpling = get_coupling(si, -ti, cls_cmb)[:lmax_qe + 1]

            r, prR, mrR, s_cL = resps[-ti]  # There should always be a single term here.
            Rst_pr = get_hl(prR * cpling * qe.leg_a.cl * fla, qe.leg_b.cl * flb, ti - r, so, -ti, to) * s_cL[:lmax_qlm + 1]
            Rst_mr = get_hl(mrR * cpling * qe.leg_a.cl * fla, qe.leg_b.cl * flb, ti + r, so, -ti, to) * s_cL[:lmax_qlm + 1]
            # Swap s and t all over
            cpling *= (-1) ** (si - ti)
            r2, prR, mrR, s_cL = resps[-si]
            assert r2 == r, (r, r2)
            Rts_pr = get_hl(prR * cpling * qe.leg_b.cl * flb, qe.leg_a.cl * fla, si - r, to, -si, so) * s_cL[:lmax_qlm + 1]
            Rts_mr = get_hl(mrR * cpling * qe.leg_b.cl * flb, qe.leg_a.cl * fla, si + r, to, -si, so) * s_cL[:lmax_qlm + 1]
            gg = (Rst_mr + Rts_mr + (-1) ** r * (Rst_pr + Rts_pr)) * qe.cL[:lmax_qlm + 1]
            cc = (Rst_mr + Rts_mr - (-1) ** r * (Rst_pr + Rts_pr)) * qe.cL[:lmax_qlm + 1]
            terms.append(Rst_mr * qe.cL[:lmax_qlm + 1])
            terms.append(Rst_pr * qe.cL[:lmax_qlm + 1])
            terms.append(Rts_mr * qe.cL[:lmax_qlm + 1])
            terms.append(Rts_pr * qe.cL[:lmax_qlm + 1])
            return np.array([gg, cc])

        if si == 0 and ti == 0:
            Rggcc += add(si, ti, so, to, fal_leg1['t'], fal_leg2['t'])

        else:
            # Here we use _{\pm |s|}X = \pm^{s} 1/2 [ _{|s|} d_{lm}(f^g \pm f^c) _{|s|}d_{lm} + (-1)^{s} _{-|s|} d_{lm}(f^g \mp f^c) _{-|s|}d_{lm}
            #FIXME
            sgs = 1 if si > 0 else (1 if abs(si)%2 == 0 else -1)
            sgt = 1 if ti > 0 else (1 if abs(ti)%2 == 0 else -1)

            prefac = 0.25 * sgs * sgt
            fla = fal_leg1['e'] + np.sign(si) * fal_leg1['b'] if abs(si) == 2 else fal_leg1['t']
            flb = fal_leg2['e'] + np.sign(ti) * fal_leg2['b'] if abs(ti) == 2 else fal_leg2['t']
            Rggcc += prefac * add(abs(si), abs(ti), so, to, fla, flb)

            fla = fal_leg1['e'] + np.sign(si) * fal_leg1['b'] if abs(si) == 2 else fal_leg1['t']
            flb = fal_leg2['e'] - np.sign(ti) * fal_leg2['b'] if abs(ti) == 2 else fal_leg2['t']
            Rggcc += (-1) ** ti *  prefac * add(abs(si), -abs(ti), so, to, fla, flb)

            fla = fal_leg1['e'] - np.sign(si) * fal_leg1['b'] if abs(si) == 2 else fal_leg1['t']
            flb = fal_leg2['e'] + np.sign(ti) * fal_leg2['b'] if abs(ti) == 2 else fal_leg2['t']
            Rggcc += (-1) ** si * prefac * add(-abs(si), abs(ti), so, to, fla, flb)

            fla = fal_leg1['e'] - np.sign(si) * fal_leg1['b'] if abs(si) == 2 else fal_leg1['t']
            flb = fal_leg2['e'] - np.sign(ti) * fal_leg2['b'] if abs(ti) == 2 else fal_leg2['t']
            Rggcc += (-1) ** (ti + si) * prefac * add(-abs(si), -abs(ti), so, to, fla, flb)
    return Rggcc if not ret_terms else (Rggcc, terms)

    #FIXME: finish this:

def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_qe, ret_terms=None):
    """(Semi-)Analytical noise level calculation.

    """
    #FIXME: works for ptt p_p p jtTP, but overall mignus sign w.r.t. PDF?
    qes1 = get_qe_sepTP(qe_key1, lmax_qe, cls_weights)
    qes2 = get_qe_sepTP(qe_key2, lmax_qe, cls_weights)
    G_N0 = np.zeros(2 * lmax_qe + 1, dtype=float)
    C_N0 = np.zeros(2 * lmax_qe + 1, dtype=float)
    def _joincls(cls_list):
        lmaxp1 = np.min([len(cl) for cl in cls_list])
        return np.prod(np.array([cl[:lmaxp1] for cl in cls_list]), axis=0)
    terms = []
    for qe1 in qes1:
        for qe2 in qes2:
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0
            assert uo + vo >= 0

            sgn = -1
            sgn2 = 1
            sgn_fix = -1

            sgn_R = (-1) ** (uo + vo + ui + vi)
            clsu = _joincls([qe1.leg_a.cl, qe2.leg_a.cl, get_coupling(si, -ui, cls_ivfs)])
            cltv = _joincls([qe1.leg_b.cl, qe2.leg_b.cl, get_coupling(ti, -vi, cls_ivfs)])
            R_sutv = sgn_R * _joincls([get_hl(clsu ,cltv, sgn*uo, sgn2*so, sgn*vo, sgn2*to), qe1.cL, qe2.cL])

            clsv = _joincls([qe1.leg_a.cl, qe2.leg_b.cl, get_coupling(si, -vi, cls_ivfs)])
            cltu = _joincls([qe1.leg_b.cl, qe2.leg_a.cl, get_coupling(ti, -ui, cls_ivfs)])
            R_svtu = sgn_R * _joincls([get_hl(clsv ,cltu, sgn*vo, sgn2*so, sgn*uo, sgn2*to), qe1.cL, qe2.cL])

            # s and t goes to -s, -t. We have mabe a minus sign (-1)**(si + so + ti + to) on top from the change in weights
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clmsu = _joincls([qe1.leg_a.cl * sgnms, qe2.leg_a.cl, get_coupling(-si, -ui, cls_ivfs)])
            clmtv = _joincls([qe1.leg_b.cl * sgnmt, qe2.leg_b.cl, get_coupling(-ti, -vi, cls_ivfs)])
            R_msumtv =  sgn_R * _joincls([get_hl(clmsu ,clmtv, sgn*uo, -sgn2*so, sgn*vo, -sgn2*to), qe1.cL, qe2.cL])

            clmsv = _joincls([qe1.leg_a.cl * sgnms, qe2.leg_b.cl, get_coupling(-si, -vi, cls_ivfs)])
            clmtu = _joincls([qe1.leg_b.cl * sgnmt, qe2.leg_a.cl, get_coupling(-ti, -ui, cls_ivfs)])
            R_msvmtu = sgn_R * _joincls([get_hl(clmsv ,clmtu, sgn*vo, -sgn2*so, sgn*uo, -sgn2*to), qe1.cL, qe2.cL])
        
            G_N0 += sgn_fix *  0.5 *  (R_sutv + R_svtu)
            G_N0 += sgn_fix *  0.5 * (-1) ** (to + so) * (R_msumtv  + R_msvmtu)
        
            C_N0 -= sgn_fix * 0.5 *  (R_sutv + R_svtu)
            C_N0 += sgn_fix * 0.5 * (-1) ** (to + so) * (R_msumtv  + R_msvmtu)
            terms = terms + [ 0.5 *R_sutv,  0.5 *R_svtu,  0.5 *R_msumtv,  0.5 *R_msvmtu]
    return (G_N0, C_N0) if not ret_terms else (G_N0, C_N0, terms)

GL_cache = {}

def get_hl(cl1, cl2, sp1, s1, sp2, s2):
    """Legendre coeff. of $ (\\xi_{sp1,s1} * \\xi_{sp2,s2})(\\cos \\theta)$ from their harmonic series."""
    print('get_hl::spins: ', sp1, s1, sp2, s2)
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmaxout = lmax1 + lmax2
    #FIXME:
    lmax_GL = lmax1 + lmax2 + 1
    if not 'xg wg %s' % lmax_GL in GL_cache.keys():
        GL_cache['xg wg %s' % lmax_GL] = gauleg.get_xgwg(lmax_GL)
    xg, wg = GL_cache['xg wg %s' % lmax_GL]
    xi1 = gaujac.get_rspace(cl1, xg, sp1, s1)
    xi2 = gaujac.get_rspace(cl2, xg, sp2, s2)
    return 2. * np.pi * np.dot(gaujac.get_wignerd(lmaxout, xg, sp1 + sp2, s1 + s2), wg * xi1 * xi2)

def get_alpha_raise(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin raising operator. """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_alpha_lower(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin lowering operator. """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def get_lensing_resp(s, lmax):
    """ 1/2 1d eth X + 1/2 -1d eth X """
    return  {1: -0.5 * get_alpha_lower(s, lmax), -1: -0.5 * get_alpha_raise(s, lmax)}

def get_coupling(s1, s2, cls):
    """<_{s1}X_{lm} _{s2}X^*{lm}>

    Note:
        This uses the spin-field conventions where _0X_{lm} = -T_{lm}
    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * get_coupling(-s1, -s2, cls)
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2 , 'not implemented')
    if s1 == 0 :
        if s2 == 0 :
            return cls['tt'].copy()
        return -cls['te'].copy()
    elif s1 == 2:
        if s2 == 0:
            return -cls['te'].copy()
        return cls['ee'] + np.sign(s2) * cls['bb']
