from __future__ import absolute_import

import numpy as np
import healpy as hp

from plancklens2018.wigners import gaujac
from plancklens2018.wigners import gauleg



class QEleg:
    def __init__(self, spin_in, spin_out, cl):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def get_lmax(self):
        return len(self.cl) - 1

class QE:
    def __init__(self, QElega, QElegb, cL):
        assert QElega.spin_ou +  QElegb.spin_ou >= 0
        self.QElega = QElega
        self.QElegb = QElegb
        self.cL = cL


def get_nhl(Q1, Q2, cls_len):
    si, ti, ui, vi = (Q1.QElega.spin_in, Q1.QElegb.spin_in, Q2.QElega.spin_in, Q2.QElegb.spin_in)
    so, to, uo, vo = (Q1.QElega.spin_ou, Q1.QElegb.spin_ou, Q2.QElega.spin_ou, Q2.QElegb.spin_ou)
    assert so + to >= 0
    assert uo + vo >= 0

    lmax = Q1.QElega.get_lmax()
    #FIXME: lmax etc
    clsu = Q1.QElega.cl * Q2.QElega.cl * get_coupling(si, ui, cls_len)[:lmax + 1]
    cltv = Q1.QElegb.cl * Q2.QElegb.cl * get_coupling(ti, vi, cls_len)[:lmax + 1]
    R_sutv = (-1) ** (uo + vo) * get_hl(clsu ,cltv, -uo, so, -vo, to) * Q1.cL * Q2.cL

    clsv = Q1.QElega.cl * Q2.QElegb.cl * get_coupling(si, vi, cls_len)[:lmax + 1]
    cltu = Q1.QElegb.cl * Q2.QElega.cl * get_coupling(ti, ui, cls_len)[:lmax + 1]
    R_svtu = (-1) ** (vo + uo) * get_hl(clsv ,cltu, -vo, so, -uo, to)* Q1.cL * Q2.cL

    clmsu = Q1.QElega.cl * Q2.QElega.cl * get_coupling(-si, ui, cls_len)[:lmax + 1]
    clmtv = Q1.QElegb.cl * Q2.QElegb.cl * get_coupling(-ti, vi, cls_len)[:lmax + 1]
    R_msumtv = (-1) ** (uo + vo) * get_hl(clmsu ,clmtv, -uo, -so, -vo, -to)* Q1.cL * Q2.cL

    clmsv = Q1.QElega.cl * Q2.QElegb.cl * get_coupling(-si, vi, cls_len)[:lmax + 1]
    clmtu = Q1.QElegb.cl * Q2.QElega.cl * get_coupling(-ti, ui, cls_len)[:lmax + 1]
    R_msvmtu = (-1) ** (vo + uo) * get_hl(clmsv ,clmtu, -vo, -so, -uo, -to)* Q1.cL * Q2.cL

    G_N0 =  0.25 * ( (R_sutv + R_svtu) * (1 + (-1) ** (si + ti + ui + vi)))
    G_N0 += 0.25 * (-1) ** (si + ti) * ( (R_msumtv + R_msvmtu) * (1 + (-1) ** (si + ti + ui + vi)))

    C_N0 = -0.25 * ((R_sutv + R_svtu) * (1 + (-1) ** (so + to + uo + vo)))
    C_N0 += 0.25 * (-1) ** (si + ti) * ((R_msumtv + R_msvmtu) * (1 + (-1) ** (si + ti + ui + vi)))
    return G_N0, C_N0, R_sutv, R_svtu, R_msumtv, R_msvmtu

GL_cache = {}

def get_hl(cl1, cl2, sp1, s1, sp2, s2):
    """ Legendre coeff of xi_sp1,s1(cost) * xi_sp2,s2(cost) , given Legendre coeff. of xi. GL quadrature. """
    print('spins: ', sp1, s1, sp2, s2)
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmaxout = lmax1 + lmax2
    lmax_GL = lmax1 + lmax2 + 1
    if not 'xg wg %s' % lmax_GL in GL_cache.keys():
        GL_cache['xg wg %s' % lmax_GL] = gauleg.get_xgwg(lmax_GL)
    xg, wg = GL_cache['xg wg %s' % lmax_GL]
    xi1 = gaujac.get_rspace(cl1, xg, sp1, s1)
    xi2 = gaujac.get_rspace(cl2, xg, sp2, s2)
    return 2. * np.pi * np.dot(gaujac.get_wignerd(lmaxout, xg, sp1 + sp2, s1 + s2), wg * xi1 * xi2)

def get_alpha_raise(s, lmax):
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_alpha_lower(s, lmax):
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def get_lensing_resp(s, lmax):
    """ 1/2 1d eth X + 1/2 -1d eth X """
    return  {1: 0.5 * get_alpha_lower(s, lmax), -1: 0.5 * get_alpha_raise(s, lmax)}

def get_coupling(s1, s2, cls_len):
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2 , 'not implemented')
    if s1 == 0 and s2 == 0 :
        return cls_len['tt'].copy()
    else:
        assert 0
        return None

def get_response(lmax, cls_len, ftl):
    # TT QE estimator for _1d
    s_in = 0
    t_in = 0
    s_out = 0
    t_out = 1
    fs_in = -ftl.copy()
    ft_in = -np.sqrt(np.arange(lmax + 1) * np.arange(1, lmax + 2)) * cls_len['tt'][:lmax + 1] * ftl

    ret = {-1: np.zeros(2 * lmax + 1, dtype=float), 1: np.zeros(2 * lmax + 1, dtype=float)}
    Cl = get_coupling(s_in, t_in, cls_len)[:lmax + 1]
    sgn = lambda sr : (-1) ** s_r
    if Cl is not None:
        Rls = get_lensing_resp(s_in, lmax)
        for s_r in Rls.keys():
            ret[s_in - s_r] += sgn(s_r) * get_hl(fs_in, ft_in  * Cl * Rls[s_r], s_in, s_out, -s_r, t_out)
        # Same but swapping some indices... (s_i-t_i, s_o-t_o)
        Rls = get_lensing_resp(t_in, lmax)
        for s_r in Rls.keys():
            ret[t_in - s_r] += sgn(s_r) * get_hl(ft_in, fs_in * Cl * Rls[s_r], t_in, t_out, -s_r, s_out)
    # Turn 1, -1 into G, C responses
    return ret[-1] + (-1) ** (s_out + t_out) * ret[1], ret[-1] - (-1) ** (s_out + t_out) * ret[1], ret



