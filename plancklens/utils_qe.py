import numpy as np
import healpy as hp
from plancklens import utils as ut, utils_spin as uspin

class qeleg:
    def __init__(self, spin_in, spin_out, cl):
        self.spin_in = spin_in
        self.spin_ou = spin_out
        self.cl = cl

    def __eq__(self, leg):
        if self.spin_in != leg.spin_in or self.spin_ou != leg.spin_ou or self.get_lmax() != self.get_lmax():
            return False
        return np.all(self.cl == leg.cl)

    def __mul__(self, other):
        return qeleg(self.spin_in, self.spin_ou, self.cl * other)

    def __add__(self, other):
        assert self.spin_in == other.spin_in and self.spin_ou == other.spin_ou
        lmax = max(self.get_lmax(), other.get_lmax())
        cl = np.zeros(lmax + 1, dtype=float)
        cl[:len(self.cl)] += self.cl
        cl[:len(other.cl)] += other.cl
        return qeleg(self.spin_in, self.spin_ou, cl)

    def copy(self):
        return qeleg(self.spin_in, self.spin_ou, np.copy(self.cl))

    def get_lmax(self):
        return len(self.cl) - 1


class qeleg_multi:
    def __init__(self, spins_in, spin_out, cls):
        assert isinstance(spins_in, list) and isinstance(cls, list) and len(spins_in) == len(cls)
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
            glm += hp.almxfl(ut.alm_copy(gclm[0], lmax), sgn_g * cl)
            if np.any(gclm[1]):
                clm += hp.almxfl(ut.alm_copy(gclm[1], lmax), sgn_c * cl)
        glm *= -1
        if self.spin_ou > 0: clm *= -1
        Red, Imd = uspin.alm2map_spin((glm, clm), nside, abs(self.spin_ou), lmax)
        if self.spin_ou < 0 and self.spin_ou % 2 == 1: Red *= -1
        if self.spin_ou < 0 and self.spin_ou % 2 == 0: Imd *= -1
        return Red + 1j * Imd


    def get_lmax(self):
        return np.max([len(cl) for cl in self.cls]) - 1

class qe:
    def __init__(self, leg_a:qeleg, leg_b:qeleg, cL):
        assert leg_a.spin_ou +  leg_b.spin_ou >= 0
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.cL = cL

    def get_lmax_a(self):
        return self.leg_a.get_lmax()

    def get_lmax_b(self):
        return self.leg_b.get_lmax()

def qe_eval(qe_list, nside, get_alm, lmax_qlm, verbose=True, get_alm2=None):
    """Evaluation of a QE from its list of leg definitions.

        Args:
            qe_list: list of qe instances
            nside: the estimator are calculated in position space at healpy resolution nside
            get_alm: callable with 't', 'e', 'b' arguments, giving the corresponding inverse-variance filtered CMB maps
            lmax_qlm: outputs are given up to multipole lmax_qlm
            get_alm2 : callable for second leg if different from the first (symmetrizes estimator by default)

        Returns:
            glm and clm healpy arrays (gradient and curl terms of the QE estimate)

    """
    if get_alm2 is None:
        get_alm2 = get_alm
    symmetrize = not (get_alm2 is get_alm)
    qes = qe_compress(qe_list, verbose=verbose)
    qe_spin = qes[0][0].spin_ou + qes[0][1].spin_ou
    cL_out = qes[0][-1](np.arange(lmax_qlm + 1))
    assert qe_spin >= 0, qe_spin
    for q in qes[1:]:
        assert np.all(q[-1](np.arange(lmax_qlm + 1)) == cL_out)
        assert q[0].spin_ou + q[1].spin_ou == qe_spin
    d = np.zeros(hp.nside2npix(nside), dtype=complex)
    for i, q in enumerate(qes):
        if verbose:
            print("QE %s out of %s :"%(i + 1, len(qes)))
            print("in-spins 1st leg and out-spin" ,q[0].spins_in, q[0].spin_ou)
            print("in-spins 2nd leg and out-spin", q[1].spins_in, q[1].spin_ou)
        d += q[0](get_alm, nside) * q[1](get_alm2, nside)
        if symmetrize:
            d += q[0](get_alm2, nside) * q[1](get_alm, nside)
    glm, clm = uspin.map2alm_spin((d.real, d.imag), qe_spin, lmax=lmax_qlm)
    if symmetrize:
        glm *= 0.5
        clm *= 0.5
    hp.almxfl(glm, cL_out, inplace=True)
    if np.any(clm):
        hp.almxfl(clm, cL_out, inplace=True)
    return glm, clm


def qe_proj(qe_list, a, b):
    """Projection of a list of QEs onto another QE using only a subset of maps.

        Args:
            qe_list: list of qe instances
            a: (in 't', 'e', 'b') The 1st leg of the output qes will only use this field
            b: (in 't', 'e', 'b') The 2nd leg of the output qes will only use this field
    """
    assert a in ['t', 'e', 'b'] and b in ['t', 'e', 'b']
    l_in = [0] if a == 't' else [-2, 2]
    r_in = [0] if b == 't' else [-2, 2]
    qes_ret = []
    for q in qe_list:
        si, ri = (q.leg_a.spin_in, q.leg_b.spin_in)
        if si in l_in and ri in r_in:
            leg_a = q.leg_a.copy()
            leg_b = q.leg_b.copy()
            if si == 0 and ri == 0:
                qes_ret.append(qe(leg_a, leg_b, q.cL))
            elif si == 0 and abs(ri) > 0:
                sgn = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a, leg_b * 0.5 * sgn, q.cL))
            elif ri == 0 and abs(si) > 0:
                sgn = 1 if a == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgn, leg_b, q.cL))
            elif abs(ri) > 0 and abs(si) > 0:
                sgna = 1 if a == 'e' else -1
                sgnb = 1 if b == 'e' else -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5, leg_b * 0.5 * sgnb, q.cL))
                leg_a.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5 * sgnb, q.cL))
                leg_b.spin_in *= -1
                qes_ret.append(qe(leg_a * 0.5 * sgna, leg_b * 0.5, q.cL))
            else:
                assert 0, (si, ri)
    return qe_simplify(qes_ret)


def qe_simplify(qe_list, _swap=False, verbose=False):
    """Simplifies a list of QE estimators by co-adding terms when possible.


    """
    skip = []
    qes_ret = []
    qes = [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qe_list] if _swap else qe_list
    for i, qe1 in enumerate(qes):
        if i not in skip:
            leg_a = qe1.leg_a.copy()
            leg_b = qe1.leg_b.copy()
            for j, qe2 in enumerate(qes[i + 1:]):
                if qe2.leg_a == leg_a:
                    if qe2.leg_b.spin_in == qe1.leg_b.spin_in and qe2.leg_b.spin_ou == qe1.leg_b.spin_ou:
                        Ls = np.arange(max(qe1.leg_b.get_lmax(), qe2.leg_b.get_lmax()) + 1)
                        if np.all(qe1.cL(Ls) == qe2.cL(Ls)):
                            leg_b += qe2.leg_b
                            skip.append(j + i + 1)
            if np.any(leg_a.cl) and np.any(leg_b.cl):
                qes_ret.append(qe(leg_a, leg_b, qe1.cL))
    if verbose and len(skip) > 0:
        print("%s terms down from %s" % (len(qes_ret), len(qes)))
    if not _swap:
        return qe_simplify(qes_ret, _swap=True, verbose=verbose)
    return [qe(q.leg_b.copy(), q.leg_a.copy(), q.cL) for q in qes_ret]


def qe_compress(qes, verbose=True):
    """This combines pairs of estimators with identical 1st leg to reduce the number of spin transform in its evaluation

    """
    # NB: this only compares first legs.
    skip = []
    qes_compressed = []
    for i, qi in enumerate(qes):
        if i not in skip:
            lega = qi.leg_a
            lega_m = qeleg_multi([qi.leg_a.spin_in], qi.leg_a.spin_ou, [qi.leg_a.cl])
            legb_m = qeleg_multi([qi.leg_b.spin_in], qi.leg_b.spin_ou, [qi.leg_b.cl])
            for j, qj in enumerate(qes[i + 1:]):
                if qj.leg_a == lega and legb_m.spin_ou == qj.leg_b.spin_ou:
                    legb_m += qj.leg_b
                    skip.append(i + 1 + j)
            qes_compressed.append( (lega_m, legb_m, qi.cL))
    if len(skip) > 0 and verbose:
        print("%s alm2map_spin transforms now required, down from %s"%(2 * (len(qes) -len(skip)) , 2 * len(qes)) )
    return qes_compressed