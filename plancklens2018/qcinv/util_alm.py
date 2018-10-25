import numpy as np
#FIXME: / vs // in this file ?

def nlm2lmax(nlm):
    """ returns the lmax for an array of alm
    with length nlm. (same as hp.Alm.getlmax(s))"""

    lmax = int(np.floor(np.sqrt(2 * nlm) - 1))
    assert (lmax + 2) * (lmax + 1) // 2 == nlm
    return lmax


def lmax2nlm(lmax):
    """ returns the length of the alm array required for lmax """
    return (lmax + 1) * (lmax + 2) // 2


def alm_cl_cross(alm1, alm2, lmax=None):
    alm1_lmax = nlm2lmax(len(alm1))
    alm2_lmax = nlm2lmax(len(alm2))

    if lmax is None:
        lmax = min(alm1_lmax, alm2_lmax)
    assert (lmax <= min(alm1_lmax, alm2_lmax))

    ret = np.zeros(lmax + 1)
    for l in range(0, lmax + 1):
        ms = np.arange(1, l + 1)
        ret[l] = np.real(alm1[l] * alm2[l])
        ret[l] += 2. * np.sum(
            np.real(alm1[ms * (2 * alm1_lmax + 1 - ms) / 2 + l] * np.conj(alm2[ms * (2 * alm2_lmax + 1 - ms) / 2 + l])))
        ret[l] /= (2. * l + 1.)
    return ret


def alm_cl(alm, lmax=None):
    return alm_cl_cross(alm, alm, lmax)


def alm_splice(alm_lo, alm_hi, lsplit):
    """ returns an alm w/ lmax = lmax(alm_hi) which is
           alm_lo for (l <= lsplit)
           alm_hi for (l  > lsplit) """
    if hasattr(alm_lo, 'alm_splice'):
        return alm_lo.alm_splice(alm_hi, lsplit)

    alm_lo_lmax = nlm2lmax(len(alm_lo))
    alm_hi_lmax = nlm2lmax(len(alm_hi))

    assert (alm_lo_lmax >= lsplit)
    assert (alm_hi_lmax >= lsplit)

    ret = np.copy(alm_hi)
    for m in range(0, lsplit + 1):
        ret[(m * (2 * alm_hi_lmax + 1 - m) / 2 + m):(m * (2 * alm_hi_lmax + 1 - m) / 2 + lsplit + 1)] = alm_lo[(
        m * (2 * alm_lo_lmax + 1 - m) / 2 + m):(m * (2 * alm_lo_lmax + 1 - m) / 2 + lsplit + 1)]
    return ret


def alm_copy(alm, lmax=None):
    """ copies the alm array, with the option to reduce its lmax """

    alm_lmax = nlm2lmax(len(alm))
    assert (lmax <= alm_lmax)

    if (alm_lmax == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros(lmax2nlm(lmax), dtype=np.complex)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) / 2) + m):(m * (2 * lmax + 1 - m) / 2 + lmax + 1)] = \
                alm[(m * (2 * alm_lmax + 1 - m) / 2 + m):(m * (2 * alm_lmax + 1 - m) / 2 + lmax + 1)]
    return ret


def alm2rlm(alm):
    """ converts a complex alm to 'real harmonic' coefficients rlm. """

    lmax = nlm2lmax(len(alm))
    rlm = np.zeros((lmax + 1) ** 2)

    ls = np.arange(0, lmax + 1)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in range(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]].imag * rt2
    return rlm


def rlm2alm(rlm):
    """ converts 'real harmonic' coefficients rlm to complex alm. """

    lmax = int(np.sqrt(len(rlm)) - 1)
    assert ((lmax + 1) ** 2 == len(rlm))

    alm = np.zeros(lmax2nlm(lmax), dtype=np.complex)

    ls = np.arange(0, lmax + 1, dtype=np.int64)
    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in range(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) / 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1.j * rlm[l2s[m:] + 2 * m + 0]) * ir2
    return alm


class eblm:
    def __init__(self, alm):
        [elm, blm] = alm
        assert (len(elm) == len(blm))

        self.lmax = nlm2lmax(len(elm))

        self.elm = elm
        self.blm = blm

    def alm_copy(self, lmax=None):
        return eblm([alm_copy(self.elm, lmax=lmax),
                     alm_copy(self.blm, lmax=lmax)])

    def alm_splice(self, alm_hi, lsplit):
        return eblm([alm_splice(self.elm, alm_hi.elm, lsplit),
                     alm_splice(self.blm, alm_hi.blm, lsplit)])

    def __add__(self, other):
        assert (self.lmax == other.lmax)
        return eblm([self.elm + other.elm, self.blm + other.blm])

    def __sub__(self, other):
        assert (self.lmax == other.lmax)
        return eblm([self.elm - other.elm, self.blm - other.blm])

    def __iadd__(self, other):
        assert (self.lmax == other.lmax)
        self.elm += other.elm;
        self.blm += other.blm
        return self

    def __isub__(self, other):
        assert (self.lmax == other.lmax)
        self.elm -= other.elm
        self.blm -= other.blm
        return self

    def __mul__(self, other):
        return eblm([self.elm * other, self.blm * other])
