import numpy as np
from healpy import Alm
#FIXME: / vs // in this file ?


def alm_splice(alm_lo, alm_hi, lsplit):
    """Returns an alm w/ lmax = lmax(alm_hi) which is alm_lo for (l <= lsplit) alm_hi for (l  > lsplit.) """
    if hasattr(alm_lo, 'alm_splice'):
        return alm_lo.alm_splice(alm_hi, lsplit)

    alm_lo_lmax = Alm.getlmax(len(alm_lo))
    alm_hi_lmax = Alm.getlmax(len(alm_hi))

    assert alm_lo_lmax >= lsplit and alm_hi_lmax >= lsplit

    alm_re = np.copy(alm_hi)
    for m in range(0, lsplit + 1):
        alm_re[(m * (2 * alm_hi_lmax + 1 - m) // 2 + m):(m * (2 * alm_hi_lmax + 1 - m) // 2 + lsplit + 1)] = \
        alm_lo[(m * (2 * alm_lo_lmax + 1 - m) // 2 + m):(m * (2 * alm_lo_lmax + 1 - m) // 2 + lsplit + 1)]
    return alm_re


def alm_copy(alm, lmax=None):
    """Copies the alm array, with the option to reduce its lmax. """
    lmox = Alm.getlmax(len(alm))
    assert (lmax <= lmox)

    if (lmox == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros(Alm.getsize(lmax), dtype=np.complex)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] = \
            alm[((m * (2 * lmox + 1 - m) // 2) + m):(m * (2 * lmox + 1 - m) // 2 + lmax + 1)]
    return ret


class eblm:
    def __init__(self, alm):
        [elm, blm] = alm
        assert len(elm) == len(blm), (len(elm), len(blm))

        self.lmax = Alm.getlmax(len(elm))

        self.elm = elm
        self.blm = blm

    def alm_copy(self, lmax=None):
        return eblm([alm_copy(self.elm, lmax=lmax),
                     alm_copy(self.blm, lmax=lmax)])

    def alm_splice(self, alm_hi, lsplit):
        return eblm([alm_splice(self.elm, alm_hi.elm, lsplit),
                     alm_splice(self.blm, alm_hi.blm, lsplit)])

    def __add__(self, other):
        assert self.lmax == other.lmax
        return eblm([self.elm + other.elm, self.blm + other.blm])

    def __sub__(self, other):
        assert self.lmax == other.lmax
        return eblm([self.elm - other.elm, self.blm - other.blm])

    def __iadd__(self, other):
        assert self.lmax == other.lmax
        self.elm += other.elm
        self.blm += other.blm
        return self

    def __isub__(self, other):
        assert self.lmax == other.lmax
        self.elm -= other.elm
        self.blm -= other.blm
        return self

    def __mul__(self, other):
        return eblm([self.elm * other, self.blm * other])
