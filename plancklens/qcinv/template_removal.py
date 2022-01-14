import numpy  as np
import healpy as hp
from plancklens.qcinv.util import read_map

class template:
    def __init__(self):
        self.nmodes = 0
        assert 0

    def apply(self, m, coeffs):
        # map -> map*[coeffs combination of templates]
        assert 0

    def apply_mode(self, m, mode):
        assert (mode < self.nmodes)
        assert (mode >= 0)

        tcoeffs = np.zeros(self.nmodes)
        tcoeffs[mode] = 1.0
        self.apply(m, tcoeffs)

    def accum(self, m, coeffs):
        assert 0

    def dot(self, m):
        ret = []

        for i in range(0, self.nmodes):
            tmap = np.copy(m)
            self.apply_mode(tmap, i)
            ret.append(np.sum(tmap))

        return ret


class template_map(template):
    def __init__(self, m):
        self.nmodes = 1
        self.map = m

    def apply(self, m, coeffs):
        assert (len(coeffs) == self.nmodes)

        m *= self.map * coeffs[0]

    def accum(self, m, coeffs):
        assert (len(coeffs) == self.nmodes)

        m += self.map * coeffs[0]

    def dot(self, m):
        return [(self.map * m).sum()]


class template_qmap(template):
    def __init__(self, m):
        """Polarization Q template

        """
        self.nmodes = 1
        self.map = m

    def apply(self, pmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        if len(pmap) == 2: # Q and U maps
            pmap[0] *= read_map(self.map) * coeffs[0]
            pmap[1] *= 0.
        elif len(pmap) == 1: # Only Q
            pmap[0] *= read_map(self.map)  * coeffs[0]
        else:
            assert 0

    def accum(self, pmap, coeffs):
        assert len(pmap) == 2, len(pmap)
        assert (len(coeffs) == self.nmodes)
        pmap[0] += read_map(self.map)  * coeffs[0]

    def dot(self, pmap):
        return [np.sum(read_map(self.map)  * pmap[0])] # either Q, U or Q only


class template_umap(template):
    def __init__(self, m):
        """Polarization U template

        """
        self.nmodes = 1
        self.map = m

    def apply(self, pmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        if len(pmap) == 2: # Q and U maps
            pmap[1] *= read_map(self.map) * coeffs[0]
            pmap[0] *= 0.
        elif len(pmap) == 1: # Only U
            pmap[0] *= read_map(self.map)  * coeffs[0]
        else:
            assert 0

    def accum(self, pmap, coeffs):
        assert (len(coeffs) == self.nmodes)
        if len(pmap) == 2: # Q and U maps
            pmap[1] += read_map(self.map) * coeffs[0]
        elif len(pmap) == 1: # Assumed U-only
            pmap[0] += read_map(self.map)  * coeffs[0]
        else:
            assert 0

    def dot(self, pmap):
        if len(pmap) == 2: # Q and U maps
            return [np.sum(read_map(self.map)  * pmap[1])]
        elif len(pmap) == 1: # Assumed U-only
            return [np.sum(read_map(self.map)  * pmap[0])]
        assert 0

class template_monopole(template):
    def __init__(self):
        self.nmodes = 1

    def apply(self, m, coeffs):
        assert (len(coeffs) == self.nmodes)

        m *= coeffs[0]

    def accum(self, m, coeffs):
        m += coeffs[0]

    def dot(self, m):
        return [np.sum(m)]


class template_dipole(template):
    def __init__(self):
        self.nmodes = 3

    def apply(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)

        nside = hp.npix2nside(len(tmap))
        tmap *= hp.alm2map(xyz_to_alm(coeffs), nside)

    def accum(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)

        nside = hp.npix2nside(len(tmap))
        tmap += hp.alm2map(xyz_to_alm(coeffs), nside)

    def dot(self, tmap):
        npix = len(tmap)
        return alm_to_xyz(hp.map2alm(tmap, lmax=1, iter=0)) * npix / 3.


def xyz_to_alm(xyz):
    assert len(xyz) == 3
    alm = np.zeros(3, dtype=complex)
    alm[1] = +xyz[2] * np.sqrt(4. * np.pi / 3.)
    alm[2] = (-xyz[0] + 1.j * xyz[1]) * np.sqrt(2. * np.pi / 3.)
    return alm


def alm_to_xyz(alm):
    assert len(alm) == 3
    x = -alm[2].real / np.sqrt(2. * np.pi / 3.)
    y = +alm[2].imag / np.sqrt(2. * np.pi / 3.)
    z = +alm[1].real / np.sqrt(4. * np.pi / 3.)
    return np.array([x, y, z])
