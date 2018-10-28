import numpy  as np
import healpy as hp
import util_alm

class template:
    def __init__(self):
        self.nmodes = 0
        assert 0

    def apply(self, map, coeffs):
        # map -> map*[coeffs combination of templates]
        assert 0

    def apply_mode(self, map, mode):
        assert (mode < self.nmodes)
        assert (mode >= 0)

        tcoeffs = np.zeros(self.nmodes)
        tcoeffs[mode] = 1.0
        self.apply(map, tcoeffs)

    def accum(self, map, coeffs):
        assert 0

    def dot(self, map):
        ret = []

        for i in range(0, self.nmodes):
            tmap = np.copy(map)
            self.apply_mode(tmap, i)
            ret.append(np.sum(tmap))

        return ret


class template_map(template):
    def __init__(self, map):
        self.nmodes = 1
        self.map = map

    def apply(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map *= self.map * coeffs[0]

    def accum(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map += self.map * coeffs[0]

    def dot(self, map):
        return [(self.map * map).sum()]


class template_monopole(template):
    def __init__(self):
        self.nmodes = 1

    def apply(self, map, coeffs):
        assert (len(coeffs) == self.nmodes)

        map *= coeffs[0]

    def accum(self, map, coeffs):
        map += coeffs[0]

    def dot(self, map):
        return [np.sum(map)]


class template_dipole(template):
    def __init__(self):
        self.nmodes = 3

    def apply(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)

        nside = hp.npix2nside(len(tmap))
        tmap *= hp.alm2map(xyz_to_alm(coeffs), nside, verbose=False)

    def accum(self, tmap, coeffs):
        assert (len(coeffs) == self.nmodes)

        nside = hp.npix2nside(len(tmap))
        tmap += hp.alm2map(xyz_to_alm(coeffs), nside, verbose=False)

    def dot(self, tmap):
        npix = len(tmap)
        return alm_to_xyz(hp.map2alm(tmap, lmax=1, iter=0)) * npix / 3.


def xyz_to_alm(xyz):
    assert len(xyz) == 3
    alm = np.zeros(3, dtype=np.complex)
    alm[1] = +xyz[2] * np.sqrt(4. * np.pi / 3.)
    alm[2] = (-xyz[0] + 1.j * xyz[1]) * np.sqrt(2. * np.pi / 3.)
    return alm


def alm_to_xyz(alm):
    assert len(alm) == 3
    x = -alm[2].real / np.sqrt(2. * np.pi / 3.)
    y = +alm[2].imag / np.sqrt(2. * np.pi / 3.)
    z = +alm[1].real / np.sqrt(4. * np.pi / 3.)
    return np.array([x, y, z])
