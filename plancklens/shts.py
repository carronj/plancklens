import os
import numpy as np

try:
    import lenspyx
    from lenspyx import utils_hp
    from lenspyx.remapping import utils_geom
    HASLENSPYX = True
    print('Using lenspyx alm2map')
    nthreads = int(os.environ.get('OMP_NUM_THREADS', 0))

    def alm2map(alm, nside):
        geom = utils_geom.Geom.get_healpix_geometry(nside)
        lmax = utils_hp.Alm.getlmax(alm.size, None)
        return geom.alm2map(gclm=alm, lmax=lmax, mmax=lmax, nthreads=nthreads).squeeze()
    def map2alm(m, lmax, **kwargs):
        nside = int(np.round(np.sqrt(m.size // 12)))
        assert 12 *  nside ** 2 == m.size, (m.size, 12 * nside ** 2)
        geom = utils_geom.Geom.get_healpix_geometry(nside)
        return geom.map2alm(m=m, lmax=lmax, mmax=lmax, nthreads=nthreads).squeeze()

    def alm2map_spin(gclm, nside, spin, lmax):
        geom = utils_geom.Geom.get_healpix_geometry(nside)
        return geom.alm2map_spin(gclm=gclm, spin=spin, lmax=lmax, mmax=lmax, nthreads=nthreads)

    def map2alm_spin(qumap, spin, lmax):
        nside = int(np.round(np.sqrt(qumap[0].size // 12)))
        assert 12 *  nside ** 2 == qumap[1].size, (qumap.size // 2, 12 * nside ** 2)
        geom = utils_geom.Geom.get_healpix_geometry(nside)
        return geom.map2alm_spin(m=qumap, spin=spin, lmax=lmax, mmax=lmax, nthreads=nthreads)


except ImportError:
    HASLENSPYX = False
    from healpy import alm2map, map2alm, alm2map_spin, map2alm_spin
    