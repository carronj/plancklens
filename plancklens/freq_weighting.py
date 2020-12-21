"""This modules contains utilities related to weighting of frequency maps


"""
import healpy as hp
import numpy as np

from plancklens.qcinv.util import read_map
from plancklens.wigners import wigners
from plancklens.utils import enumerate_progress


def _w2wsq(wl, s1, s2, lmax_out):
    """Returns Legendre coefficient of squared Wigner correlation function

    """
    lmax = len(wl) - 1
    npts = (2 * lmax + lmax_out) // 2 + 1
    xg, wg = wigners.get_xgwg(-1., 1., npts)
    return wigners.wignercoeff(wigners.wignerpos(wl, xg, s1, s2) ** 2 * wg, xg, 0, 0, lmax_out)

def vmaps2vmap_I(pix_vmaps, weights, nside):
    """From individual freq pixel variance maps and weights create expected pixel variance map


       Args:
            pix_vmaps: list of pixel variance maps
            weights: weights for intensity freq. weighting (as applied onto the noise maps)
            nside: desired output map resolution

       See Planck 2018 gravitational lensing paper Eqs 16-17

    """
    assert len(pix_vmaps) == len(weights), (len(pix_vmaps), len(weights))
    nf, lmaxp1 = weights.shape
    lmax_out = min(2 * lmaxp1 - 2, 3 * nside - 1)
    ret_lm = np.zeros(hp.Alm.getsize(lmax_out), dtype=complex)
    for i, (pix_vmap, wl) in enumerate_progress(list(zip(pix_vmaps, weights))):
        m = read_map(pix_vmap)
        vpix = hp.nside2pixarea(hp.npix2nside(m.size), degrees=False)
        this_s2lm = hp.map2alm(m, iter=0, lmax=lmax_out)
        wl2 = _w2wsq(wl, 0, 0, lmax_out)  * vpix
        hp.almxfl(this_s2lm, wl2, inplace=True)
        ret_lm += this_s2lm
    return hp.alm2map(ret_lm, nside, verbose=False)

def vmaps2vmap_P(pix_vmaps, weights_e, weights_b, nside):
    """From individual Q and U freq pixel variance maps and weights create expected pixel variance map

        Args:
            pix_vmaps: list of pixel variance maps
            weights_e: weights for E-mode freq. weighting (as applied onto the noise maps)
            weights_b: weights for B-mode freq. weighting (as applied onto the noise maps)
            nside: desired output map resolution

       Note:
           the pix_vmaps in pol in this routine are expected to be ~ 1/2 (s2_Q + s2_U)

           See Planck 2018 gravitational lensing paper Eqs 16-17


    """
    assert len(pix_vmaps) == len(weights_e), (len(pix_vmaps), len(weights_e))
    assert len(pix_vmaps) == len(weights_b), (len(pix_vmaps), len(weights_b))

    nf, lmaxp1_e = weights_e.shape
    nf, lmaxp1_b = weights_b.shape

    lmax_out = min(2 * max(lmaxp1_e, lmaxp1_b) - 2, 3 * nside - 1)
    ret_lm = np.zeros(hp.Alm.getsize(lmax_out), dtype=complex)
    for i, (pix_vmap, wle, wlb) in enumerate_progress(list(zip(pix_vmaps, weights_e, weights_b))):
        m = read_map(pix_vmap)
        vpix = hp.nside2pixarea(hp.npix2nside(m.size), degrees=False)
        this_s2lm = hp.map2alm(m, iter=0, lmax=lmax_out)
        wl2  = 0.25 * vpix * _w2wsq(wle + wlb, 2,  2, lmax_out)
        wl2 += 0.25 * vpix * _w2wsq(wle - wlb, 2, -2, lmax_out)
        hp.almxfl(this_s2lm, wl2, inplace=True)
        ret_lm += this_s2lm
    return hp.alm2map(ret_lm, nside, verbose=False)