"""Planck 2018 lensing utils module

Collects a couple of misc. utility functions.

"""
from __future__ import print_function
from __future__ import division

import os


import healpy as hp
from healpy.projector import CartesianProj
import time
import numpy as np
import sys
import hashlib

def alm_copy(alm, lmax=None):
    """Copies the healpy alm array, with the option to reduce its lmax

    Args:
        alm (ndarray): healpy alm array.
        lmax (int, optional): new alm lmax.
    """
    alm_lmax = int(np.floor(np.sqrt(2 * len(alm)) - 1))
    assert lmax <= alm_lmax, (lmax, alm_lmax)
    if (alm_lmax == lmax) or (lmax is None):
        ret = np.copy(alm)
    else:
        ret = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=complex)
        for m in range(0, lmax + 1):
            ret[((m * (2 * lmax + 1 - m) // 2) + m):(m * (2 * lmax + 1 - m) // 2 + lmax + 1)] \
                = alm[(m * (2 * alm_lmax + 1 - m) // 2 + m):(m * (2 * alm_lmax + 1 - m) // 2 + lmax + 1)]
    return ret

def alm2rlm(alm):
    """ converts a complex alm to 'real harmonic' coefficients rlm. """

    lmax = hp.Alm.getlmax(alm.size)
    rlm = np.zeros((lmax + 1) ** 2, dtype=float)

    ls = np.arange(0, lmax + 1, dtype=int)
    l2s = ls ** 2
    rt2 = np.sqrt(2.)

    rlm[l2s] = alm[ls].real
    for m in range(1, lmax + 1):
        rlm[l2s[m:] + 2 * m - 1] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].real * rt2
        rlm[l2s[m:] + 2 * m + 0] = alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]].imag * rt2
    return rlm


def rlm2alm(rlm):
    """ converts 'real harmonic' coefficients rlm to complex alm. """

    lmax = int(np.sqrt(len(rlm)) - 1)
    assert ((lmax + 1) ** 2 == len(rlm))

    alm = np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=complex)

    ls = np.arange(0, lmax + 1, dtype=int)
    l2s = ls ** 2
    ir2 = 1.0 / np.sqrt(2.)

    alm[ls] = rlm[l2s]
    for m in range(1, lmax + 1):
        alm[m * (2 * lmax + 1 - m) // 2 + ls[m:]] = (rlm[l2s[m:] + 2 * m - 1] + 1j * rlm[l2s[m:] + 2 * m + 0]) * ir2
    return alm


def projectmap(hpmap, lcell_amin, Npts, lon_lat= (0., -45. )):
    """Projects portion of healpix map onto square map.

    Args:
        hpmap (ndarray): healpy map
        lcell_amin (float): desired pixel cell size in arcmin.
        Npts (int): desired number of pixel per side
        lon_lat (tuple, optional): longitude and latitude of center of projection in deg. Defaults to 0 and -45.

    Returns :
        projected map and projector for future calls.

    """
    lon, lat = lon_lat
    assert 0. <= lon <= 360. and -90. <= lat <= 90., (lon,lat)
    _lon = lon if lon <= 180 else lon - 360
    lonra = [-lcell_amin * Npts / 60. / 2., lcell_amin / 60 * Npts / 2.]
    latra = [-lcell_amin * Npts / 60  / 2., lcell_amin / 60 * Npts / 2.]
    P = CartesianProj(rot = [_lon,lat,0.],lonra=lonra, latra=latra, xsize=Npts, ysize=Npts)
    P.set_flip('astro')
    return P.projmap(hpmap, lambda x, y, z: hp.vec2pix(hp.npix2nside(len(hpmap)), x, y, z)), P

def enumerate_progress(list, label=''):
    """Simple progress bar.

    """
    t0 = time.time()
    ni = len(list)
    for i, v in enumerate(list):
        yield i, v
        ppct = int(100. * (i - 1) / ni)
        cpct = int(100. * (i + 0) / ni)
        if cpct > ppct:
            dt = time.time() - t0
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            sys.stdout.write("\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " +
                             label + " " + int(10. * cpct / 100) * "-" + "> " + ("%02d" % cpct) + r"%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def clhash(cl, dtype=np.float16):
    """Hash for generic numpy array.

    By default we avoid here double precision checks since this might be machine dependent.

        Note: casting to low precision can be a really bad choice for small numbers...


    """
    return hashlib.sha1(np.copy(cl.astype(dtype), order='C')).hexdigest()

def mchash(cl):
    """Hash for integer (e.g. sim indices) array where order does not matter.

    """
    return hashlib.sha1(np.copy(np.sort(cl), order='C')).hexdigest()

def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def joincls(cls_list):
    lmaxp1 = np.min([len(cl) for cl in cls_list])
    return np.prod(np.array([cl[:lmaxp1] for cl in cls_list]), axis=0)

def hash_check(hash1, hash2, ignore=['lib_dir', 'prefix'], keychain=[], fn=None):
    keys1 = hash1.keys()
    keys2 = hash2.keys()
    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)
    for key in set(keys1).union(set(keys2)):
        # v1 = hash1[key]
        # v2 = hash2[key]
        try:
            v1 = hash1[key]
            v2 = hash2[key]
        except KeyError:
            raise KeyError(f"Cannot find key {key} in hashdict {fn}")
        
        def hashfail(msg=None):
            print(f"CHECKING HASHFILE {fn}")
            print(f"ERROR: HASHCHECK FAIL AT KEY {key}")
            if msg is not None:
                print("   " + msg)
            print("   ", "V1 = ", v1)
            print("   ", "V2 = ", v2)
            print(keys1)
            print(keys2)

            assert 0

        if type(v1) != type(v2):
            hashfail(f'UNEQUAL TYPES: type(v1) = {type(v1)}, type(v2)={type(v2)}')
        elif type(v2) == dict:
            hash_check( v1, v2, ignore=ignore, keychain=keychain + [key], fn=fn )
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not( v1 == v2 ):
                hashfail('UNEQUAL VALUES')
class stats:
    """Simple minded library for means and covariances from sims.

    """

    def __init__(self, size, xcoord=None, docov=True):
        self.N = 0  # number of samples
        self.size = size  # dim of data vector
        self.sum = np.zeros(self.size)  # sum_i x_i
        if docov: self.mom = np.zeros((self.size, self.size))  # sum_i x_ix_i^t
        self.xcoord = xcoord
        self.docov = docov

    def add(self, v):
        assert (v.shape == (self.size,)), "input not understood"
        self.sum += v
        if self.docov:
            self.mom += np.outer(v, v)
        self.N += 1

    def mean(self):
        assert (self.N > 0)
        return self.sum / float(self.N)

    def avg(self):
        return self.mean()

    def cov(self):
        assert self.docov
        assert (self.N > 0)
        if self.N == 1: return np.zeros((self.size, self.size))
        mean = self.mean()
        return self.mom / (self.N - 1.) - self.N / (self.N - 1.) * np.outer(mean, mean)

    def sigmas(self):
        return np.sqrt(np.diagonal(self.cov()))

    def corrcoeffs(self):
        sigmas = self.sigmas()
        return self.cov() / np.outer(sigmas, sigmas)

    def sigmas_on_mean(self):
        assert self.N > 0
        return self.sigmas() / np.sqrt(self.N)

    def inverse(self, bias_p=None):  # inverse cov, using unbiasing a factor following G. statistics
        assert self.N > self.size, "Non invertible cov.matrix"
        if bias_p is None: bias_p = (self.N - self.size - 2.) / (self.N - 1)
        return bias_p * np.linalg.inv(self.cov())

    def get_chisq(self, data):  # Returns (data -mean)Sig^{-1}(data-mean)
        assert data.size == self.size, (data.size, self.size)
        dx = data - self.mean()
        return np.sum(np.outer(dx, dx) * self.inverse())

    def get_chisq_pte(self, data):  # probability to exceed, or survival function
        from scipy.stats import chi2
        return chi2.sf(self.get_chisq(data), self.N - 1)  # 'survival function' of chisq distribution with N -1 dof

    def rebin_that_nooverlap(self, orig_coord, lmins, lmaxs, weights=None):
        # Returns a new stat instance rebinning with non-overlapping weights
        # >= a gauche, <= a droite.
        assert orig_coord.size == self.size, "Incompatible input"
        assert lmins.size == lmaxs.size, "Incompatible input"
        assert np.all(np.diff(np.array(lmins)) > 0.), "This only for non overlapping bins."
        assert np.all(np.diff(np.array(lmaxs)) > 0.), "This only for non overlapping bins."
        assert np.all(lmaxs - lmins) > 0., "This only for non overlapping bins."

        if weights is None: weights = np.ones(self.size)
        assert weights.size == self.size, "incompatible input"
        newsize = len(lmaxs)
        assert self.size > newsize, "Incompatible dimensions"
        Tmat = np.zeros((newsize, self.size))
        newsum = np.zeros(newsize)
        for k, lmin, lmax in zip(np.arange(newsize), lmins, lmaxs):
            idc = np.where((orig_coord >= lmin) & (orig_coord <= lmax))
            if len(idc) > 0:
                norm = np.sum(weights[idc])
                Tmat[k, idc] = weights[idc] / norm
                newsum[k] = np.sum(weights[idc] * self.sum[idc]) / norm

        newmom = np.dot(np.dot(Tmat, self.mom), Tmat.transpose())  # New mom. matrix is T M T^T
        newstats = stats(newsize, xcoord=0.5 * (lmins[0:len(lmins) - 1] + lmaxs[1:]))

        newstats.mom = newmom
        newstats.sum = newsum
        newstats.N = self.N
        return newstats

def apodize_mask(mask, sigma_arcmin=12., lmax=None, method='hybrid', cache_dir='caches/',
                 mult_factor=3, min_factor=0.1):
    """Apodize a mask so it can safely be used for Pseudo-CL inversion.

    Args:
        mask: input healpix map array
        sigma_arcmin: characteritic width of smoothing
        lmax: lmax when apodizing mask
        method: gaussian or hybrid (hybrid mainly smooths outside existing mask, so reduces fsky)
        cache_dir: if not None, cache result here if possible
        mult_factor: for hybrid method, multiply (1-mask) by this factor and truncate (enlarges mask, before resmoothing)
        min_factor: for hybrid method, set to unity tails larger than 1-min_factor after scaling by mult_factor
    Returns:
        the apodized map array
    """

    if not sigma_arcmin: return mask
    sigma_rad = sigma_arcmin / 180. / 60. * np.pi
    if cache_dir: name = os.path.join(cache_dir, 'ap_mask_' + '_'.join(
        '%s' % s for s in
        [sigma_arcmin, method, lmax, mult_factor, min_factor, hashlib.sha1(mask).hexdigest()])) + '.fits'
    if cache_dir and os.path.exists(name):
        ap_mask = hp.read_map(name)
    else:
        print('apodizing... (fsky_unapodized=%s)' % (np.sum(mask ** 2) / mask.size))
        ap_mask = hp.smoothing(mask, sigma=sigma_rad, lmax=lmax)
        print('Min/max mask smoothed mask', np.min(ap_mask), np.max(ap_mask))
        print('fsky=', np.sum(ap_mask ** 2) / ap_mask.size)
        if method == 'gaussian': return ap_mask
        if method != 'hybrid': raise ValueError('Unknown apodization method')
        ap_mask = 1 - np.minimum(1., np.maximum(0., mult_factor * (1 - ap_mask) - min_factor))
        ap_mask = hp.smoothing(ap_mask, sigma=sigma_rad / 2, lmax=lmax)
        print('Min/max mask re-smoothed mask', np.min(ap_mask), np.max(ap_mask))
        print('fsky=', np.sum(ap_mask ** 2) / ap_mask.size)
        if cache_dir:
            hp.write_map(name, ap_mask)
    return ap_mask

def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.

    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.

    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    if len(cols) > 5:
        wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
        wptpe = lambda ell : np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi)
        for i, k in enumerate(['pp', 'pt', 'pe']):
            cls[k] = np.zeros(lmax + 1, dtype=float)
        cls['pp'][ell[idc]] = cols[5][idc] / wpp(ell[idc])
        cls['pt'][ell[idc]] = cols[6][idc] / wptpe(ell[idc])
        cls['pe'][ell[idc]] = cols[7][idc] / wptpe(ell[idc])
    return cls


def cl_inverse(cls):
    """Inverse of T E B spectral matrices. Input and ouputs are dictionaries.

    """
    def extend_cl(cl, lmax):
        ret  =  np.zeros(lmax + 1, dtype=float)
        ret[:min(len(cl), lmax + 1)] = np.copy(cl[:min(len(cl), lmax + 1)])
        return ret
    lmax = np.max([len(cl) for cl in cls.values()]) - 1
    clsm = np.zeros((lmax + 1, 3, 3))
    clsm[:, 0, 0] = extend_cl(cls.get('tt', [0.]), lmax)
    clsm[:, 1, 1] = extend_cl(cls.get('ee', [0.]), lmax)
    clsm[:, 2, 2] = extend_cl(cls.get('bb', [0.]), lmax)
    clsm[:, 0, 1] = extend_cl(cls.get('te', [0.]), lmax)
    clsm[:, 1, 0] = extend_cl(cls.get('te', [0.]), lmax)
    clsm[:, 0, 2] = extend_cl(cls.get('tb', [0.]), lmax)
    clsm[:, 2, 0] = extend_cl(cls.get('tb', [0.]), lmax)
    clsm[:, 1, 2] = extend_cl(cls.get('eb', [0.]), lmax)
    clsm[:, 2, 1] = extend_cl(cls.get('eb', [0.]), lmax)

    if np.__version__ >= '1.14':
        clsmi = np.linalg.pinv(clsm) # This may require numpy version > 1.14
    else:
        clsmi = np.array([np.linalg.pinv(clsm[l]) for l in range(clsm.shape[0])])
    clsi ={}
    for k, (i, j) in zip(['tt', 'ee', 'bb', 'te', 'tb', 'eb'], [[0, 0],[1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]):
        arr = clsmi[:, i, j].copy()
        if np.any(arr):
            clsi[k] = arr
    return clsi

def extcl(lmax, cl):
    if len(cl) - 1 < lmax:
        dl = np.zeros(lmax + 1)
        dl[:len(cl)] = cl
    else:
        dl = cl[:lmax+1]
    return dl

def _cldict2arr(cls_dict):
    lmaxp1 = np.max([len(cl) for cl in cls_dict.values()])
    ret = np.zeros((3, 3, lmaxp1), dtype=float)
    for i, x in enumerate(['t', 'e', 'b']):
        for j, y in enumerate(['t', 'e', 'b']):
            ret[i, j] =  extcl(lmaxp1 - 1, cls_dict.get(x + y, cls_dict.get(y + x, np.array([0.]))))
    return ret

def cls_dot(cls_list, ret_dict=False):
    """T E B spectral matrices product

        Args:
            list of dict cls spectral matrices to multiply (given as dictionaries or (3, 3, lmax + 1) arrays

        Returns:
            (3, 3, lmax + 1) array where 0, 1, 2 stands for T E B


    """
    if  len(cls_list) == 1:
        return _cldict2arr(cls_list[0]) if isinstance(cls_list[0], dict) else cls_list[0]
    cls = cls_dot(cls_list[1:])
    cls_0 =  _cldict2arr(cls_list[0]) if isinstance(cls_list[0], dict) else cls_list[0]
    ret = np.zeros_like(cls_0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ret[i, j] += cls_0[i, k] * cls[k, j]
    if ret_dict:
        clsi = {}
        for k, (i, j) in zip(['tt', 'ee', 'bb', 'te', 'tb', 'eb'], [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]):
            arr = ret[i, j, :].copy()
            if np.any(arr):
                clsi[k] = arr
        return clsi
    return ret