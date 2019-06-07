"""Module with spin-weight related utilities.

    Conventions are $_{\pm |s|} X_{lm} = - (\pm)^{|s|} (G_{lm} \pm i  C_{lm})$.

    For CMB maps,

    $ _{0}X_{lm} = T_{lm} $
    $ _{\pm}X_{lm} = -1/2 (E_{lm} \pm i B_{lm}) $

    hence

    $ G^{0}_{lm} = -T_{lm} $
    $ G^{2}_{lm} =  E_{lm} $
    $ C^{2}_{lm} =  B_{lm} $.

"""

import healpy as hp
import numpy as np

def alm2map_spin(gclm, nside, spin, lmax, mmax=None):
    assert spin >= 0, spin
    assert len(gclm) == 2, len(gclm)
    if spin > 0:
        return hp.alm2map_spin(gclm, nside, spin, lmax, mmax=mmax)
    elif spin == 0:
        return hp.alm2map(-gclm[0], nside, lmax=lmax, mmax=mmax), 0.

def map2alm_spin(maps, spin, lmax=None, mmax=None):
    assert spin >= 0, spin
    if spin > 0:
        return hp.map2alm_spin(maps, spin, lmax=lmax, mmax=mmax)
    else:
        return -hp.map2alm(maps[0], lmax=lmax, mmax=mmax, iter=0), 0.

try:
    from plancklens2018.wigners import wigners  # fortran 90 shared object
    HASWIGNER = True
except:
    print("wigners.so fortran shared object not found")
    print('try f2py -c -m wigners wigners.f90 from the command line in wigners directory')
    print("Falling back on python2 weave implementation")
    HASWIGNER = False
    from plancklens2018.wigners import gaujac, gauleg

GL_cache = {}

def wignerc(cl1, cl2, sp1, s1, sp2, s2, lmax_out=None):
    """Legendre coeff. of $ (\\xi_{sp1,s1} * \\xi_{sp2,s2})(\\cos \\theta)$ from their harmonic series.

        This uses Gauss-Legendre quadrature to solve this exactly.

    """
    lmax1 = len(cl1) - 1
    lmax2 = len(cl2) - 1
    lmax_out = lmax1 + lmax2 if lmax_out is None else lmax_out
    lmaxtot = lmax1 + lmax2 + lmax_out
    N = (lmaxtot + 2 - lmaxtot % 2) // 2
    if not 'xg wg %s' % N in GL_cache.keys():
        GL_cache['xg wg %s' % N] = wigners.get_xgwg(-1., 1., N) if HASWIGNER else gauleg.get_xgwg(N)
    xg, wg = GL_cache['xg wg %s' % N]

    if HASWIGNER:
        if np.iscomplexobj(cl1):
            xi1 = wigners.wignerpos(np.real(cl1), xg, sp1, s1) + 1j * wigners.wignerpos(np.imag(cl1), xg, sp1, s1)
        else:
            xi1 = wigners.wignerpos(cl1, xg, sp1, s1)
        if np.iscomplexobj(cl2):
            xi2 = wigners.wignerpos(np.real(cl2), xg, sp2, s2) + 1j * wigners.wignerpos(np.imag(cl2), xg, sp2, s2)
        else:
            xi2 = wigners.wignerpos(cl2, xg, sp2, s2)
        xi1xi2w = xi1 * xi2 * wg
        if np.iscomplexobj(xi1xi2w):
            ret = wigners.wignercoeff(np.real(xi1xi2w), xg, sp1 + sp2, s1 + s2, lmax_out)
            ret = ret + 1j * wigners.wignercoeff(np.imag(xi1xi2w), xg, sp1 + sp2, s1 + s2, lmax_out)
            return ret
        else:
            return wigners.wignercoeff(xi1xi2w, xg, sp1 + sp2, s1 + s2, lmax_out)
    else:
        xi1 = gaujac.get_rspace(cl1, xg, sp1, s1)
        xi2 = gaujac.get_rspace(cl2, xg, sp2, s2)
        return 2. * np.pi * np.dot(gaujac.get_wignerd(lmax_out, xg, sp1 + sp2, s1 + s2), wg * xi1 * xi2)


def get_spin_raise(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin raising operator.

        $+\sqrt{ (l - s) (l + s + 1) }$ for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin lowering operator.

        $-\sqrt{ (l + s) (l - s + 1) }$ for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret

def get_spin_coupling(s1, s2, cls):
    """Spin-weighted power spectrum $<_{s1}X_{lm} _{s2}X^{*}_{lm}>$

    Note:
        The output is real unless necessary.

    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(get_spin_coupling(-s1, -s2, cls))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, 'not implemented')
    if s1 == 0:
        if s2 == 0:
            return cls['tt']
        tb = cls.get('tb', None)
        return  -cls['te'] if tb is None else  -cls['te'] + 1j * np.sign(s2) * tb
    elif s1 == 2:
        if s2 == 0:
            tb = cls.get('tb', None)
            return  -cls['te'] if tb is None else  -cls['te'] - 1j * tb
        elif s2 == 2:
            return cls['ee'] + cls['bb']
        elif s2 == -2:
            eb = cls.get('eb', None)
            return  cls['ee'] - cls['bb'] if eb is None else  cls['ee'] - cls['bb'] + 2j * eb
        else:
            assert 0

def get_spin_matrix(sout, sin, cls):
    """Spin-space matrix R^{-1} cls[T, E, B] R where R is the mapping from _{0, \pm 2}X to T, E, B.

        cls is dictionary with keys 'tt', 'te', 'ee', 'bb'.
        If a key is not present the corresponding spectrum is assumed to be zero.
        ('t' 'e' and 'b' keys also works in place of 'tt' 'ee', 'bb'.)

        Output is complex only when necessary (that is, TB and/or EB present and relevant).

    """
    assert sin in [0, 2, -2] and sout in [0, 2, -2], (sin, sout)
    if sin == 0:
        if sout == 0:
            return cls.get('tt', cls.get('t', 0.))
        tb = cls.get('tb', None)
        return (-cls.get('te', 0.) - 1j * np.sign(sout) * tb) if tb is not None else -cls.get('te', 0.)
    if sin == 2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return -0.5 * (te - 1j * tb) if tb is not None else -0.5 * te
        if sout == 2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
        if sout == -2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return ret - 1j * eb if eb is not None else ret
    if sin == -2:
        if sout == 0:
            te = cls.get('te', 0.)
            tb = cls.get('tb', None)
            return -0.5 * (te + 1j * tb) if tb is not None else -0.5 * te
        if sout == 2:
            ret =  0.5 * (cls.get('ee', cls.get('e', 0.)) - cls.get('bb', cls.get('b', 0.)))
            eb = cls.get('eb', None)
            return ret + 1j * eb if eb is not None else ret
        if sout == -2:
            return 0.5 * (cls.get('ee', cls.get('e', 0.)) + cls.get('bb', cls.get('b', 0.)))
    assert 0, (sin, sout)

