from __future__ import print_function

from scipy.special import gammaln
import numpy as np
import weave

from . import gauleg

def get_Pn(N, x, alpha, beta, norm = False):
    """ Jacobi Polynomial with parameters alpha, beta, up to order N (incl.) at points x.

    Normalized to get orthonormal Poly. The output is (N + 1, Nx) shaped.

    """
    x =  np.array(x)
    Pn = np.ones(x.size,dtype = float)
    if N == 0 : return Pn
    res = np.zeros( (N + 1,x.size))
    Pn1 =  0.5 * (2 * (alpha + 1) +  (alpha + beta + 2) * (x - 1))
    res[0,:] = Pn
    res[1,:] = Pn1
    if N == 1 : return res
    alfbet = alpha + beta
    for I in range(1,N) :
        n2_ab2 = 2 * I  + alfbet  # 2(I + 1) + a + b - 2
        a = 2 * (I + 1) * (I + 1 + alfbet) * n2_ab2
        res[I + 1,:] = ( ((n2_ab2 + 1)* ( (n2_ab2 + 2) * (n2_ab2)*x + alpha ** 2 - beta ** 2 ))*res[I,:]
                         - 2 * (I + alpha)*(I + beta) * (n2_ab2 + 2)*res[I-1,:])/a
    if not norm :
        return res
    lnnorm = gammaln(np.arange(1,N + 2,dtype = float) + alpha) + gammaln(np.arange(1,N + 2,dtype = float) + beta) \
          - gammaln(np.arange(1,N + 2,dtype = float) + alpha + beta) - gammaln(np.arange(1,N + 2,dtype = float))
    lnnorm += (alpha + beta + 1)*np.log(2.) - np.log(2 * np.arange(N + 1,dtype = float) + alpha + beta + 1)
    return res * np.outer(np.exp(-0.5 * lnnorm),np.ones(x.shape))

def get_rspace(cl, cost, mp, m):
    """ Wigner corr. fct. $\\sum_l c_l (2l + 1) / 4\\pi d^l_{m'm}(\\cos \\theta)$ from its harmonic series. """
    return np.dot( get_wignerd(len(cl) - 1,cost,mp,m).transpose(),cl * (2 * np.arange(len(cl)) + 1)/(4. * np.pi))

def get_wignerd(lmax, cost, mp, m):
    """ Small Wigner d matrix $d^l_{mp,m}$ for fixed mp,m, (assumed fairly small) up to lmax.

    Uses Jacobi Polynomial(a,b) representation. Integer spins only.
    a + b is  2 max(|m|,|mp|). For even spins, d^l_mp,m is a polynomial of degree l.

    """
    _k = - max(abs(m),abs(mp))
    lmin = -_k
    if _k == m:
        a = mp -m
        sgn = mp - m
    elif _k == -m:
        a = m- mp
        sgn = 0
    elif _k == mp:
        a = m - mp
        sgn = 0
    else:
        a = mp -m
        sgn = mp - m
    b = -2 * _k - a
    assert a >=0 and b >= 0
    lmax = max(lmax,lmin)

    k = np.arange(lmin,lmax + 1,dtype = int) + _k
    j = np.arange(lmin,lmax + 1,dtype = int)
    lnfacl = gammaln(2 * j - k + 1) - gammaln(k + b + 1)
    lnfacl -= gammaln(k + a + 1) - gammaln(b + 1)
    lnfacl -= gammaln(2 * j - 2 * k - a + 1) - gammaln(k + 1)
    sinb2cosb2 =  np.sqrt( (1. - cost) / 2.) ** a * np.sqrt( (1. + cost) / 2.) ** b
    ret = np.zeros((lmax + 1,cost.size))
    ret[lmin:] = get_Pn(lmax + _k,cost,a,b) * np.outer(np.exp(0.5 * lnfacl) * (-1) ** sgn,sinb2cosb2)
    return ret


def wig2leg(cl, sp, s):
    """ Converts a polarized wigner fct to its Legendre coefficient series. """
    assert sp % 2 == 0 and s % 2 == 0,'Have not checked exactness of G-L integrals in this case'
    lmax = len(cl) - 1
    xg,wg = gauleg.get_xgwg(lmax + 1) # Have to integration pol of order 2 * lmax
    Pn = gauleg.get_Pn(lmax,xg)
    return np.dot(Pn,get_rspace(cl,xg,sp,s) * wg * (2. * np.pi))


def _test_wig():
    x = np.linspace(-1,1,100)
    mps = [1,1,1,0]
    ms = [1,0,-1,0]
    ls = [1,1,1,1]
    sols = [0.5* (1. + x),-np.sqrt(1. - x ** 2)/np.sqrt(2.),0.5 * (1- x),x]
    for mp,m,l,sol in zip(mps,ms,ls,sols):
        print(l, mp, m, np.allclose(get_wignerd(l,x,mp,m)[l],sol))

    mps = [2,2,2,2,2,1,1,1,0]
    ms = [2,1,0,-1,-2,1,0,-1,0]
    ls = [2] * 9
    si = np.sqrt(1. - x ** 2)
    sols = [0.25 * (1 + x) ** 2,-0.5 * si * (1. + x),np.sqrt(3./8.) * si ** 2,
            -0.5 * si * (1. - x),0.25 * (1. - x)**2,0.5 * (2 * x ** 2 + x -1.),
            -np.sqrt(3./8.)* 2 * x *si,0.5 * (-2 * x ** 2 + x + 1), 0.5 * (3 * x ** 2 -1.)]
    for mp,m,l,sol in zip(mps,ms,ls,sols):
        print(l, mp, m, np.allclose(get_wignerd(l,x,mp,m)[l],sol))