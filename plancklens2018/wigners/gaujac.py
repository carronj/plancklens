from __future__ import print_function

from scipy.special import gammaln
import numpy as np
import weave

from . import gauleg

def get_Pn(N, x, alpha, beta, norm = False):
    """ Jacobi Polynomial with parameters alpha, beta, up to order N (incl.) at points x.

    Normalized to get orthonormal Poly. The output is (N + 1, Nx) shaped.

    """
    res = _get_Pn_weave(N, x , alpha, beta)
    if not norm :
        return res
    lnnorm = gammaln(np.arange(1,N + 2,dtype = float) + alpha) + gammaln(np.arange(1,N + 2,dtype = float) + beta) \
          - gammaln(np.arange(1,N + 2,dtype = float) + alpha + beta) - gammaln(np.arange(1,N + 2,dtype = float))
    lnnorm += (alpha + beta + 1)*np.log(2.) - np.log(2 * np.arange(N + 1,dtype = float) + alpha + beta + 1)
    return res * np.outer(np.exp(-0.5 * lnnorm),np.ones(x.shape))

def _get_Pn_weave(N, x, alpha, beta):
    Pn = r"""
        double alfbet, a2, b2, n2_ab2, norm;
        
        alfbet= alpha + beta;
        a2 = alpha * alpha;
        b2 = beta * beta;
        
        for (int ix = 0; ix < nx; ix++){
            ret[ix] = 1.;
            }
        if (n > 0){
            for (int ix = 0;ix < nx; ix++) {
                ret[nx + ix] =  0.5 * (2 * (alpha + 1) +  (alfbet + 2) * (x[ix] - 1));
                }
        }
        for (int in = 1; in < n; in++) {
            n2_ab2 = 2 * in + alfbet;
            norm = 2 * (in + 1) * (in + 1 + alfbet) * n2_ab2;
            for (int ix = 0; ix < nx; ix++) { 
                 ret[(in + 1) * nx + ix] = (((n2_ab2 + 1.) * ((n2_ab2 + 2.) * n2_ab2 * x[ix] + a2 - b2)) * ret[ in  * nx + ix] - 2 * ( in + alpha) * (in + beta) * (n2_ab2 + 2) * ret[(in - 1) * nx + ix]) / norm;
            }
        }
    """
    x =  np.array(x)
    alpha = float(alpha)
    beta = float(beta)
    ret = np.empty((N + 1, len(x)), dtype=float)
    nx = int(ret.shape[1])
    n = int(N)
    weave.inline(Pn, ['x', 'ret', 'alpha', 'beta','n', 'nx'], headers = ["<stdlib.h>","<math.h>"])
    return ret

def get_wignerd(lmax, cost, mp, m):
    """ Small Wigner d matrix $d^l_{mp,m}$ for fixed mp,m, (assumed fairly small) up to lmax.

    Uses Jacobi Polynomial(a,b) representation. Integer spins only.
    a + b is  2 max(|m|,|mp|). For even spins, d^l_mp,m is a polynomial of degree l.

    """
    Pn = r"""
        double alfbet, a2, b2, n2_ab2, norm;
        double a0, ak_km1, akm1_km2;

        alfbet= a + b;
        a2 = a * a;
        b2 = b * b;

        a0 = exp(0.5 * (lgamma(2. * lmin + 1) - lgamma(a + 1.) - lgamma(2 * lmin - a + 1.)));
        ak_km1 = sqrt((1. + 2 * lmin) / (1. + a) / (1. + b)); 
        /* a1 / a0. (ak is coefficient relating Jacobi to Wigner) */

        for (int ix = 0; ix < nx; ix++){
            ret[lmin * nx + ix] = sgn * a0 *pow( (1. - x[ix]) * 0.5, 0.5 * a) * pow((1. + x[ix]) * 0.5, b * 0.5);

            }
        if (n > 0){
            for (int ix = 0;ix < nx; ix++) {
                ret[(lmin + 1) * nx + ix] = ak_km1 * ret[lmin *nx +  ix] * 0.5 * (2 * (a + 1) +  (alfbet + 2) * (x[ix] - 1));
                }
        }
        for (int in = 1; in < n; in++) {
            akm1_km2 = ak_km1;
            ak_km1 = sqrt((1. + lmin * 2. / (in + 1)) / (1. + a / (in + 1)) / (1. + b/(in+1))); 
            n2_ab2 = 2 * in + alfbet;
            norm = 2 * (in + 1) * (in + 1 + alfbet) * n2_ab2;
            for (int ix = 0; ix < nx; ix++) { 
                 ret[(lmin + in + 1) * nx + ix] = (((n2_ab2 + 1.) * ((n2_ab2 + 2.) * n2_ab2 * x[ix] + a2 - b2)) * ak_km1 * ret[ (lmin + in)  * nx + ix] - 2 * ( in + a) * (in + b) * (n2_ab2 + 2) * akm1_km2 * ak_km1 * ret[(lmin + in - 1) * nx + ix]) / norm;
            }
        }
    """
    _k = - max(abs(m), abs(mp))
    lmin = -_k
    if _k == m:
        a = mp - m
        sgn = mp - m
    elif _k == -m:
        a = m - mp
        sgn = 0
    elif _k == mp:
        a = m - mp
        sgn = 0
    else:
        a = mp - m
        sgn = mp - m
    b = -2 * _k - a
    assert a >= 0 and b >= 0
    lmax = max(lmax, lmin)
    x = np.require(cost, requirements='C')
    a = float(a)
    b = float(b)
    ret = np.zeros((lmax + 1, len(x)), dtype=float)
    nx = int(ret.shape[1])
    n = int(lmax + _k)
    sgn = (-1) ** sgn
    weave.inline(Pn, ['x', 'ret', 'a', 'b', 'n', 'nx', 'lmin', 'sgn'], headers=["<stdlib.h>", "<math.h>"])
    return ret


def get_rspace(cl, cost, mp, m):
    """ Wigner corr. fct. $\\sum_l c_l (2l + 1) / 4\\pi d^l_{m'm}(\\cos \\theta)$ from its harmonic series. """
    return np.dot( get_wignerd(len(cl) - 1,cost,mp,m).transpose(),cl * (2 * np.arange(len(cl)) + 1)/(4. * np.pi))


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