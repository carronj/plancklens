from __future__ import print_function

from scipy.special import gammaln
import numpy as np
import weave

from . import gauleg



def get_xgwg(n, alpha, beta, MAXIT=10):
    """ Gauss-Jacobi quadrature points and weights.

    (C) Copr. 1986 - 92 Numerical Recipes Software ?421.1 - 9.

    """
    header = ["<stdio.h>","<math.h>"]
    support_code = "#define EPS 3.0e-14 /* EPS is the relative precision. */"
    gaujac = """
	int i,its,j;
	float alfbet,an,bn,r1,r2,r3;
	double a,b,c,p1,p2,p3,pp,temp,z,z1;

	for (i=0;i<n;i++) {
		if (i == 0) {
			an=alf/n;
			bn=bet/n;
			r1=(1.0+alf)*(2.78/(4.0+n*n)+0.768*an/n);
			r2=1.0+1.48*an+0.96*bn+0.452*an*an+0.83*an*bn;
			z=1.0-r1/r2;
		} else if (i == 1) {
			r1=(4.1+alf)/((1.0+alf)*(1.0+0.156*alf));
			r2=1.0+0.06*(n-8.0)*(1.0+0.12*alf)/n;
			r3=1.0+0.012*bet*(1.0+0.25*fabs(alf))/n;
			z -= (1.0-z)*r1*r2*r3;
		} else if (i == 2) {
			r1=(1.67+0.28*alf)/(1.0+0.37*alf);
			r2=1.0+0.22*(n-8.0)/n;
			r3=1.0+8.0*bet/((6.28+bet)*n*n);
			z -= (x[0]-z)*r1*r2*r3;
		} else if (i == n-2) {
			r1=(1.0+0.235*bet)/(0.766+0.119*bet);
			r2=1.0/(1.0+0.639*(n-4.0)/(1.0+0.71*(n-4.0)));
			r3=1.0/(1.0+20.0*alf/((7.5+alf)*n*n));
			z += (z-x[n-4])*r1*r2*r3;
		} else if (i == n-1) {
			r1=(1.0+0.37*bet)/(1.67+0.28*bet);
			r2=1.0/(1.0+0.22*(n-8.0)/n);
			r3=1.0/(1.0+8.0*alf/((6.28+alf)*n*n));
			z += (z-x[n-3])*r1*r2*r3;
		} else {
			z=3.0*x[i-1]-3.0*x[i-2]+x[i-3];
		}
		alfbet=alf+bet;
		for (its=1;its<=MAXIT;its++) {
			temp=2.0+alfbet;
			p1=(alf-bet+temp*z)/2.0;
			p2=1.0;
			for (j=2;j<=n;j++) {
				p3=p2;
				p2=p1;
				temp=2*j+alfbet;
				a=2*j*(j+alfbet)*(temp-2.0);
				b=(temp-1.0)*(alf*alf-bet*bet+temp*(temp-2.0)*z);
				c=2.0*(j-1+alf)*(j-1+bet)*temp;
				p1=(b*p2-c*p3)/a;
			}
			pp=(n*(alf-bet-temp*z)*p1+2.0*(n+alf)*(n+bet)*p2)/(temp*(1.0-z*z));
			z1=z;
			z=z1-p1/pp;
			if (fabs(z-z1) <= EPS) break;
		}
		if (its > MAXIT) printf("too many iterations in gaujac");
		x[i]=z;
		w[i]=exp(lgamma(alf+n)+lgamma(bet+n)-lgamma(n+1.0)-
			lgamma(n+alfbet+1.0))*temp*pow(2.0,alfbet)/(pp*p2);
	}"""
    n = int(n)
    MAXIT = int(MAXIT)
    x = np.empty(n,dtype = float)
    w = np.empty(n,dtype = float)
    alf = float(alpha)
    bet = float(beta)
    weave.inline(gaujac, ['n','x','w','alf','bet','MAXIT'], headers=header,support_code=support_code)
    return x,w


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