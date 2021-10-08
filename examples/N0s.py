"""Plot defaults N0 curves


"""
import os, numpy as np
from plancklens import utils, n0s
import plancklens

if __name__ == '__main__':
    # gets N0 and plot them
    import pylab as pl

    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')
    cls_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
    wls = lambda L : L ** 2 * (L  + 1) ** 2 * 1e7 / (2 * np.pi)
    ls = np.arange(1, 3001)
    ls_curl = np.arange(2, 3001)

    N0s, N0_curls = n0s.get_N0() # Check this out for options
    for qe_key in N0s.keys():
        label = {'tt':'TT', '_p':'PP', '':'MV'}[qe_key[1:]]
        ln = pl.loglog(ls, wls(ls) * N0s[qe_key][ls], label=label)
        pl.loglog(ls_curl, wls(ls_curl) * N0_curls[qe_key][ls_curl], label=label + ' (curl)', ls='--', c=ln[0].get_color())
    pl.plot(ls, wls(ls) * cls_unl['pp'][ls], c='k', label=r'$C_L^{\phi\phi, \rm fid}$')
    pl.xlabel(r'$L$')
    pl.ylabel(r'$10^7 \: L^2(L + 1)^2 N_L^{(0)} / 2 \pi$')
    pl.legend(ncol=4)
    pl.show()
