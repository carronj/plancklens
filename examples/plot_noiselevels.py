"""This plots the Gaussian noise levels for a given ansisotropy source QE estimators.

"""
import numpy as np
import healpy as hp
import pylab as pl
import os

import plancklens
from plancklens import utils
from plancklens import nhl, qresp

cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')

ksource = 'p'
fname = None # If set, will try to save figure to this file.

lmax_ivf = 2048
lmin_ivf = 100
nlev_t = 35.
nlev_p = 35. * np.sqrt(2.)
beam_fwhm = 6.
lmax_qlm = lmax_ivf


if ksource in ['p', 'f']:
    qe_keys = [ksource + 'tt', ksource+'_p', ksource]
    qe_keys_lab = [ (r'$\hat\phi^{%s}$' if ksource == 'p' else 'f')%l for l in ['TT', 'P.', 'MV']]
elif ksource in ['a', 'a_p', 'stt']:
    qe_keys = [ksource]
    qe_keys_lab = [ksource]
else:
    assert 0

transf = hp.gauss_beam(beam_fwhm / 60. / 180. * np.pi, lmax=lmax_ivf)

cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cls_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

fal_sepTP =  {'tt': utils.cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2),
              'ee': utils.cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2),
              'bb': utils.cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)}


cls_ivfs_sepTP = {'tt':fal_sepTP['tt'].copy(),
                  'ee':fal_sepTP['ee'].copy(),
                  'bb':fal_sepTP['bb'].copy(),
                  'te':cls_len['te'][:lmax_ivf + 1] * fal_sepTP['tt'] * fal_sepTP['ee']}
cls_dat = {
    'tt': (cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. /180. * np.pi) ** 2 / transf ** 2),
    'ee': (cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. /180. * np.pi) ** 2 / transf ** 2),
    'bb': (cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. /180. * np.pi) ** 2 / transf ** 2),
    'te':  np.copy(cls_len['te'][:lmax_ivf + 1]) }

fal_jtTP = utils.cl_inverse(cls_dat)
cls_ivfs_jtTP = utils.cl_inverse(cls_dat)

for cls in [fal_sepTP, fal_jtTP, cls_ivfs_sepTP, cls_ivfs_jtTP]:
    for cl in cls.values():
        cl[:max(1, lmin_ivf)] *= 0.

if ksource == 'p':
    w = lambda ell : ell ** 2 *(ell + 1) ** 2 * 1e7 * 0.5 / np.pi
    ylabel = r'$10^7\cdot L^2(L + 1)^2 C_L^{\phi\phi} / 2\pi$'
elif ksource == 'f':
    w = lambda ell : 1.
    ylabel = r'$C_L^{ff}$'
elif ksource == 'a':
    w = lambda ell : 1.
    ylabel = r'$C_L^{\alpha\alpha}$'
else:
    assert 0
ellp = np.arange(1, lmax_qlm + 1)
ellc = np.arange(2, lmax_qlm + 1)

for qe_key, label in zip(qe_keys, qe_keys_lab):
    NG, NC, NGC, NCG = nhl.get_nhl(qe_key, qe_key, cls_weight, cls_ivfs_sepTP, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)
    RG, RC, RGC, RCG = qresp.get_response(qe_key, lmax_ivf, ksource, cls_weight, cls_len, fal_sepTP, lmax_qlm=lmax_qlm)

    N0 = utils.cli(RG ** 2) * NG
    ln = pl.loglog(ellp, w(ellp) * N0[ellp], label=label + ' (sep. TP filt.)' * (qe_key == 'p'))
    if qe_key[0] in ['p', 'x', 'f']:
        N0 = utils.cli(RC ** 2) * NC
        pl.loglog(ellc, w(ellc) * N0[ellc], label=(label.replace('\hat\phi', '\hat\omega') + ' (Curl)') * (qe_key == 'ptt'), c=ln[0].get_color(), ls='--')

NG, NC, NGC, NCG = nhl.get_nhl(ksource, ksource, cls_weight, cls_ivfs_jtTP, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)
RG, RC, RGC, RCG = qresp.get_response(ksource, lmax_ivf, ksource, cls_weight, cls_len, fal_jtTP, lmax_qlm=lmax_qlm)
N0 = utils.cli(RG ** 2) * NG
ln =pl.loglog(ellp, w(ellp) * N0[ellp], label=label + ' (jt. TP filt.)')
N0 = utils.cli(RC ** 2) * NC
pl.loglog(ellc, w(ellc) * N0[ellc], c=ln[0].get_color(), ls='--')
pl.xlabel(r'$L$')
pl.ylabel(ylabel)
pl.legend()
pl.tight_layout()
if fname is not None:
    pl.savefig(fname, bbox_inches='tight')
pl.show()

