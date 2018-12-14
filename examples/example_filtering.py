#FIXME: mask paths

import os
import healpy as hp
import numpy as np

from plancklens2018.filt import filt_cinv, filt_util
from plancklens2018 import utils
from plancklens2018.sims import planck2018_sims

PL2018 = os.environ['PL2018']

lmin_ivf = 100
lmax_ivf = 2048
nside = 2048
nlev_t = 35.
nlev_p = 55.

#transf = hp.beam_from_fwhm_arcmin(5., params['lmax_sky']) * ist.healpix_window(params['nside'],params['lmax_sky'])
transf = hp.gauss_beam(5. / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
cl_unl = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))


sim_lib  = planck2018_sims.smica_dx12()

# Masks
Tmaskpaths = [os.environ['CSCRATCH'] + '/jpipe/inputs/PR3vApr6_temp_lensingmask_gPR2_70_psPR2_143_COT2_psPR2_217_sz.fits.gz']
Tmaskpaths += ['/project/projectdirs/planck/data/compsep/comparison/dx12_v3/dx12_v3_common_ps_mask_int_005a_2048_v2.fits.gz',
               '/project/projectdirs/planck/data/compsep/comparison/dx12_v3/dx12_v3_common_ps_mask_pol_005a_2048_v2.fits.gz']


libdir_cinvt = os.path.join(PL2018, 'temp', 'example_filtering', 'cinv_t')
libdir_cinvp = os.path.join(PL2018, 'temp', 'example_filtering', 'cinv_p')
libdir_ivfs  = os.path.join(PL2018, 'temp', 'example_filtering', 'ivfs')

ninv_t = [np.array([3. / nlev_t ** 2])] + Tmaskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])
ninv_p = [[np.array([3. / nlev_p ** 2])] + Tmaskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sim_lib, cinv_t, cinv_p)

