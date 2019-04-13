"""Parameter file for lensing reconstrution on the Planck public release 3 SMICA cmb map.

"""

import os
import healpy as hp
import numpy as np

from plancklens2018.filt import filt_cinv, filt_util
from plancklens2018 import utils
from plancklens2018 import qest, qecl, qresp
from plancklens2018 import nhl
from plancklens2018.n1 import n1
from plancklens2018.sims import planck2018_sims, cmbs, phas, maps, utils as maps_utils

assert 'PL2018' in os.environ.keys(), 'Set env. variable PL2018 to the planck 2018 lensing directory'
PL2018 = os.environ['PL2018']

lmax_ivf = 2048
lmin_ivf = 100
lmax_qlm = 4096
nside = 2048
nlev_t = 35.
nlev_p = 55.
nsims = 300

transf = hp.gauss_beam(5. / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
cl_unl = utils.camb_clfile(os.path.join(PL2018, 'inputs', 'cls', 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs', 'cls', 'FFP10_wdipole_lensedCls.dat'))
cl_weight = utils.camb_clfile(os.path.join(PL2018, 'inputs', 'cls', 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.

# Masks
#FIXME: paths
Tmaskpaths = ['/global/cscratch1/sd/jcarron/jpipe/inputs/PR3vApr6_temp_lensingmask_gPR2_70_psPR2_143_COT2_psPR2_217_sz.fits.gz']

libdir_cinvt = os.path.join(PL2018, 'temp', 'smicadx12', 'cinv_t')
libdir_cinvp = os.path.join(PL2018, 'temp', 'smicadx12', 'cinv_p')
libdir_ivfs  = os.path.join(PL2018, 'temp', 'smicadx12', 'ivfs')
libdir_dclphas = os.path.join(PL2018, 'temp', 'smicadx12', 'dcl_phas')

dcl_phas = phas.lib_phas(libdir_dclphas, 3, 2048)

#FIXME:
dcl = np.loadtxt('/global/cscratch1/sd/jcarron/share/Planck_L08_inputs/dcls/smicadx12_Dec5_dcl_tteebbsigsmo200b0a3f9a87d6dcdd4c8ec85ece9498540f7e742bcsmooth200_dcl.dat').transpose()
dcl_dat = np.loadtxt('/global/cscratch1/sd/jcarron/share/Planck_L08_inputs/dcls_dat/smicadx12_Dec5_dcl_tteebbsigsmo200b0a3f9a87d6dcdd4c8ec85ece9498540f7e742bcsmooth200_dcl.dat').transpose()

sims_raw  = planck2018_sims.smica_dx12()
sims_dcl_sim = maps.cmb_maps_noisefree(cmbs.sims_cmb_unl({'tt':dcl[0], 'ee':dcl[1], 'bb':dcl[2]}, dcl_phas), transf)
sims_dcl_dat = maps_utils.sim_lib_shuffle(maps.cmb_maps_noisefree(cmbs.sims_cmb_unl({'tt':dcl_dat[0], 'ee':dcl_dat[1], 'bb':dcl_dat[2]}, dcl_phas), transf), {-1:nsims})
sims = maps_utils.sim_lib_add_dat([maps_utils.sim_lib_add_sim([sims_raw, sims_dcl_sim]), sims_dcl_dat])


ninv_t = [np.array([3. / nlev_t ** 2])] + Tmaskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([3. / nlev_p ** 2])] + Tmaskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)
# This remaps idx -> idx + 1 by blocks of 60 up to 300:
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
# This remap all sim. indices to the data maps
ds_dict = { k : -1 for k in range(300)}

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

libdir_qlmsdd = os.path.join(PL2018, 'temp', 'smicadx12', 'qlms_dd')
libdir_qlmsds = os.path.join(PL2018, 'temp', 'smicadx12', 'qlms_ds')
libdir_qlmsss = os.path.join(PL2018, 'temp', 'smicadx12', 'qlms_ss')

qlms_dd = qest.library_sepTP(libdir_qlmsdd, ivfs, ivfs,   cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ds = qest.library_sepTP(libdir_qlmsds, ivfs, ivfs_d, cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ss = qest.library_sepTP(libdir_qlmsss, ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})


libdir_qcls_dd = os.path.join(PL2018, 'temp', 'smicadx12', 'qcls_dd')
libdir_qcls_ds = os.path.join(PL2018, 'temp', 'smicadx12', 'qcls_ds')
libdir_qcls_ss = os.path.join(PL2018, 'temp', 'smicadx12', 'qcls_ss')
libdir_nhl_dd = os.path.join(PL2018, 'temp', 'smicadx12', 'nhl_dd')
libdir_n1_dd = os.path.join(PL2018, 'temp', 'n1ffp10')
libdir_resp_dd = os.path.join(PL2018, 'temp', 'smicadx12', 'qresp')

mc_sims_bias = np.arange(60)
mc_sims_var  = np.arange(60, 300)

mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ds = np.array([])
mc_sims_mf_ss = np.array([])

qcls_dd = qecl.library(libdir_qcls_dd, qlms_dd, qlms_dd, mc_sims_mf_dd)
qcls_ds = qecl.library(libdir_qcls_ds, qlms_ds, qlms_ds, mc_sims_mf_ds)
qcls_ss = qecl.library(libdir_qcls_ss, qlms_ss, qlms_ss, mc_sims_mf_ss)

nhl_dd = nhl.nhl_lib_simple(libdir_nhl_dd, ivfs, cl_weight, lmax_qlm)
n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])
qresp_dd = qresp.resp_lib_simple(libdir_resp_dd, lmax_ivf, cl_weight, cl_len,{'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)
