"""Parameter file for lensing reconstrution on a idealized, full-sky simulation library.

"""

import os
import healpy as hp
import numpy as np

from plancklens2018.filt import filt_simple, filt_util
from plancklens2018 import utils
from plancklens2018 import qest, qecl, qresp
from plancklens2018 import nhl
from plancklens2018.n1 import n1
from plancklens2018.sims import planck2018_sims, phas, maps, utils as maps_utils

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
cl_unl = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))
cl_weight = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.


libdir_ivfs  = os.path.join(PL2018, 'temp', 'idealized_example', 'ivfs')
libdir_pixphas = os.path.join(PL2018, 'temp', 'pix_phas_nside%s'%nside)


pix_phas = phas.pix_lib_phas(libdir_pixphas, 3, (hp.nside2npix(nside),))
sims = maps_utils.sim_lib_shuffle(maps.cmb_maps_nlev(planck2018_sims.cmb_len_ffp10(), transf, nlev_t, nlev_p, nside,
                            pix_lib_phas=pix_phas), {idx: nsims if idx == -1 else idx for idx in range(-1, nsims)})

ftl = utils.cli(cl_len['tt'][:lmax_ivf+ 1] + (nlev_t / 60. / 180. *np.pi  / transf) ** 2)
fel = utils.cli(cl_len['ee'][:lmax_ivf+ 1] + (nlev_p / 60. / 180. *np.pi  / transf) ** 2)
fbl = utils.cli(cl_len['bb'][:lmax_ivf+ 1] + (nlev_p / 60. / 180. *np.pi  / transf) ** 2)
ftl[:lmin_ivf] *= 0.
fel[:lmin_ivf] *= 0.
fbl[:lmin_ivf] *= 0.


ivfs    = filt_simple.library_fullsky_sepTP(libdir_ivfs, sims, nside, transf, cl_len, ftl, fel, fbl, cache=True)
# This remaps idx -> idx + 1 by blocks of 60 up to 300:
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
# This remap all sim. indices to the data maps
ds_dict = { k : -1 for k in range(300)}

ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

libdir_qlmsdd = os.path.join(PL2018, 'temp', 'idealized_example', 'qlms_dd')
libdir_qlmsds = os.path.join(PL2018, 'temp', 'idealized_example', 'qlms_ds')
libdir_qlmsss = os.path.join(PL2018, 'temp', 'idealized_example', 'qlms_ss')

qlms_dd = qest.library_sepTP(libdir_qlmsdd, ivfs, ivfs,   cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ds = qest.library_sepTP(libdir_qlmsds, ivfs, ivfs_d, cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ss = qest.library_sepTP(libdir_qlmsss, ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})


libdir_qcls_dd = os.path.join(PL2018, 'temp', 'idealized_example', 'qcls_dd')
libdir_qcls_ds = os.path.join(PL2018, 'temp', 'idealized_example', 'qcls_ds')
libdir_qcls_ss = os.path.join(PL2018, 'temp', 'idealized_example', 'qcls_ss')
libdir_nhl_dd = os.path.join(PL2018, 'temp', 'idealized_example', 'nhl_dd')
libdir_n1_dd = os.path.join(PL2018, 'temp', 'n1_ffp10')
libdir_resp_dd = os.path.join(PL2018, 'temp', 'idealized_example', 'qresp')

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
