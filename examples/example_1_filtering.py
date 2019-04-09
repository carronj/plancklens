#FIXME: mask paths and sims
"""

    NB: on first call, the dense temperature and polarization pre-conditioners will be computed and cached.
"""

import os
import healpy as hp
import numpy as np
import pickle as pk

from plancklens2018.filt import filt_cinv
from plancklens2018 import utils
from plancklens2018.sims import planck2018_sims, phas, cmbs, maps, utils as maps_utils

assert 'PL2018' in os.environ.keys(), 'Set env. variable PL2018 to the planck 2018 lensing directory'
PL2018 = os.environ['PL2018']

lmax_ivf = 2048
nside = 2048
nlev_t = 35.
nlev_p = 55.
nsims = 300

transf = hp.gauss_beam(5. / 60. / 180. * np.pi, lmax=lmax_ivf) * hp.pixwin(nside)[:lmax_ivf + 1]
cl_unl = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))



# Masks
#FIXME: paths
Tmaskpaths = ['/global/cscratch1/sd/jcarron/jpipe/inputs/PR3vApr6_temp_lensingmask_gPR2_70_psPR2_143_COT2_psPR2_217_sz.fits.gz']

#dcl:
#FIXME:
dcl = pk.load(open('/global/cscratch1/sd/jcarron/share/Planck_L08_inputs/dcls/smicadx12_Dec5_dcl_tteebbsigsmo200b0a3f9a87d6dcdd4c8ec85ece9498540f7e742bcsmooth200_dcl.pk','r'))
dcl_dat = pk.load(open('/global/cscratch1/sd/jcarron/share/Planck_L08_inputs/dcls_dat/smicadx12_Dec5_dcl_tteebbsigsmo200b0a3f9a87d6dcdd4c8ec85ece9498540f7e742bcsmooth200_dcl.pk','r'))


libdir_cinvt = os.path.join(PL2018, 'temp', 'example', 'cinv_t')
libdir_cinvp = os.path.join(PL2018, 'temp', 'example', 'cinv_p')
libdir_ivfs  = os.path.join(PL2018, 'temp', 'example', 'ivfs')
libdir_dclphas = os.path.join(PL2018, 'temp', 'example', 'dcl_phas')

dcl_phas = phas.lib_phas(libdir_dclphas, 3, 2048)

sims_raw  = planck2018_sims.smica_dx12()
sims_dcl_sim = maps.cmb_maps_noisefree(cmbs.sims_cmb_unl(dcl, dcl_phas), transf)
sims_dcl_dat = maps_utils.sim_lib_shuffle(maps.cmb_maps_noisefree(cmbs.sims_cmb_unl(dcl_dat, dcl_phas), transf), {-1:nsims})
sims = maps_utils.sim_lib_add_dat([maps_utils.sim_lib_add_sim([sims_raw, sims_dcl_sim]), sims_dcl_dat])


ninv_t = [np.array([3. / nlev_t ** 2])] + Tmaskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([3. / nlev_p ** 2])] + Tmaskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)

if __name__ == '__main__':
    import argparse
    from plancklens2018 import mpi

    parser = argparse.ArgumentParser(description='Planck 2018 filtering example')
    parser.add_argument('-imin', dest='imin', default=-1, type=int, help='starting index (-1 stands for data map)')
    parser.add_argument('-imax', dest='imax', default=-2, type=int, help='last index')
    parser.add_argument('-t', dest='t', action='store_false', help='do not filter temperature')
    parser.add_argument('-p', dest='p', action='store_false', help='do not filter polarization')

    args = parser.parse_args()
    jobs = []
    if args.t: jobs +=  [ (idx, 't') for idx in range(args.imin, args.imax + 1)]
    if args.p: jobs +=  [ (idx, 'p') for idx in range(args.imin, args.imax + 1)]

    for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing sim %s %s, job %s in %s'%(mpi.rank, idx, lab, i, len(jobs)))
        if lab == 't':
            ivfs.get_sim_tlm(idx)
        elif lab == 'p':
            ivfs.get_sim_elm(idx) # This will cache blm as well.

    mpi.barrier()
    mpi.finalize()


