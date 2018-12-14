#FIXME: mask paths and sims

import os
import healpy as hp
import numpy as np

from plancklens2018.filt import filt_cinv
from plancklens2018 import utils
from plancklens2018.sims import planck2018_sims

assert 'PL2018' in os.environ.keys(), 'Set env. variable PL2018 to the planck 2018 lensing directory'
PL2018 = os.environ['PL2018']

lmax_ivf = 2048
nside = 2048
nlev_t = 35.
nlev_p = 55.

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

if __name__ == '__main__':
    import argparse
    from plancklens2018 import mpi

    parser = argparse.ArgumentParser(description='Planck 2018 filtering example')
    parser.add_argument('-imin', dest='imin', default=-1, dtype=int, help='starting index (-1 stands for data map)')
    parser.add_argument('-imax', dest='imax', default=-2, dtype=int, help='last index')
    args = parser.parse_args()

    jobs =  [ (idx, 't') for idx in range(args.imin, args.imax + 1)]
    jobs += [ (idx, 'p') for idx in range(args.imin, args.imax + 1)]

    for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing sim %s %s, job %s in %s'%(mpi.rank, idx, lab, i, len(jobs)))
        if lab == 't':
            ivfs.get_sim_tlm(idx)
        elif lab == 'p':
            ivfs.get_sim_elm(idx) # This will cache blm as well.

    mpi.barrier()
    mpi.finalize()


