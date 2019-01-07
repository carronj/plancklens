import os
import numpy as np

from plancklens2018.filt import filt_util
from plancklens2018 import qest

import example_1_filtering

assert 'PL2018' in os.environ.keys(), 'Set env. variable PL2018 to the planck 2018 lensing directory'
PL2018 = os.environ['PL2018']

lmin_ivf = 100
lmax_ivf = 2048
lmax_qlm = 4096
nside = 2048
clte = example_1_filtering.cl_len['te']

ftl = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)
fel = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)
fbl = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)

# This remaps idx -> idx + 1 by blocks of 60 up to 300:
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                    np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
# This remap all sim. indices to the data maps
ds_dict = { k : -1 for k in range(300)}

ivfs   = filt_util.library_ftl(example_1_filtering.ivfs, lmax_ivf, ftl, fel, fbl)
ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)

libdir_qlmsdd = os.path.join(PL2018, 'example_qlms', 'qlms_dd')
libdir_qlmsds = os.path.join(PL2018, 'example_qlms', 'qlms_ds')
libdir_qlmsss = os.path.join(PL2018, 'example_qlms', 'qlms_ss')

qlms_dd = qest.library_sepTP(libdir_qlmsdd, ivfs, ivfs  , clte, nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ds = qest.library_sepTP(libdir_qlmsds, ivfs, ivfs_d, clte, nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})
qlms_ss = qest.library_sepTP(libdir_qlmsss, ivfs, ivfs_s, clte, nside, lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})

if __name__ == '__main__':
    import argparse
    from plancklens2018 import mpi

    parser = argparse.ArgumentParser(description='Planck 2018 QE calculation example')
    parser.add_argument('-imin', dest='imin', default=-1, type=int, help='starting index (-1 stands for data map)')
    parser.add_argument('-imax', dest='imax', default=-2, type=int, help='last index')
    parser.add_argument('-k', dest='k', action='store', default=['p'], nargs='+',
                        help='QE keys (NB: both gradient and curl are calculated at the same time)')

    parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms library QEs')
    parser.add_argument('-ds', dest='ds', action='store_true', help='perform ds qlms library QEs')
    parser.add_argument('-ss', dest='ss', action='store_true', help='perform ss qlms library QEs')

    args = parser.parse_args()

    #--- filtering
    jobs =  [ (idx, 't') for idx in range(args.imin, args.imax + 1)]
    jobs += [ (idx, 'p') for idx in range(args.imin, args.imax + 1)]

    for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s filtering sim %s %s, job %s in %s'%(mpi.rank, idx, lab, i, len(jobs)))
        if lab == 't':
            ivfs.get_sim_tlm(idx)
        elif lab == 'p':
            ivfs.get_sim_elm(idx) # This will cache blm as well.
    mpi.barrier()

    # --- QE calculation
    qlibs = [qlms_dd] * args.dd + [qlms_ds] * args.ds +  [qlms_ss] * args.ss
    jobs = []
    for qlib in qlibs:
        for k in args.k:
            jobs += [(qlib, idx, k) for idx in range(args.imin, args.imax + 1)]

    for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing QE sim %s %s, qlm_lib %s, job %s in %s' % (mpi.rank, idx, k, qlib.lib_dir, i, len(jobs)))
        qlib.get_sim_qlm(k, idx)

    mpi.barrier()
    mpi.finalize()




