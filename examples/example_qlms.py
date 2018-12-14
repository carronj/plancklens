
import os
import numpy as np

from plancklens2018.filt import filt_util
from plancklens2018 import qest

from . import example_filtering

PL2018 = os.environ['PL2018']

lmin_ivf = 100
lmax_ivf = 2048
lmax_qlm = 4096

ftl = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)
fel = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)
fbl = np.where(np.arange(lmax_ivf + 1) >= lmin_ivf, 1., 0.)

ivfs = filt_util.library_ftl(example_filtering.ivfs, lmax_ivf, ftl, fel, fbl)

libdir_qlms = os.path.join(PL2018, 'example_qlms', 'qlms_dd')
qlms_dd = qest.library_sepTP(libdir_qlms, ivfs, ivfs, example_filtering.cl_len['te'], example_filtering.nside,
                             lmax_qlm={'P': lmax_qlm, 'T':lmax_qlm})

if __name__ == '__main__':
    import argparse
    from plancklens2018 import mpi

    parser = argparse.ArgumentParser(description='PL2018 filtering example')
    parser.add_argument('-imin', dest='imin', default=-1, dtype=int, help='starting index (-1 stands for data map)')
    parser.add_argument('-imax', dest='imax', default=-2, dtype=int, help='last index')
    parser.add_argument('-k', dest='k', action='+', default=['p'],
                        help='QE keys (NB: both gradient anc curl are calculated at the same time)')

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
    jobs = []
    for k in args.k:
        jobs += [(idx, k) for idx in range(args.imin, args.imax + 1)]

    for i, (idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing QE sim %s %s, job %s in %s' % (mpi.rank, idx, k, i, len(jobs)))
        qlms_dd.get_sim_qlm(k, idx)

    mpi.barrier()
    mpi.finalize()




