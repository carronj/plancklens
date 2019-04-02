import os
import numpy as np

from plancklens2018 import qecl
from plancklens2018 import nhl
from plancklens2018 import utils
from plancklens2018 import n1
import example_2_qlms as qlms

assert 'PL2018' in os.environ.keys(), 'Set env. variable PL2018 to the planck 2018 lensing directory'
PL2018 = os.environ['PL2018']

mc_sims_bias = np.arange(60)
mc_sims_var  = np.arange(60, 300)

libdir_qcls_dd = os.path.join(PL2018, 'temp', 'example_qcls', 'qcls_dd')
libdir_qcls_ds = os.path.join(PL2018, 'temp', 'example_qcls', 'qcls_ds')
libdir_qcls_ss = os.path.join(PL2018, 'temp', 'example_qcls', 'qcls_ss')
libdir_nhl_dd = os.path.join(PL2018, 'temp', 'example_qcls', 'nhl_dd')
libdir_n1 = os.path.join(PL2018, 'temp', 'example_qcls', 'n1')

mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ds = np.array([])
mc_sims_mf_ss = np.array([])

qcls_dd = qecl.library(libdir_qcls_dd, qlms.qlms_dd, qlms.qlms_dd, mc_sims_mf_dd)
qcls_ds = qecl.library(libdir_qcls_ds, qlms.qlms_ds, qlms.qlms_ds, mc_sims_mf_ds)
qcls_ss = qecl.library(libdir_qcls_ss, qlms.qlms_ss, qlms.qlms_ss, mc_sims_mf_ss)

cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))
cl_weight = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))

nhl_dd = nhl.nhl_lib_simple(libdir_nhl_dd, qlms.ivfs, cl_weight, qlms.lmax_qlm)
n1_dd = n1.library_n1(libdir_n1,cl_len['tt'],cl_len['te'],cl_len['ee'])

if __name__ == '__main__':
    import argparse
    from plancklens2018 import mpi

    parser = argparse.ArgumentParser(description='Planck 2018 QE power spectra calc. example')
    parser.add_argument('-imin', dest='imin', default=-1, type=int, help='starting index (-1 stands for data map)')
    parser.add_argument('-imax', dest='imax', default=-2, type=int, help='last index')
    parser.add_argument('-k', dest='k', nargs='+', action='store', default=['p'],
                        help='QE keys (NB: both gradient anc curl are calculated at the same time)')

    parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qcls library QEs')
    parser.add_argument('-ds', dest='ds', action='store_true', help='perform ds qcls library QEs')
    parser.add_argument('-ss', dest='ss', action='store_true', help='perform ss qcls library QEs')

    args = parser.parse_args()

    #--- filtering
    jobs =  [ (idx, 't') for idx in range(args.imin, args.imax + 1)]
    jobs += [ (idx, 'p') for idx in range(args.imin, args.imax + 1)]

    for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s filtering sim %s %s, job %s in %s'%(mpi.rank, idx, lab, i, len(jobs)))
        if lab == 't':
            qlms.ivfs.get_sim_tlm(idx)
        elif lab == 'p':
            qlms.ivfs.get_sim_elm(idx) # This will cache blm as well.
    mpi.barrier()

    #--- QE calculation
    qlibs = [qlms.qlms_dd] * args.dd + [qlms.qlms_ds] * args.ds +  [qlms.qlms_ss] * args.ss
    jobs = []
    for qlib in qlibs:
        for k in args.k:
            jobs += [(qlib, idx, k) for idx in range(args.imin, args.imax + 1)]

    for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing QE sim %s %s, qlm_lib %s, job %s in %s' % (mpi.rank, idx, k, qlib.lib_dir, i, len(jobs)))
        qlib.get_sim_qlm(k, idx)
    mpi.barrier()

    #--- Mean-field calculation (this assumes the two qlms lib. in qcls are identical)
    #    Here we require all qlms in mc_sims_mf to be calculated for the mf calc to go through.
    qlibs = [qcls_dd] * args.dd + [qcls_ds] * args.ds +  [qcls_ss] * args.ss
    jobs = []
    for qlib in qlibs:
        for k in args.k:
            if len(qlib.mc_sims_mf) > 0:
                if np.all([idx in range(args.imin, args.imax) for idx in qlib.mc_sims_mf]):
                    jobs.append( (qlib, k, 0))
                    jobs.append( (qlib, k, 1))
                else:
                    if mpi.rank == 0:
                        print('Skipping mf calc for ' + qlib.lib_dir)
    for i, (qlib, k, start) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing MF leg %s %s, qcl_lib %s, job %s in %s' % (mpi.rank, start, k, qlib.lib_dir, i, len(jobs)))
        qlib.qeA.get_sim_qlm_mf(k, qlib.mc_sims_mf[start::2])
    mpi.barrier()

    #--- QE power spectra
    qlibs = [qcls_dd] * args.dd + [qcls_ds] * args.ds +  [qcls_ss] * args.ss
    jobs = []
    for qlib in qlibs:
        for k in args.k:
            for idx in range(args.imin, args.imax):
                if idx not in qlib.mc_sims_mf:
                    jobs.append((qlib, idx, k))

    for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing QE spectra sim %s %s, qcl_lib %s, job %s in %s' % (mpi.rank, idx, k, qlib.lib_dir, i, len(jobs)))
        qlib.get_sim_qcl(k, idx)

    mpi.barrier()
    mpi.finalize()




