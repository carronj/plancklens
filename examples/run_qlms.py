"""This script may be used to

    - inverse-variance filter the CMB maps
    - build the QE estimates from them
    - build the mean-field estimates
    - build the QE spectra, MC-N0 and RD-N0 terms
    - calculate the semi-analytical N0's
    - calculate cross-spectra to FFP10 CMB lensing potential input maps

    The script takes as input a parameter file looking like e.g. /params/smicadx12_planck2018.py

    For example, to builds the polarization QE,

        srun -n 30 python params/smicadx12_planck2018.py run_qlms.py  -imin 0 -imax 299 -k p_p -ivp -dd

    uses 30 processes to calculate the polarization-only QE for the simulations 0 to 299

"""

import argparse
import numpy as np
from importlib.machinery import SourceFileLoader
from plancklens.helpers import mpi

parser = argparse.ArgumentParser(description='Planck 2018 QE calculation example')
parser.add_argument('parfile', type=str, nargs=1)
parser.add_argument('-imin', dest='imin', default=-1, type=int, help='starting index (-1 stands for data map)')
parser.add_argument('-imax', dest='imax', default=-2, type=int, help='last index')
parser.add_argument('-k', dest='k', action='store', default=[], nargs='+',
                    help='QE keys (NB: both gradient and curl are calculated at the same time)')
parser.add_argument('-kxi', dest='kxi', action='store', default=[], nargs='+',
                    help='QE keys to calculate x to input to lensing for')
parser.add_argument('-kA', dest='kA', action='store', default=[], nargs='+',  help='QE spectra keys (left leg)')
parser.add_argument('-kB', dest='kB', action='store', default=[], nargs='+', help='QE spectra keys (right leg)')
parser.add_argument('-ivt', dest='ivt', action='store_true', help='do T. filtering')
parser.add_argument('-ivp', dest='ivp', action='store_true', help='do P. filtering')
parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms / qcls library QEs')
parser.add_argument('-ds', dest='ds', action='store_true', help='perform ds qlms / qcls library QEs')
parser.add_argument('-ss', dest='ss', action='store_true', help='perform ss qlms / qcls library QEs')
parser.add_argument('-mfdd', dest='mfdd', action='store_true', help='perform dd qlms mean-fields for qcls keys')
parser.add_argument('-kN', dest='kN', action='store', default=[], nargs='+', help='keys for QE semi-analytical noise spectra')


args = parser.parse_args()
par = SourceFileLoader('run_qlms_parfile', args.parfile[0]).load_module()

#--- filtering
jobs = []
if args.ivt:
    jobs += [(idx, 't') for idx in range(args.imin, args.imax + 1)]
    if args.ds and args.imin >= 0: #  Make data to avoid problems with ds librairies
        jobs += [(-1, 't')]
if args.ivp:
    jobs += [(idx, 'p') for idx in range(args.imin, args.imax + 1)]
    if args.ds and args.imin >= 0: #  Make data to avoid problems with ds librairies
        jobs += [(-1, 'p')]
for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s filtering sim %s %s, job %s in %s' % (mpi.rank, idx, lab, i, len(jobs[mpi.rank::mpi.size])))
    if lab == 't':
        par.ivfs.get_sim_tlm(idx)
    elif lab == 'p':
        par.ivfs.get_sim_elm(idx) # This will cache blm as well.
mpi.barrier()

# --- unnormalized QE calculation
qlibs = [par.qlms_dd] * args.dd +  [par.qlms_ss] * args.ss + [par.qlms_ds] * args.ds
jobs = []
for qlib in qlibs:
    for k in args.k:
        jobs += [(qlib, idx, k) for idx in range(args.imin, args.imax + 1)]

for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s doing QE sim %s %s, qlm_lib %s, job %s in %s' % (mpi.rank, idx, k, qlib.lib_dir, i, len(jobs)))
    qlib.get_sim_qlm(k, idx)
mpi.barrier()

# --- crosses to input:
if hasattr(par, 'qlms_x_in'):
    qlibs = [par.qlms_x_in]
    jobs = []
    for qlib in qlibs:
        for k in args.kxi:
            jobs += [(qlib, idx, k) for idx in range(args.imin, args.imax + 1)]
    for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
        print('rank %s doing QE x inpu sim %s %s, job %s in %s' % (mpi.rank, idx, k, i, len(jobs)))
        qlib.get_sim_qcl(k, idx)

#--- mean-fields
if args.mfdd:
    jobs = list(np.unique(np.concatenate([args.kA, args.kB])))
    jobs = [(job, 0) for job in jobs] + [(job, 1) for job in jobs]
    for i, (k, id0) in enumerate(jobs[mpi.rank::mpi.size]):
        print("rank %s doing %s QE MF %s"%(mpi.rank, k, id0))
        par.qlms_dd.get_sim_qlm_mf(k, par.qcls_dd.mc_sims_mf[id0::2])
mpi.barrier()
#--- unnormalized QE power spectra
qlibs = [par.qcls_dd] * args.dd +  [par.qcls_ss] * args.ss + [par.qcls_ds] * args.ds
jobs = []
for qlib in qlibs:
    for kA in args.kA:
        for kB in args.kB:
            for idx in range(args.imin, args.imax):
                if idx not in qlib.mc_sims_mf:
                    jobs.append((qlib, idx, kA, kB))

for i, (qlib, idx, kA, kB) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s doing QE spectra sim %s %s %s, qcl_lib %s, job %s in %s' % (
    mpi.rank, idx, kA, kB, qlib.lib_dir, i, len(jobs)))
    qlib.get_sim_qcl(kA, idx, k2=kB)

# --- semi-analytical unnormalized N0 calculation
jobs = []
for k in args.kN:
    jobs += [(idx, k) for idx in range(args.imin, args.imax + 1)]

for i, (idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s doing QE sim %s %s, qlm_lib %s, job %s in %s' % (mpi.rank, idx, k, par.nhl_dd.lib_dir, i, len(jobs)))
    par.nhl_dd.get_sim_nhl(idx, k, k)

mpi.barrier()
mpi.finalize()




