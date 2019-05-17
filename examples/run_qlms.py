import argparse
import imp
from plancklens2018 import mpi

parser = argparse.ArgumentParser(description='Planck 2018 QE calculation example')
parser.add_argument('parfile', type=str, nargs=1)
parser.add_argument('-imin', dest='imin', default=-1, type=int, help='starting index (-1 stands for data map)')
parser.add_argument('-imax', dest='imax', default=-2, type=int, help='last index')
parser.add_argument('-k', dest='k', action='store', default=['p'], nargs='+',
                    help='QE keys (NB: both gradient and curl are calculated at the same time)')
parser.add_argument('-ivt', dest='ivt', action='store_true', help='do T. filtering')
parser.add_argument('-ivp', dest='ivp', action='store_true', help='do P. filtering')
parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms library QEs')
parser.add_argument('-ds', dest='ds', action='store_true', help='perform ds qlms library QEs')
parser.add_argument('-ss', dest='ss', action='store_true', help='perform ss qlms library QEs')

args = parser.parse_args()
par = imp.load_source('run_qlms_parfile', args.parfile[0])

#--- filtering
jobs = []
if args.ivt:
    jobs +=  [ (idx, 't') for idx in range(args.imin, args.imax + 1)]
if args.ivp:
    jobs += [ (idx, 'p') for idx in range(args.imin, args.imax + 1)]

for i, (idx, lab) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s filtering sim %s %s, job %s in %s'%(mpi.rank, idx, lab, i, len(jobs)))
    if lab == 't':
        par.ivfs.get_sim_tlm(idx)
    elif lab == 'p':
        par.ivfs.get_sim_elm(idx) # This will cache blm as well.
mpi.barrier()

# --- QE calculation
qlibs = [par.qlms_dd] * args.dd + [par.qlms_ds] * args.ds +  [par.qlms_ss] * args.ss
jobs = []
for qlib in qlibs:
    for k in args.k:
        jobs += [(qlib, idx, k) for idx in range(args.imin, args.imax + 1)]

for i, (qlib, idx, k) in enumerate(jobs[mpi.rank::mpi.size]):
    print('rank %s doing QE sim %s %s, qlm_lib %s, job %s in %s' % (mpi.rank, idx, k, qlib.lib_dir, i, len(jobs)))
    qlib.get_sim_qlm(k, idx)

mpi.barrier()
mpi.finalize()




