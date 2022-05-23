"""mpi4py wrapper module.

"""

from __future__ import print_function
import os

verbose = True

has_key = lambda key : key in os.environ.keys()
cond4mpi4py = not has_key('NERSC_HOST') or (has_key('NERSC_HOST') and has_key('SLURM_SUBMIT_DIR'))

if cond4mpi4py:
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
        if verbose: print('mpi.py : setup OK, rank %s in %s' % (rank, size))
    except:
        rank = 0
        size = 1
        barrier = lambda: -1
        finalize = lambda: -1
        if verbose: print('mpi.py: unable to import mpi4py\n')
else:
    if verbose: print('mpi.py: not importing mpi4py\n')
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1