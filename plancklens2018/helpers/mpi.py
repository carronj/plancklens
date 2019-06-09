"""mpi4py wrapper module.

"""

from __future__ import print_function
import os


verbose = False

#FIXME:
has_key = lambda key : key in os.environ.keys()
cond4mpi4py = not has_key('NERSC_HOST') or (has_key('NERSC_HOST') and has_key('SLURM_SUBMIT_DIR'))
"""Somehow my mpi4py import kills the python kernel on the NERSC login nodes ?!"""

if cond4mpi4py:
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        finalize = MPI.Finalize
        if verbose: print('mpi.py : setup OK, rank %s in %s' % (rank, size))
    except:
        pass

else:
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    if verbose: print('mpi.py: unable to import mpi4py\n')
