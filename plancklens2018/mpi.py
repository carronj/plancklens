from __future__ import print_function

verbose = False
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    barrier = MPI.COMM_WORLD.Barrier
    finalize = MPI.Finalize
    if verbose:
        print('mpi.py : setup OK, rank %s in %s' % (rank, size))
except:
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1
    if verbose:
        print('mpi.py: unable to import mpi4py\n')
