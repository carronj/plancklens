"""mpi4py wrapper module.

"""

import os

verbose = False
has_key = lambda key : key in os.environ.keys()

# cond4mpi4py = not has_key('NERSC_HOST') or (has_key('NERSC_HOST') and has_key('SLURM_SUBMIT_DIR'))

if has_key('USE_PLANCKLENS_MPI'):
    use = os.environ['USE_PLANCKLENS_MPI']
else:
    use = True

cond4mpi4py = 'srun' in os.environ.get('_', '') or 'mpirun' in os.environ.get('_', '')

if cond4mpi4py and use:
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        bcast = MPI.COMM_WORLD.bcast
        send = MPI.COMM_WORLD.send
        receive = MPI.COMM_WORLD.recv
        finalize = MPI.Finalize
        ANY_SOURCE = MPI.ANY_SOURCE

        if verbose: print('mpi.py : setup OK, rank %s in %s' % (rank, size))
    except:
        rank = 0
        size = 1
        barrier = lambda: -1
        bcast = lambda _: 0
        send = lambda _, dest: 0
        receive = lambda _, source: 0
        finalize = lambda: -1
        ANY_SOURCE = 0

        if verbose and use: print('mpi.py: unable to import mpi4py\n')
else:
    if verbose: print('mpi.py: not importing mpi4py\n')
    rank = 0
    size = 1
    barrier = lambda: -1
    bcast = lambda _: 0
    send = lambda _, dest: 0
    receive = lambda _, source: 0
    finalize = lambda: -1
    ANY_SOURCE = 0
    
    if not use: print('mpi.py: Plancklens.mpi disabled as per environ variable \n')