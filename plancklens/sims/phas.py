from __future__ import print_function

import os
import sqlite3

import healpy as hp
import numpy as np
import pickle as pk

from plancklens.helpers import mpi
from plancklens import utils

class rng_db:
    """ Class to save and read random number generators states in a sqlite database file.

    """

    def __init__(self, fname, idtype="INTEGER"):
        if not os.path.exists(fname) and mpi.rank == 0:
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=3600)
            cur = con.cursor()
            cur.execute("create table rngdb (id %s PRIMARY KEY, "
                        "type STRING, pos INTEGER, has_gauss INTEGER,cached_gaussian REAL, keys STRING)" % idtype)
            con.commit()
        mpi.barrier()

        self.con = sqlite3.connect(fname, timeout=3600., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, idx, state):
        idx = int(idx)
        try:
            assert (self.get(idx) is None)
            keys_string = '_'.join(str(s) for s in state[1])
            self.con.execute("INSERT INTO rngdb (id, type, pos, has_gauss, cached_gaussian, keys) VALUES (?,?,?,?,?,?)",
                             (idx, state[0], state[2], state[3], state[4], keys_string))
            self.con.commit()
        except:
            print("rng_db::rngdb add failed!")

    def get(self, idx):
        idx = int(idx)
        cur = self.con.cursor()
        cur.execute("SELECT type, pos, has_gauss, cached_gaussian, keys FROM rngdb WHERE id=?", (idx,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            assert (len(data) == 5)
            typ, pos, has_gauss, cached_gaussian, keys = data
            keys = np.array([int(a) for a in keys.split('_')], dtype=np.uint32)
            return [typ, keys, pos, has_gauss, cached_gaussian]

    def delete(self, idx):
        idx = int(idx)
        try:
            if self.get(idx) is None:
                return
            self.con.execute("DELETE FROM rngdb WHERE id=?", (idx,))
            self.con.commit()
        except:
            print("rng_db::rngdb delete %s failed!" % idx)


class sim_lib(object):
    """Generic class for simulations where only rng state is stored.

    np.random rng states are stored in a sqlite3 database. By default the rng state function is np.random.get_state.
    The rng_db class is tuned for this state fct, you may need to adapt this.

    """

    def __init__(self, lib_dir, get_state_func=np.random.get_state, nsims_max=None):
        if not os.path.exists(lib_dir) and mpi.rank == 0:
            os.makedirs(lib_dir)
        self.nmax = nsims_max
        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if mpi.rank == 0 and not os.path.exists(fn_hash):
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()

        hsh = pk.load(open(fn_hash, 'rb'))
        utils.hash_check(hsh, self.hashdict(), ignore=['lib_dir'], fn=fn_hash)

        self._rng_db = rng_db(os.path.join(lib_dir, 'rngdb.db'), idtype='INTEGER')
        self._get_rng_state = get_state_func

    def get_sim(self, idx, **kwargs):
        """Returns sim number idx and caches random number generator state. """
        if self.has_nmax(): assert idx < self.nmax
        if not self.is_stored(idx):
            self._rng_db.add(idx, self._get_rng_state())
        return self._build_sim_from_rng(self._rng_db.get(idx), **kwargs)

    def has_nmax(self):
        return not self.nmax is None

    def is_stored(self, idx):
        """Checks whether sim idx is stored or not. Boolean output. """
        return not self._rng_db.get(idx) is None

    def is_full(self):
        """Checks whether all sims are stored or not. Boolean output. """
        if not self.has_nmax(): return False
        for idx in range(self.nmax):
            if not self.is_stored(idx): return False
        return True

    def is_empty(self):
        """Checks whether any sims is stored. Boolean output. """
        assert self.nmax is not None
        for idx in range(self.nmax):
            if self.is_stored(idx): return False
        return True

    def hashdict(self):
        """Override this """
        assert 0

    def _build_sim_from_rng(self, rng_state):
        """Override this """
        assert 0


class _pix_lib_phas(sim_lib):
    def __init__(self, lib_dir, shape, **kwargs):
        self.shape = shape
        super(_pix_lib_phas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, **kwargs):
        np.random.set_state(rng_state)
        return np.random.standard_normal(self.shape)

    def hashdict(self):
        return {'shape': self.shape}

class pix_lib_phas:
    def __init__(self, lib_dir, nfields, shape, **kwargs):
        self.nfields = nfields
        self.lib_pix = {}
        self.shape = shape
        for _i in range(nfields):
            self.lib_pix[_i] = _pix_lib_phas(os.path.join(lib_dir, 'pix_pha_%04d' % _i), shape, **kwargs)

    def is_full(self):
        return np.all([lib.is_full() for lib in self.lib_pix.values()])

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self.lib_pix[idf].get_sim(idx, phas_only=phas_only)
        return np.array([self.lib_pix[_idf].get_sim(idx, phas_only=phas_only) for _idf in range(self.nfields)])

    def hashdict(self):
        return {'nfields': self.nfields, 'shape': self.shape}

class _lib_phas(sim_lib):
    def __init__(self, lib_dir,lmax, **kwargs):
        self.lmax = lmax
        super(_lib_phas, self).__init__(lib_dir, **kwargs)

    def _build_sim_from_rng(self, rng_state, phas_only=False):
        np.random.set_state(rng_state)
        alm = (np.random.standard_normal(hp.Alm.getsize(self.lmax)) + 1j * np.random.standard_normal(hp.Alm.getsize(self.lmax))) / np.sqrt(2.)
        if phas_only: return
        m0 = hp.Alm.getidx(self.lmax, np.arange(self.lmax + 1,dtype = int),0)
        alm[m0] = np.sqrt(2.) * alm[m0].real
        return alm

    def hashdict(self):
        return {'lmax':self.lmax}

class lib_phas:
    def __init__(self, lib_dir, nfields, lmax, **kwargs):
        self.lmax = lmax
        self.nfields = nfields
        self.lib_phas = {}
        for i in range(nfields):
            self.lib_phas[i] = _lib_phas(os.path.join(lib_dir, 'pha_%04d'%i), lmax, **kwargs)

    def __getitem__(self, item):
        return self.lib_phas[item]

    def is_full(self):
        return np.all([lib.is_full() for lib in self.lib_phas.values()])

    def get_sim(self, idx, idf=None, phas_only=False):
        if idf is not None:
            assert idf < self.nfields, (idf, self.nfields)
            return self.lib_phas[idf].get_sim(idx, phas_only=phas_only)
        return np.array([self.lib_phas[_idf].get_sim(idx, phas_only=phas_only) for _idf in range(self.nfields)])


    def hashdict(self):
        return {'nfields': self.nfields, 'lmax':self.lmax}
