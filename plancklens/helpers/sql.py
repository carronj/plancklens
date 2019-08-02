from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import sqlite3
import io
import os
import six # Hack for memoryview for py2.7 comptability

from plancklens.helpers import mpi


def adapt_array(arr):
    out = io.BytesIO(); np.save(out, arr)
    out.seek(0)
    return buffer(out.read()) if six.PY2 else memoryview(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)

class npdb:
    """A simple wrapper class to store np arrays in an sqlite3 database.

     """
    def __init__(self, fname, idtype="STRING"):
        if not os.path.exists(fname) and mpi.rank == 0:
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=3600)
            cur = con.cursor()
            cur.execute("CREATE TABLE npdb (id %s PRIMARY KEY, arr ARRAY)" % idtype)
            con.commit()
        mpi.barrier()

        self.con = sqlite3.connect(fname, timeout=3600., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, idx, vec):
        try:
            assert self.get(idx) is None
            self.con.execute("INSERT INTO npdb (id,  arr) VALUES (?,?)", (idx, vec.reshape((1, len(vec)))))
            self.con.commit()
        except:
            print("npdb add failed!")

    def remove(self, idx):
        try:
            assert self.get(idx) is not None
            self.con.execute("DELETE FROM npdb WHERE id=?", (idx,))
            self.con.commit()
        except:
            print("npdb remove failed!")

    def get(self, idx):
        cur = self.con.cursor()
        cur.execute("SELECT arr FROM npdb WHERE id=?", (idx,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            return data[0].flatten()

class fldb:
    """A simple wrapper class to store floats in an sqlite3 database.

     """
    def __init__(self, fname, idtype="STRING"):
        if not os.path.exists(fname) and mpi.rank == 0:
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES, timeout=3600)
            cur = con.cursor()
            cur.execute("CREATE TABLE fldb (id %s PRIMARY KEY, fl REAL)" % idtype)
            con.commit()
        mpi.barrier()

        self.con = sqlite3.connect(fname, timeout=3600., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, idx, fl):
        try:
            assert self.get(idx) is None
            self.con.execute("INSERT INTO fldb (id,  fl) VALUES (?,?)", (idx, fl))
            self.con.commit()
        except:
            print("fldb add failed!")

    def remove(self, idx):
        try:
            assert self.get(idx) is not None
            self.con.execute("DELETE FROM fldb WHERE id=?", (idx,))
            self.con.commit()
        except:
            print("fldb remove failed!")

    def get(self, idx):
        cur = self.con.cursor()
        cur.execute("SELECT fl FROM fldb WHERE id=?", (idx,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            return data[0]
