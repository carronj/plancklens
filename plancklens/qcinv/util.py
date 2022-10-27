from __future__ import print_function

import time
import numpy  as np
import healpy as hp
from plancklens import utils

class dt:
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                    np.floor(np.mod(self.dt, 60 * 60) / 60),
                                    np.floor(np.mod(self.dt, 60))))

    def __int__(self):
        return int(self.dt)


class stopwatch:
    def __init__(self):
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        lt = time.time()
        ret = (dt(lt - self.st), dt(lt - self.lt))
        self.lt = lt
        return ret

    def elapsed(self):
        lt = time.time()
        ret = dt(lt - self.st)
        self.lt = lt
        return ret


class jit:
    """ just-in-time instantiation wrapper class.

    """
    def __init__(self, ctype, *cargs, **ckwds):
        self.__dict__['__jit_args'] = [ctype, cargs, ckwds]
        self.__dict__['__jit_obj'] = None

    def instantiate(self):
        [ctype, cargs, ckwds] = self.__dict__['__jit_args']
        print('jit: instantiating ctype =', ctype)
        self.__dict__['__jit_obj'] = ctype(*cargs, **ckwds)
        del self.__dict__['__jit_args']

    def __getattr__(self, attr):
        if self.__dict__['__jit_obj'] is None:
            self.instantiate()
        return getattr(self.__dict__['__jit_obj'], attr)

    def __setattr__(self, attr, val):
        if self.__dict__['__jit_obj'] is None:
            self.instantiate()
        setattr(self.__dict__['__jit_obj'], attr, val)

def read_map(m):
    """Reads a map whether given as (list of) string (with ',f' denoting field f), array or callable

    """
    if callable(m):
        return m()
    if isinstance(m, list):
        ma = read_map(m[0])
        for m2 in m[1:]:
            ma = ma * read_map(m2)
        return ma
    if not isinstance(m, str):
        return m
    if ',' not in m:
        return hp.read_map(m)
    m, field = m.split(',')
    return hp.read_map(m, field=int(field))

def mask_hash(m, dtype=bool):
    if m is None:
        return "none"
    if isinstance(m, list):
        mh = mask_hash(m[0], dtype=dtype)
        for m2 in m[1:]:
            mh += mask_hash(m2, dtype=dtype)
        return mh
    if isinstance(m, str):
        return m.replace('/','_sl_').replace('.', '_')
    elif isinstance(m, np.ndarray):
        return utils.clhash(m, dtype=dtype)
    elif callable(m):
        return 'callable'
    assert 0, 'not implemented'

def load_map(f): # Same as read_map
    return read_map(f)
