from __future__ import print_function

import numpy as np
import healpy as hp

from plancklens2018 import utils

verbose = False

def _get_fields(cls):
    if verbose: print(cls.keys())
    fields = ['p', 't', 'e', 'b', 'o']
    ret = ['p', 't', 'e', 'b', 'o']
    for _f in fields:
        if not ((_f + _f) in cls.keys()): ret.remove(_f)
    for _k in cls.keys():
        for _f in _k:
            if _f not in ret: ret.append(_f)
    return ret

class sims_cmb_unl:
    """Unlensed CMB skies simulation library.

    """
    def __init__(self, cls_unl, lib_pha):
        lmax = lib_pha.lmax
        lmin = 0
        fields = _get_fields(cls_unl)
        Nf = len(fields)
        if verbose: print("I see %s fields: " % Nf + " ".join(fields))
        rmat = np.zeros((lmax + 1, Nf, Nf), dtype=float)
        str = ''
        for _i, _t1 in enumerate(fields):
            for _j, _t2 in enumerate(fields):
                if _j >= _i:
                    if _t1 + _t2 in cls_unl.keys():
                        rmat[lmin:, _i, _j] = cls_unl[_t1 + _t2][:lmax + 1]
                        rmat[lmin:, _j, _i] = rmat[lmin:, _i, _j]
                    else:
                        str += " " + _t1 + _t2
        if verbose and str != '': print(str + ' set to zero')
        for ell in range(lmin,lmax + 1):
            t, v = np.linalg.eigh(rmat[ell, :, :])
            assert np.all(t >= 0.), (ell, t, rmat[ell, :, :])  # Matrix not positive semidefinite
            rmat[ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))

        self._cl_hash = {}
        for k in cls_unl.keys():
            self._cl_hash[k] =utils.clhash(cls_unl[k])
        self.rmat = rmat
        self.lib_pha = lib_pha
        self.fields = fields

    def hashdict(self):
        ret = {k : self._cl_hash[k] for k in self._cl_hash.keys()}
        ret['phas'] = self.lib_pha.hashdict()
        return ret

    def _get_sim_alm(self, idx, idf):
        # FIXME : triangularise this
        ret = hp.almxfl(self.lib_pha.get_sim(idx, idf=0), self.rmat[:, idf, 0])
        for _i in range(1,len(self.fields)):
            ret += hp.almxfl(self.lib_pha.get_sim(idx, idf=_i), self.rmat[:, idf, _i])
        return ret

    def get_sim_alm(self, idx, field):
        assert field in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index(field))

    def get_sim_plm(self, idx):
        assert 'p' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('p'))

    def get_sim_olm(self, idx):
        assert 'o' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('o'))

    def get_sim_tlm(self, idx):
        assert 't' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('t'))

    def get_sim_elm(self, idx):
        assert 'e' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('e'))

    def get_sim_blm(self, idx):
        assert 'b' in self.fields, self.fields
        return self._get_sim_alm(idx, self.fields.index('b'))

    def get_sim_alms(self, idx):
        phases = self.lib_pha.get_sim(idx)
        ret = np.zeros_like(phases)
        Nf = len(self.fields)
        for _i in range(Nf):
            for _j in range(Nf):
                ret[_i] += hp.almxfl(phases[_j], self.rmat[:, _i, _j])
