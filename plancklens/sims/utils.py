import numpy as np

class sim_lib_shuffle:
    """A simulation library with remapped indices.

    """

    def __init__(self, sim_lib, shuffle_dict):
        self.sim_lib = sim_lib
        self._shuffle = shuffle_dict

    def get_sim_tmap(self, idx): return self.sim_lib.get_sim_tmap(int(self._shuffle[idx]))

    def get_sim_pmap(self, idx): return self.sim_lib.get_sim_pmap(int(self._shuffle[idx]))

    def hashdict(self):
        return {'sim_lib': self.sim_lib.hashdict(), 'shuffle': self._shuffle}


class sim_lib_add_sim:
    """Added simulation libraries.

        Addition only for sim (>= 0) indices.

    """
    def __init__(self, sim_libs, weights=None):
        self.w = weights if weights is not None else np.ones(len(sim_libs))
        self.sim_libs = sim_libs

    def get_sim_tmap(self, idx):
        t = self.sim_libs[0].get_sim_tmap(idx) * self.w[0]
        if idx >= 0:
            for s, w in zip(self.sim_libs[1:], self.w[1:]):
                t += s.get_sim_tmap(idx) * w
        return t

    def get_sim_pmap(self, idx):
        q, u = self.sim_libs[0].get_sim_pmap(idx)
        q *= self.w[0]
        u *= self.w[0]
        if idx >= 0:
            for s, w in zip(self.sim_libs[1:], self.w[1:]):
                _q, _u = s.get_sim_pmap(idx)
                q += w * _q
                u += w * _u
        return q, u

    def hashdict(self):
        ret = {'lib': 'add_sim'}
        for i, (s, w) in enumerate(zip(self.sim_libs, self.w)):
            ret['sim_lib ' + str(i)] = s.hashdict()
            ret['w ' + str(i)] = w
        return ret



class sim_lib_add_dat:
    """Added simulation libraries.

        Addition only for data (< 0) indices.

    """
    def __init__(self, sim_libs, weights=None):
        self.w = weights if weights is not None else np.ones(len(sim_libs))
        self.sim_libs = sim_libs

    def get_sim_tmap(self, idx):
        t = self.sim_libs[0].get_sim_tmap(idx) * self.w[0]
        if idx < 0:
            for s, w in zip(self.sim_libs[1:], self.w[1:]):
                t += s.get_sim_tmap(idx) * w
        return t

    def get_sim_pmap(self, idx):
        q, u = self.sim_libs[0].get_sim_pmap(idx)
        q *= self.w[0]
        u *= self.w[0]
        if idx < 0:
            for s, w in zip(self.sim_libs[1:], self.w[1:]):
                _q, _u = s.get_sim_pmap(idx)
                q += w * _q
                u += w * _u
        return q, u

    def hashdict(self):
        ret = {'lib': 'add_dat'}
        for i, (s, w) in enumerate(zip(self.sim_libs, self.w)):
            ret['sim_lib ' + str(i)] = s.hashdict()
            ret['w ' + str(i)] = w
        return ret





