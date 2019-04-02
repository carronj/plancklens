from __future__ import print_function

import os
import pickle as pk
import healpy as hp
import numpy as np

from plancklens2018.utils import clhash, hash_check
from plancklens2018 import mpi

class cmb_maps(object):
    def __init__(self,sims_cmb_len,cl_transf,nside=2048,lib_dir=None):
        self.sims_cmb_len = sims_cmb_len
        self.cl_transf = cl_transf
        self.nside = nside
        if lib_dir is not None:
            if mpi.rank == 0 and not os.path.exists(lib_dir + '/sim_hash.pk'):
                pk.dump(self.hashdict(), open(lib_dir + '/sim_hash.pk', 'w'))
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(lib_dir + '/sim_hash.pk', 'r')))

    def hashdict(self):
        return {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,'cl_transf':clhash(self.cl_transf)}

    def get_sim_tmap(self,idx):
        tmap = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tmap,self.cl_transf,inplace=True)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_qumap(self,idx):
        elm = self.sims_cmb_len.get_sim_elm(idx)
        hp.almxfl(elm,self.cl_transf,inplace=True)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(blm, self.cl_transf, inplace=True)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q + self.get_sim_qnoise(idx),U + self.get_sim_unoise(idx)

    def get_sim_tnoise(self,idx):
        assert 0,'subclass this'

    def get_sim_qnoise(self, idx):
        assert 0, 'subclass this'

    def get_sim_unoise(self, idx):
        assert 0, 'subclass this'

class cmb_maps_noisefree(cmb_maps):
    def __init__(self,sims_cmb_len,cl_transf,nside=2048):
        super(cmb_maps_noisefree, self).__init__(sims_cmb_len, cl_transf, nside=nside)

    def get_sim_tnoise(self,idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_qnoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_unoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))





