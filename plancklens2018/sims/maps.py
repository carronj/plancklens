from __future__ import print_function

import os
import pickle as pk
import healpy as hp
import numpy as np

from plancklens2018.utils import clhash, hash_check
from plancklens2018.helpers import mpi
from plancklens2018.sims import phas

class cmb_maps(object):
    def __init__(self,sims_cmb_len,cl_transf,nside=2048,lib_dir=None):
        self.sims_cmb_len = sims_cmb_len
        self.cl_transf = cl_transf
        self.nside = nside
        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists():
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))

    def hashdict(self):
        return {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,'cl_transf':clhash(self.cl_transf)}

    def get_sim_tmap(self,idx):
        tmap = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tmap,self.cl_transf,inplace=True)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_pmap(self,idx):
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

class cmb_maps_nlev(cmb_maps):
    def __init__(self,sims_cmb_len, cl_transf, nlev_t, nlev_p, nside, lib_dir=None, pix_lib_phas=None):
        if pix_lib_phas is None:
            assert lib_dir is not None
            pix_lib_phas = phas.pix_lib_phas(lib_dir, 3, (hp.nside2npix(nside),))
        assert pix_lib_phas.shape == (hp.nside2npix(nside),), (pix_lib_phas.shape, (hp.nside2npix(nside),))
        self.pix_lib_phas = pix_lib_phas
        self.nlev_t = nlev_t
        self.nlev_p = nlev_p

        super(cmb_maps_nlev, self).__init__(sims_cmb_len, cl_transf, nside=nside, lib_dir=lib_dir)


    def hashdict(self):
        return {'sims_cmb_len':self.sims_cmb_len.hashdict(),
                'nside':self.nside,'cl_transf':clhash(self.cl_transf),
                'nlev_t':self.nlev_t,'nlev_p':self.nlev_p, 'pixphas':self.pix_lib_phas.hashdict()}

    def get_sim_tnoise(self,idx):
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_t / vamin * self.pix_lib_phas.get_sim(idx, idf=0)

    def get_sim_qnoise(self, idx):
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=1)

    def get_sim_unoise(self, idx):
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=2)




