from __future__ import print_function

import os
import pickle as pk
import healpy as hp
import numpy as np
import plancklens.sims.phas

from plancklens.utils import clhash, hash_check
from plancklens.helpers import mpi
from plancklens.sims import phas, cmbs

class cmb_maps(object):
    r"""CMB simulation library combining a lensed CMB library and a transfer function.

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cl_transf: CMB temperature transfer function
            nside: healpy resolution of the maps. Defaults to 2048.
            lib_dir(optional): hash checks will be cached, as well as possibly other things for subclasses.
            cl_transf_P: CMB pol transfer function (if different from cl_transf)

    """
    def __init__(self, sims_cmb_len, cl_transf, nside=2048, cl_transf_P=None, lib_dir=None):
        if cl_transf_P is None:
            cl_transf_P = np.copy(cl_transf)

        self.sims_cmb_len = sims_cmb_len
        self.cl_transf_T = cl_transf
        self.cl_transf_P = cl_transf_P
        self.nside = nside

        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)

    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,'cl_transf':clhash(self.cl_transf_T)}
        if not (np.all(self.cl_transf_P == self.cl_transf_T)):
            ret['cl_transf_P'] = clhash(self.cl_transf_P)
        return ret

    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        tmap = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tmap,self.cl_transf_T,inplace=True)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_pmap(self,idx):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Q and U healpy maps

        """
        elm = self.sims_cmb_len.get_sim_elm(idx)
        hp.almxfl(elm,self.cl_transf_P,inplace=True)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(blm, self.cl_transf_P, inplace=True)
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
    def __init__(self,sims_cmb_len,cl_transf,nside=2048, cl_transf_P=None):
        super(cmb_maps_noisefree, self).__init__(sims_cmb_len, cl_transf, nside=nside, cl_transf_P=cl_transf_P)

    def get_sim_tnoise(self,idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_qnoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))

    def get_sim_unoise(self, idx):
        return np.zeros(hp.nside2npix(self.nside))

class cmb_maps_nlev(cmb_maps):
    r"""CMB simulation library combining a lensed CMB library, transfer function and idealized homogeneous noise.

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cl_transf: CMB transfer function, identical in temperature and polarization
            nlev_t: temperature noise level in :math:`\mu K`-arcmin
            nlev_p: polarization noise level in :math:`\mu K`-arcmin
            nside: healpy resolution of the maps
            lib_dir(optional): noise maps random phases will be cached there. Only relevant if *pix_lib_phas is not set*
            pix_lib_phas(optional): random phases library for the noise maps (from *plancklens.sims.phas.py*).
                                    If not set, *lib_dir* arg must be set.


    """
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
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(),
                'nside':self.nside,'cl_transf':clhash(self.cl_transf_T),
                'nlev_t':self.nlev_t,'nlev_p':self.nlev_p, 'pixphas':self.pix_lib_phas.hashdict()}
        if not (np.all(self.cl_transf_P == self.cl_transf_T)):
            ret['cl_transf_P'] = clhash(self.cl_transf_P)
        return ret

    def get_sim_tnoise(self,idx):
        """Returns noise temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_t / vamin * self.pix_lib_phas.get_sim(idx, idf=0)

    def get_sim_qnoise(self, idx):
        """Returns noise Q-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=1)

    def get_sim_unoise(self, idx):
        """Returns noise U-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        return self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=2)



class cmb_maps_harmonicspace(object):
    r"""CMB simulation library combining a lensed CMB library and a transfer function

        Note:
            In this version, maps are directly produced in harmonic space with possibly non-white but stat. isotropic noise

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cls_transf: dict with transfer function for 't' 'e' and 'b' CMB fields
            cls_noise: dict with noise spectra for 't' 'e' and 'b'
            noise_phas: *plancklens.sims.phas.lib-phas* with at least 3 fields for the random phase library for the noise generation
            lib_dir(optional): hash checks will be cached, as well as possibly other things for subclasses.
            nside: If provided, maps are returned in pixel space instead of harmonic space

        Note:
            lmax's of len cmbs and noise phases must match


    """
    def __init__(self, sims_cmb_len, cls_transf:dict, cls_noise:dict, noise_phas:plancklens.sims.phas.lib_phas, lib_dir=None, nside=None):
        assert noise_phas.nfields >= 3, noise_phas.nfields
        self.sims_cmb_len = sims_cmb_len
        self.cls_transf = cls_transf
        self.cls_noise = cls_noise
        self.phas = noise_phas
        self.nside = nside

        if hasattr(sims_cmb_len, 'lmax'):
            assert self.sims_cmb_len.lmax == self.phas.lmax, f"Lmax of lensed CMB and of noise phases should match, here {self.sims_cmb_len.lmax} and {self.phas.lmax}"

        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))

    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(), 'phas':self.phas.hashdict()}
        for k in self.cls_noise:
            ret['noise' + k] = clhash(self.cls_noise[k])
        for k in self.cls_transf:
            ret['transf' + k] = clhash(self.cls_transf[k])
        return ret

    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                Temperature alm's 
                or Temperature healpy map if nside is given

        """
        assert 't' in self.cls_transf
        tlm = self.sims_cmb_len.get_sim_tlm(idx)
        hp.almxfl(tlm,self.cls_transf['t'],inplace=True)
        tlm +=  self.get_sim_tnoise(idx)
        if self.nside:
            return hp.alm2map(tlm, self.nside)
        return tlm 

    def get_sim_pmap(self,idx):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Elm and Blm
                 or Q and U healpy maps if nside is given

        """
        assert 'e' in self.cls_transf
        assert 'b' in self.cls_transf

        elm = self.sims_cmb_len.get_sim_elm(idx)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        hp.almxfl(elm, self.cls_transf['e'], inplace=True)
        hp.almxfl(blm, self.cls_transf['b'], inplace=True)
        elm += self.get_sim_enoise(idx)
        blm += self.get_sim_bnoise(idx)
        if self.nside is not None:
            return hp.alm2map_spin([elm,blm], self.nside, 2, hp.Alm.getlmax(elm.size))
        return elm, blm 

    def get_sim_tnoise(self,idx):
        assert 't' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 0), np.sqrt(self.cls_noise['t']))

    def get_sim_enoise(self, idx):
        assert 'e' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 1), np.sqrt(self.cls_noise['e']))

    def get_sim_bnoise(self, idx):
        assert 'b' in self.cls_noise
        return hp.almxfl(self.phas.get_sim(idx, 2), np.sqrt(self.cls_noise['b']))
