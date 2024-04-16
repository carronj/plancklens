from __future__ import print_function

import os
import numpy as np
import healpy as hp
import pickle as pk

from plancklens import utils
from plancklens.helpers import mpi
from plancklens.sims import phas

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
    def __init__(self, cls_unl, lib_pha:phas.lib_phas):
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
        self.lmax = lmax
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


class sims_cmb_len:
    """Lensed CMB skies simulation library.

        Note:
            To produce the lensed CMB, the package lenspyx is mandatory

        Note:
            These sims do not contain aberration or modulation

        Args:
            lib_dir: lensed cmb alms will be cached there
            lmax: lensed cmbs are produced up to lmax
            cls_unl(dict): unlensed cmbs power spectra
            lib_pha(optional): random phases library for the unlensed maps (see *plancklens.sims.phas*)
            dlmax(defaults to 1024): unlensed cmbs are produced up to lmax + dlmax, for accurate lensing at lmax
            nside_lens(defaults to 4096): healpy resolution at which the lensed maps are produced
            facres(defaults to 0): sets the interpolation resolution in lenspyx
            nbands(defaults to 16): number of band-splits in *lenspyx.alm2lenmap(_spin)*
            verbose(defaults to True): lenspyx timing info printout

    """
    def __init__(self, lib_dir, lmax, cls_unl, lib_pha=None,
                 dlmax=1024, nside_lens=4096, facres=0, nbands=16, verbose=True):
        if not os.path.exists(lib_dir) and mpi.rank == 0:
            os.makedirs(lib_dir)
        mpi.barrier()
        fields = _get_fields(cls_unl)

        if lib_pha is None and mpi.rank == 0:
            lib_pha = phas.lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lmax + dlmax)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lmax == lmax + dlmax
        mpi.barrier()

        self.lmax = lmax
        self.dlmax = dlmax
        # lenspyx parameters:
        self.nside_lens = nside_lens
        self.nbands = nbands
        self.facres = facres

        self.unlcmbs = sims_cmb_unl(cls_unl, lib_pha)
        self.lib_dir = lib_dir
        self.fields = _get_fields(cls_unl)

        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if mpi.rank == 0 and not os.path.exists(fn_hash) :
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)
        try:
            import lenspyx
        except ImportError:
            print("Could not import lenspyx module")
            lenspyx = None
        self.lens_module = lenspyx
        self.verbose=verbose

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,
                'nside_lens':self.nside_lens, 'facres':self.facres}

    def _is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'e':
            return self.get_sim_elm(idx)
        elif field == 'b':
            return self.get_sim_blm(idx)
        elif field == 'p':
            return self.get_sim_plm(idx)
        elif field == 'o':
            return self.get_sim_olm(idx)
        else :
            assert 0,(field,self.fields)

    def get_sim_plm(self, idx):
        return self.unlcmbs.get_sim_plm(idx)

    def get_sim_olm(self, idx):
        return self.unlcmbs.get_sim_olm(idx)

    def _cache_eblm(self, idx):
        elm = self.unlcmbs.get_sim_elm(idx)
        blm = None if 'b' not in self.fields else self.unlcmbs.get_sim_blm(idx)
        dlm = self.get_sim_plm(idx)
        assert 'o' not in self.fields, 'not implemented'

        lmaxd = hp.Alm.getlmax(dlm.size)
        hp.almxfl(dlm, np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2)), inplace=True)
        # Qlen, Ulen = self.lens_module.alm2lenmap_spin([elm, blm], [dlm, None], self.nside_lens, 2,
        #                                         nband=self.nbands, facres=self.facres, verbose=self.verbose)
        geom_info = ('healpix', {'nside':self.nside_lens})
        Qlen, Ulen = self.lens_module.alm2lenmap_spin([elm, blm], dlm, 2, geometry=geom_info, verbose=1)
        elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=self.lmax)
        del Qlen, Ulen
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx), elm)
        del elm
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx), blm)

    def get_sim_tlm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits' % idx)
        if not os.path.exists(fname):
            tlm= self.unlcmbs.get_sim_tlm(idx)
            dlm = self.get_sim_plm(idx)
            assert 'o' not in self.fields, 'not implemented'

            lmaxd = hp.Alm.getlmax(dlm.size)
            hp.almxfl(dlm, np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2)), inplace=True)
            # Tlen = self.lens_module.alm2lenmap(tlm, [dlm, None], self.nside_lens,
            #                                    facres=self.facres, nband=self.nbands, verbose=self.verbose)
            geom_info = ('healpix', {'nside':self.nside_lens}) 
            Tlen = self.lens_module.alm2lenmap(tlm, dlm,  geometry=geom_info, verbose=1)
            hp.write_alm(fname, hp.map2alm(Tlen, lmax=self.lmax, iter=0))
        return hp.read_alm(fname)

    def get_sim_elm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)

    def get_sim_blm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)


class sims_cmb_unl_fixed_phi(sims_cmb_unl):
    """Simumaltion library for unlensed CMB with fixed lensing potential field.
    
    By default the lensing potential field is the one from the simulation index 0.
    """

    def __init__(self, cls_unl, lib_pha, plm=None):
        super(sims_cmb_unl_fixed_phi, self).__init__(cls_unl, lib_pha) 
        
        if plm is None: 
            self.fixed_plm = super(sims_cmb_unl_fixed_phi, self)._get_sim_alm(0, self.fields.index('p'))
        else:
            self.fixed_plm = plm


    def _get_sim_alm(self, idx, idf):
        
        if idf == self.fields.index('p'):
            ret = self.fixed_plm
        else:
            ret = hp.almxfl(self.lib_pha.get_sim(idx, idf=0), self.rmat[:, idf, 0])
            for _i in range(1,len(self.fields)):
                ret += hp.almxfl(self.lib_pha.get_sim(idx, idf=_i), self.rmat[:, idf, _i])
            
        return ret


class sims_cmb_len_fixed_phi(sims_cmb_len):
    """Simumaltion library for lensed CMB with fixed lensing potential field.
    """

    def __init__(self, lib_dir, lmax, cls_unl, plm=None, lib_pha=None,
                 dlmax=1024, nside_lens=4096, facres=0, nbands=16, verbose=True):

        fields = _get_fields(cls_unl)
        if lib_pha is None and mpi.rank == 0:
            lib_pha = phas.lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lmax + dlmax)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lmax == lmax + dlmax
        mpi.barrier()
        
        super(sims_cmb_len_fixed_phi, self).__init__(lib_dir, lmax, cls_unl, lib_pha,
                 dlmax, nside_lens, facres, nbands, verbose)         

        self.unlcmbs = sims_cmb_unl_fixed_phi(cls_unl, lib_pha, plm)
