"""simple CMB filtering module.

This module collects a couple of fast (non-iterative) filtering methods.

Todo:
    * Full-sky isotropic
    * doc
    * libaml missing
"""
from __future__ import print_function

import healpy as hp
import numpy  as np
import pickle as pk
import os


from plancklens2018 import mpi
from plancklens2018 import utils

#FIXME: problems with different lmax in ftl, fel, fbl ?

class library_sepTP(object):
    """Template class for CMB inverse-variance and Wiener-filtering library.

    This is suitable whenever the temperature and polarization maps are independently filtered.

    Args:
        lib_dir (str): directory where hashes and filtered maps will be cached.
        sim_lib : simulation library instance. *sim_lib* must have *get_sim_tmap* and *get_sim_pmap* methods.
        cl_weights: CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.

    """
    def __init__(self, lib_dir, sim_lib, cl_weights, soltn_lib=None, cache=True):


        self.lib_dir = lib_dir
        self.sim_lib = sim_lib
        self.cl = cl_weights
        self.soltn_lib = soltn_lib
        self.cache = cache
        fn_hash = os.path.join(lib_dir, 'filt_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'))
        mpi.barrier()
        utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())

    def hashdict(self):
        assert 0, 'override this'

    def _apply_ivf_t(self, tmap, soltn=None):
        assert 0, 'override this'

    def _apply_ivf_p(self, pmap, soltn=None):
        assert 0, 'override this'

    def get_ftl(self):
        """ Isotropic approximation to temperature inverse variance filtering  $$N_L$$.

        Note:
            bla $$ ( C_\\ell^{TT} + N^{TT}_\\ell / b^2_\\ell ) $$

        """
        assert 0, 'override this'

    def get_fel(self):
        """ Isotropic approximation to E-polarization inverse variance filtering.

        This typically has the form $ 1 / \left( C_\ell^{\rm EE} + N^{\rm EE}_\ell / b^2_\ell \right) $

        """
        assert 0, 'override this'

    def get_fbl(self):
        """ Isotropic approximation to B-polarization inverse variance filtering.

        $$ N_\\ell $$
        """
        assert 0, 'override this'

    def get_tal(self, a):
        assert 0, 'override this'

    def get_sim_tlm(self, idx):
        tfname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits'%idx if idx >= 0 else 'dat_tlm.fits')
        if not os.path.exists(tfname):
            tlm = self._apply_ivf_t(self.sim_lib.get_sim_tmap(idx), soltn=None if self.soltn_lib is None else self.soltn_lib.get_sol_tlm(idx))
            if self.cache: hp.write_alm(tfname, tlm)
            return tlm
        return hp.read_alm(tfname)

    def get_sim_elm(self, idx):
        tfname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx  if idx >= 0 else 'dat_elm.fits')
        if not os.path.exists(tfname):
            elm, blm = self._apply_ivf_p(self.sim_lib.get_sim_pmap(idx))
            if self.cache:
                hp.write_alm(tfname, elm)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx if idx >= 0 else 'dat_blm.fits'), blm)
            return elm
        else:
            return hp.read_alm(tfname)

    def get_sim_blm(self, idx):
        tfname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx  if idx >= 0 else 'dat_blm.fits')
        if not os.path.exists(tfname):
            elm, blm = self._apply_ivf_p(self.sim_lib.get_sim_pmap(idx))
            if self.cache:
                hp.write_alm(tfname, blm)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx if idx >= 0 else 'dat_elm.fits'), elm)
            return blm
        else:
            return hp.read_alm(tfname)

    def get_sim_tmliklm(self, idx):
        return hp.almxfl(self.get_sim_tlm(idx), self.cl['tt'])

    def get_sim_emliklm(self, idx):
        return hp.almxfl(self.get_sim_elm(idx), self.cl['ee'])

    def get_sim_bmliklm(self, idx):
        return hp.almxfl(self.get_sim_blm(idx), self.cl['bb'])

class library_apo_sepTP(library_sepTP):
    """
    Library to perform inverse variance filtering on the sim_lib library using simple mask apo and isotropic filtering.

    Note:
        This uses independent T and Pol. filtering.

    Args:
        lib_dir :
        sim_lib :
        masknoapo_path :
        cl_len :
        transf :
        ftl (1d-array):
        fel (1d-array):
        fbl (1d-array):
    """
    def __init__(self, lib_dir, sim_lib, masknoapo_path, cl_len, transf, ftl, fel, fbl, cache=False):
        assert len(transf) >= np.max([len(ftl), len(fel), len(fbl)])
        assert np.all([k in cl_len.keys() for k in ['tt', 'ee', 'bb']])
        
        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.transf = transf
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.masknoapo_path = masknoapo_path
        super(library_apo_sepTP, self).__init__(lib_dir, sim_lib, cl_len, cache=cache)

        if mpi.rank == 0:
            if not os.path.exists(os.path.join(self.lib_dir, 'fmask.fits')):
                #FIXME:
                from libaml import apodize
                apomask = apodize.apodize_mask(hp.read_map(masknoapo_path))
                hp.write_map(os.path.join(self.lib_dir, 'fmask.fits'), apomask)
        mpi.barrier()

    def hashdict(self):
        return {'sim_lib':self.sim_lib.hashdict(),
                'masknoapo': self.masknoapo_path, 'transf': utils.clhash(self.transf),
                'cl_len': {k: utils.clhash(self.cl[k]) for k in ['tt', 'ee', 'bb']},
                'ftl': utils.clhash(self.ftl),'fel': utils.clhash(self.fel),'fbl': utils.clhash(self.fbl)}

    def get_fmask(self):
        return hp.read_map(os.path.join(self.lib_dir, 'fmask.fits'))

    def get_tal(self, a):
        assert (a.lower() in ['t', 'e', 'b'])
        return utils.cli(self.transf)

    def get_ftl(self):
        return np.copy(self.ftl)

    def get_fel(self):
        return np.copy(self.fel)

    def get_fbl(self):
        return np.copy(self.fbl)

    def _apply_ivf_t(self, tmap, soltn=None):
        alm = hp.map2alm(tmap * self.get_fmask(), lmax=self.lmax_fl)
        return hp.almxfl(alm, self.get_ftl() * utils.cli(self.transf[:len(self.ftl)]))

    def _apply_ivf_p(self, pmap, soltn=None):
        elm, blm = hp.map2alm_spin([m * self.get_fmask() for m in pmap], 2, lmax=self.lmax_fl)
        elm = hp.almxfl(elm, self.get_fel() * utils.cli(self.transf[:len(self.fel)]))
        blm = hp.almxfl(blm, self.get_fbl() * utils.cli(self.transf[:len(self.fbl)]))
        return elm, blm

class library_fullsky_sepTP(library_sepTP):
    """Full-sky isotropic filtering instance.

    Note:
        This uses independent T and Pol. filtering.

    """
    def __init__(self, lib_dir, sim_lib, transf, cl_len, ftl, fel, fbl, cache=False):
        self.sim_lib = sim_lib
        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.transf = transf
        super(library_fullsky_sepTP, self).__init__(lib_dir, sim_lib, cl_len, cache=cache)

    def hashdict(self):
        return {'sim_lib':self.sim_lib.hashdict(), 'transf': utils.clhash(self.transf),
                'cl_len': {k: utils.clhash(self.cl[k]) for k in ['tt', 'ee', 'bb']},
                'ftl': utils.clhash(self.ftl),'fel': utils.clhash(self.fel),'fbl': utils.clhash(self.fbl)}

    def get_tal(self, a):
        assert (a.lower() in ['t', 'e', 'b'])
        return utils.cli(self.transf)

    def get_ftl(self):
        return np.copy(self.ftl)

    def get_fel(self):
        return np.copy(self.fel)

    def get_fbl(self):
        return np.copy(self.fbl)

    def _apply_ivf_t(self, tmap, soltn=None):
        alm = hp.map2alm(tmap, lmax=self.lmax_fl)
        return hp.almxfl(alm, self.get_ftl() * utils.cli(self.transf[:len(self.ftl)]))

    def _apply_ivf_p(self, pmap, soltn=None):
        elm, blm = hp.map2alm_spin([m for m in pmap], 2, lmax=self.lmax_fl)
        elm = hp.almxfl(elm, self.get_fel() * utils.cli(self.transf[:len(self.fel)]))
        blm = hp.almxfl(blm, self.get_fbl() * utils.cli(self.transf[:len(self.fbl)]))
        return elm, blm
