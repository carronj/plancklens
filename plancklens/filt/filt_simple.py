"""simple CMB filtering module.

This module collects a couple of fast (non-iterative) filtering methods.

"""
from __future__ import print_function

import healpy as hp
import numpy  as np
import pickle as pk
import os

from plancklens.helpers import mpi
from plancklens import utils

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
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())

    def hashdict(self):
        assert 0, 'override this'

    def get_fmask(self):
        assert 0, 'override this'

    def _apply_ivf_t(self, tmap, soltn=None):
        assert 0, 'override this'

    def _apply_ivf_p(self, pmap, soltn=None):
        assert 0, 'override this'

    def get_ftl(self):
        """Isotropic approximation to temperature inverse variance filtering  $$N_L$$.

        """
        assert 0, 'override this'

    def get_fel(self):
        """Isotropic approximation to E-polarization inverse variance filtering.

        """
        assert 0, 'override this'

    def get_fbl(self):
        """Isotropic approximation to B-polarization inverse variance filtering.

        """
        assert 0, 'override this'

    def get_tal(self, a):
        assert 0, 'override this'

    def get_sim_tlm(self, idx):
        """Returns an inverse-filtered temperature simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered temperature healpy alm array

        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits'%idx if idx >= 0 else 'dat_tlm.fits')
        if not os.path.exists(tfname):
            tlm = self._apply_ivf_t(self.sim_lib.get_sim_tmap(idx), soltn=None if self.soltn_lib is None else self.soltn_lib.get_sim_tmliklm(idx))
            if self.cache: hp.write_alm(tfname, tlm)
            return tlm
        return hp.read_alm(tfname)

    def get_sim_elm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered E-polarization healpy alm array

        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx  if idx >= 0 else 'dat_elm.fits')
        if not os.path.exists(tfname):
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])
            elm, blm = self._apply_ivf_p(self.sim_lib.get_sim_pmap(idx), soltn=soltn)
            if self.cache:
                hp.write_alm(tfname, elm)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx if idx >= 0 else 'dat_blm.fits'), blm)
            return elm
        else:
            return hp.read_alm(tfname)

    def get_sim_blm(self, idx):
        """Returns an inverse-filtered B-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered B-polarization healpy alm array

        """
        tfname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits'%idx  if idx >= 0 else 'dat_blm.fits')
        if not os.path.exists(tfname):
            if self.soltn_lib is None:
                soltn = None
            else:
                soltn = np.array([self.soltn_lib.get_sim_emliklm(idx), self.soltn_lib.get_sim_bmliklm(idx)])
            elm, blm = self._apply_ivf_p(self.sim_lib.get_sim_pmap(idx), soltn=soltn)
            if self.cache:
                hp.write_alm(tfname, blm)
                hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits'%idx if idx >= 0 else 'dat_elm.fits'), elm)
            return blm
        else:
            return hp.read_alm(tfname)

    def get_sim_tmliklm(self, idx):
        """Returns a Wiener-filtered temperature simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered temperature healpy alm array

        """
        return hp.almxfl(self.get_sim_tlm(idx), self.cl['tt'])

    def get_sim_emliklm(self, idx):
        """Returns a Wiener-filtered E-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered E-polarization healpy alm array

        """
        return hp.almxfl(self.get_sim_elm(idx), self.cl['ee'])

    def get_sim_bmliklm(self, idx):
        """Returns a Wiener-filtered B-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered B-polarization healpy alm array

        """
        return hp.almxfl(self.get_sim_blm(idx), self.cl['bb'])



class library_jTP(object):
    """Template class for CMB inverse-variance and Wiener-filtering library.

    This one is suitable whenever the temperature and polarization maps are jointly filtered.

    Args:
        lib_dir (str): directory where hashes and filtered maps will be cached.
        sim_lib : simulation library instance. *sim_lib* must have *get_sim_tmap* and *get_sim_pmap* methods.
        cl_weights: CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.

    """
    def __init__(self, lib_dir, sim_lib, cl_weights, soltn_lib=None, cache=True):

        assert np.all([k in cl_weights.keys() for k in ['tt', 'ee', 'bb']])
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
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())

    def hashdict(self):
        assert 0, 'override this'

    def get_fmask(self):
        assert 0, 'override this'

    def _apply_ivf(self, tqumap, soltn=None):
        assert 0, 'override this'


    def get_fal(self):
        """Isotropic matrix approximation to temperature inverse variance filtering

            :math:`F_\ell \sim (C_\ell + N_\ell / b_\ell^2)^{-1}`

        """
        assert 0, 'override this'

    def _get_alms(self, a, idx):
        assert a in ['t', 'e', 'b']
        tfname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits' % idx if idx >= 0 else 'dat_tlm.fits')
        fname = tfname.replace('tlm.fits', a + 'lm.fits')
        if not os.path.exists(fname):
            T = self.sim_lib.get_sim_tmap(idx)
            Q, U = self.sim_lib.get_sim_pmap(idx)
            if self.soltn_lib is None:
                soltn = None
            else:
                tlm = self.soltn_lib.get_sim_tmliklm(idx)
                elm = self.soltn_lib.get_sim_emliklm(idx)
                blm = self.soltn_lib.get_sim_bmliklm(idx)
                soltn = (tlm, elm, blm)
            tlm, elm, blm = self._apply_ivf([T, Q, U],  soltn=soltn)
            if self.cache:
                hp.write_alm(tfname.replace('tlm.fits', 'tlm.fits'), tlm)
                hp.write_alm(tfname.replace('tlm.fits', 'elm.fits'), elm)
                hp.write_alm(tfname.replace('tlm.fits', 'blm.fits'), blm)
        return hp.read_alm(fname)

    def get_sim_tlm(self, idx):
        """Returns an inverse-filtered temperature simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered temperature healpy alm array

        """
        return self._get_alms('t', idx)

    def get_sim_elm(self, idx):
        """Returns an inverse-filtered E-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered E-polarization healpy alm array

        """
        return self._get_alms('e', idx)


    def get_sim_blm(self, idx):
        """Returns an inverse-filtered B-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                inverse-filtered B-polarization healpy alm array

        """
        return self._get_alms('b', idx)


    def get_sim_tmliklm(self, idx):
        """Returns a Wiener-filtered temperature simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered temperature healpy alm array

        """
        ret = hp.almxfl(self.get_sim_tlm(idx), self.cl['tt'])
        for k in ['te', 'tb']:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret

    def get_sim_emliklm(self, idx):
        """Returns a Wiener-filtered E-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered E-polarization healpy alm array

        """
        ret = hp.almxfl(self.get_sim_elm(idx), self.cl['ee'])
        for k in ['et', 'eb']:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret

    def get_sim_bmliklm(self, idx):
        """Returns a Wiener-filtered B-polarization simulation.

            Args:
                idx: simulation index

            Returns:
                Wiener-filtered B-polarization healpy alm array

        """
        ret = hp.almxfl(self.get_sim_blm(idx), self.cl['bb'])
        for k in ['bt', 'be']:
            cl = self.cl.get(k[0] + k[1], self.cl.get(k[1] + k[0], None))
            if cl is not None:
                ret += hp.almxfl(self._get_alms(k[1], idx), cl)
        return ret


class library_fullsky_sepTP(library_sepTP):
    """Full-sky isotropic filtering instance.

    Args:
        lib_dir: directory where hashes and filtered maps will be cached.
        sim_lib: simulation library instance to inverse-filter
        nside: healpix resolution of the simulation library
        cl_len : CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.
        transf : fiducial transfer function of the CMB maps. (if dict, then must have keys 't' 'e' and 'b' for individual transfer functions)
        ftl (1d-array): isotropic filtering array for temperature (filtered tlm's are ftl * tlm of the data)
        fel (1d-array): isotropic filtering array for E-pol. (filtered elm's are fel * elm of the data)
        fbl (1d-array): isotropic filtering array for B-po. (filtered blm's are fbl * blm of the data)
        cache: filtered alm's will be cached if set.

    """
    def __init__(self, lib_dir, sim_lib, nside, transf:np.ndarray or dict, cl_len, ftl, fel, fbl, cache=False):

        transfd = transf if isinstance(transf, dict) else {'t': transf, 'e': transf, 'b': transf}
        assert 't' in transfd.keys() and 'e' in transfd.keys() and 'b' in transfd.keys()

        self.sim_lib = sim_lib
        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.nside = nside
        self.transf = transfd

        super(library_fullsky_sepTP, self).__init__(lib_dir, sim_lib, cl_len, cache=cache)

    def hashdict(self):
        return {'sim_lib':self.sim_lib.hashdict(), 'transf': utils.clhash(self.transf['t']),
                'cl_len': {k: utils.clhash(self.cl[k]) for k in ['tt', 'ee', 'bb']},
                'ftl': utils.clhash(self.ftl), 'fel': utils.clhash(self.fel), 'fbl': utils.clhash(self.fbl)}

    def get_fmask(self):
        return np.ones(hp.nside2npix(self.nside), dtype=float)

    def get_tal(self, a):
        assert (a.lower() in ['t', 'e', 'b'])
        return utils.cli(self.transf[a.lower()])

    def get_ftl(self):
        return np.copy(self.ftl)

    def get_fel(self):
        return np.copy(self.fel)

    def get_fbl(self):
        return np.copy(self.fbl)

    def _apply_ivf_t(self, tmap, soltn=None):
        assert len(tmap) == hp.nside2npix(self.nside), (hp.npix2nside(tmap.size), self.nside)
        alm = hp.map2alm(tmap, lmax=self.lmax_fl, iter=0)
        return hp.almxfl(alm, self.get_ftl() * utils.cli(self.transf['t'][:len(self.ftl)]))

    def _apply_ivf_p(self, pmap, soltn=None):
        assert len(pmap[0]) == hp.nside2npix(self.nside) and len(pmap[0]) == len(pmap[1])
        elm, blm = hp.map2alm_spin([m for m in pmap], 2, lmax=self.lmax_fl)
        elm = hp.almxfl(elm, self.get_fel() * utils.cli(self.transf['e'][:len(self.fel)]))
        blm = hp.almxfl(blm, self.get_fbl() * utils.cli(self.transf['b'][:len(self.fbl)]))
        return elm, blm

class library_fullsky_alms_sepTP(library_sepTP):
    """Full-sky isotropic filtering instance, but with harmonic space inputs

    Args:
        lib_dir: directory where hashes and filtered maps will be cached.
        sim_lib: simulation library instance to inverse-filter
        cl_len : CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.
        transf : fiducial transfer function of the CMB maps. (if dict, then must have keys 't' 'e' and 'b' for individual transfer functions)
        ftl (1d-array): isotropic filtering array for temperature (filtered tlm's are ftl * tlm of the data)
        fel (1d-array): isotropic filtering array for E-pol. (filtered elm's are fel * elm of the data)
        fbl (1d-array): isotropic filtering array for B-po. (filtered blm's are fbl * blm of the data)
        cache: filtered alm's will be cached if set.

    """
    def __init__(self, lib_dir, sim_lib, transf:np.ndarray or dict, cl_len, ftl, fel, fbl, cache=False):

        transfd = transf if isinstance(transf, dict) else {'t': transf, 'e': transf, 'b': transf}
        assert 't' in transfd.keys() and 'e' in transfd.keys() and 'b' in transfd.keys()

        self.sim_lib = sim_lib
        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.transf = transfd

        super(library_fullsky_alms_sepTP, self).__init__(lib_dir, sim_lib, cl_len, cache=cache)

    def hashdict(self):
        return {'sim_lib':self.sim_lib.hashdict(), 'transf': utils.clhash(self.transf['t']),
                'cl_len': {k: utils.clhash(self.cl[k]) for k in ['tt', 'ee', 'bb']},
                'ftl': utils.clhash(self.ftl), 'fel': utils.clhash(self.fel), 'fbl': utils.clhash(self.fbl)}

    def get_fmask(self):
        return np.array([1.]) # For compatibility purposes only...

    def get_tal(self, a):
        assert (a.lower() in ['t', 'e', 'b'])
        return utils.cli(self.transf[a.lower()])

    def get_ftl(self):
        return np.copy(self.ftl)

    def get_fel(self):
        return np.copy(self.fel)

    def get_fbl(self):
        return np.copy(self.fbl)

    def _apply_ivf_t(self, tlm, soltn=None):
        return hp.almxfl(tlm, self.get_ftl() * utils.cli(self.transf['t'][:len(self.ftl)]))

    def _apply_ivf_p(self, eblm, soltn=None):
        elm = hp.almxfl(eblm[0], self.get_fel() * utils.cli(self.transf['e'][:len(self.fel)]))
        blm = hp.almxfl(eblm[1], self.get_fbl() * utils.cli(self.transf['b'][:len(self.fbl)]))
        return elm, blm


class library_apo_sepTP(library_sepTP):
    """
    Library to perform inverse variance filtering on the sim_lib library using simple mask apo and isotropic filtering.

    Args:
        lib_dir: directory where hashes and filtered maps will be cached.
        sim_lib: simulation library instance to inverse-filter
        apomask_path : path of the (presumably apodized) mask
        cl_len : CMB spectra, used to compute the Wiener-filtered CMB from the inverse variance filtered maps.
        transf : fiducial transfer function of the CMB maps.
        ftl (1d-array): isotropic filtering array for temperature (filtered tlm's are ftl * tlm of the data)
        fel (1d-array): isotropic filtering array for E-pol. (filtered elm's are fel * elm of the data)
        fbl (1d-array): isotropic filtering array for B-po. (filtered blm's are fbl * blm of the data)
        cache: filtered alm's will be cached if set.

    """
    def __init__(self, lib_dir, sim_lib, apomask_path, cl_len, transf, ftl, fel, fbl, cache=False):
        assert len(transf) >= np.max([len(ftl), len(fel), len(fbl)])
        assert np.all([k in cl_len.keys() for k in ['tt', 'ee', 'bb']])
        assert os.path.exists(apomask_path)

        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.transf = transf
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.apomask_path = apomask_path
        self.nside = hp.npix2nside(hp.read_map(apomask_path).size)
        super(library_apo_sepTP, self).__init__(lib_dir, sim_lib, cl_len, cache=cache)

    def hashdict(self):
        return {'sim_lib':self.sim_lib.hashdict(),
                'apomask': self.apomask_path, 'transf': utils.clhash(self.transf),
                'cl_len': {k: utils.clhash(self.cl[k]) for k in ['tt', 'ee', 'bb']},
                'ftl': utils.clhash(self.ftl), 'fel': utils.clhash(self.fel), 'fbl': utils.clhash(self.fbl)}

    def get_fmask(self):
        return hp.read_map(self.apomask_path)

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
        assert len(tmap) == hp.nside2npix(self.nside), (hp.npix2nside(tmap.size), self.nside)
        alm = hp.map2alm(tmap * self.get_fmask(), lmax=self.lmax_fl, iter=0)
        return hp.almxfl(alm, self.get_ftl() * utils.cli(self.transf[:len(self.ftl)]))

    def _apply_ivf_p(self, pmap, soltn=None):
        assert len(pmap[0]) == hp.nside2npix(self.nside) and len(pmap[0]) == len(pmap[1])
        elm, blm = hp.map2alm_spin([m * self.get_fmask() for m in pmap], 2, lmax=self.lmax_fl)
        elm = hp.almxfl(elm, self.get_fel() * utils.cli(self.transf[:len(self.fel)]))
        blm = hp.almxfl(blm, self.get_fbl() * utils.cli(self.transf[:len(self.fbl)]))
        return elm, blm