from __future__ import print_function

import healpy as hp
import numpy  as np
import pickle as pk
import os

from libaml import apodize

from . import mpi
from . import utils

class library_sepTP(object):
    """ Template filtering library, where temperature and polarization are filtered independently """
    def __init__(self, lib_dir, sim_lib, soltn_lib=None, cache=True):
        self.sim_lib = sim_lib
        self.lib_dir = lib_dir
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
        return {'sim_lib': self.sim_lib.hashdict()}

    def get_sim_tmliklm(self, idx):
        assert 0, 'override this'

    def get_sim_emliklm(self, idx):
        assert 0, 'override this'

    def get_sim_bmliklm(self, idx):
        assert 0, 'override this'

    def _apply_ivf_t(self, tmap, soltn=None):
        assert 0, 'override this'
        return None

    def _apply_ivf_p(self, pmap, soltn=None):
        assert 0, 'override this'
        return None, None

    def get_ftl(self):
        assert 0, 'override this'

    def get_fel(self):
        assert 0, 'override this'

    def get_fbl(self):
        assert 0, 'override this'

    def get_tal(self, a):
        assert 0, 'override this'

    def get_sim_tlm(self, idx):
        """
        Apply inverse_variance filtering to a simulated map.
        If the fits file map is not found it will generate the simulation from the sim_lib library instance.
        Uses -1 for data.
        """
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


class library_apo_sepTP(library_sepTP):
    """
    Library to perform inverse variance filtering on the sim_lib library using simple mask apo and isotropic filtering.
    Separate T and Pol. filtering.
    """
    def __init__(self, lib_dir, sim_lib, masknoapo_path, cl_len, transf, ftl, fel, fbl, cache=False):
        assert len(transf) >= np.max([len(ftl), len(fel), len(fbl)])
        assert np.all([k in cl_len.keys() for k in ['tt', 'ee', 'bb']])
        
        self.cl = cl_len
        self.ftl = ftl
        self.fel = fel
        self.fbl = fbl
        self.transf = transf
        self.lmax_fl = np.max([len(ftl), len(fel), len(fbl)]) - 1
        self.masknoapo_path = masknoapo_path
        super(library_apo_sepTP, self).__init__(lib_dir, sim_lib, cache=cache)

        if mpi.rank == 0:
            if not os.path.exists(os.path.join(self.lib_dir, 'fmask.fits')):
                apomask = apodize.apodize_mask(hp.read_map(masknoapo_path))
                hp.write_map(os.path.join(self.lib_dir, 'fmask.fits'), apomask)
        mpi.barrier()

    def hashdict(self):
        return {'masknoapo': self.masknoapo_path, 'transf': utils.clhash(self.transf),
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

    def get_sim_tmliklm(self, idx):
        return hp.almxfl(self.get_sim_tlm(idx), self.cl['tt'])

    def get_sim_emliklm(self, idx):
        return hp.almxfl(self.get_sim_elm(idx), self.cl['ee'])

    def get_sim_bmliklm(self, idx):
        return hp.almxfl(self.get_sim_blm(idx), self.cl['bb'])

class library_ftl:
    def __init__(self, ivfs, lmax, lfilt_t, lfilt_e, lfilt_b):
        """ Library of re-weighted filtered maps.  lmax defines the new healpy alm array shape (identical for T E B) """
        assert len(lfilt_t) > lmax and len(lfilt_e) > lmax and len(lfilt_b) > lmax
        self.ivfs = ivfs
        self.lmax = lmax
        self.lfilt_t = lfilt_t
        self.lfilt_e = lfilt_e
        self.lfilt_b = lfilt_b
        self.lib_dir = ivfs.lib_dir

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(),
                'filt_t': utils.clhash(self.lfilt_t[:self.lmax + 1]),
                'filt_e': utils.clhash(self.lfilt_e[:self.lmax + 1]),
                'filt_b': utils.clhash(self.lfilt_b[:self.lmax + 1])}

    def get_fmask(self):
        return self.ivfs.get_fmask()

    def get_tal(self, a):
        return self.ivfs.get_tal(a)

    def get_ftl(self):
        return self.ivfs.get_ftl()[:self.lmax + 1] * self.lfilt_t[:self.lmax + 1]

    def get_fel(self):
        return self.ivfs.get_fel()[:self.lmax + 1] * self.lfilt_e[:self.lmax + 1]

    def get_fbl(self):
        return self.ivfs.get_fbl()[:self.lmax + 1] * self.lfilt_b[:self.lmax + 1]

    def get_sim_tlm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_tlm(idx), lmax=self.lmax), self.lfilt_t)

    def get_sim_elm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_elm(idx), lmax=self.lmax), self.lfilt_e)

    def get_sim_blm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_blm(idx), lmax=self.lmax), self.lfilt_b)

    def get_sim_tmliklm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_tmliklm(idx), lmax=self.lmax),self.lfilt_t)

    def get_sim_emliklm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_emliklm(idx), lmax=self.lmax), self.lfilt_e)

    def get_sim_bmliklm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_bmliklm(idx), lmax=self.lmax), self.lfilt_b)


class library_shuffle:
    def __init__(self, ivfs, idxs):
        """ A library of filtered sims with remapped indices according to idxs[idx] in place of id """
        self.ivfs = ivfs
        self.idxs = idxs

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(), 'idxs': self.idxs}

    def get_fmask(self):
        return self.ivfs.get_fmask()

    def get_tal(self, a):
        return self.ivfs.get_tal(a)

    def get_ftl(self):
        return self.ivfs.get_ftl()

    def get_fel(self):
        return self.ivfs.get_fel()

    def get_fbl(self):
        return self.ivfs.get_fbl()

    def get_sim_tlm(self, idx):
        return self.ivfs.get_sim_tlm(self.idxs[idx])

    def get_sim_elm(self, idx):
        return self.ivfs.get_sim_elm(self.idxs[idx])

    def get_sim_blm(self, idx):
        return self.ivfs.get_sim_blm(self.idxs[idx])

    def get_sim_tmliklm(self, idx):
        return self.ivfs.get_sim_tmliklm(self.idxs[idx])

    def get_sim_emliklm(self, idx):
        return self.ivfs.get_sim_emliklm(self.idxs[idx])

    def get_sim_bmliklm(self, idx):
        return self.ivfs.get_sim_bmliklm(self.idxs[idx])
