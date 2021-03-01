r"""CMB filtering utilities module.

This module collects some convenience wrapper libraries.

"""
import healpy as hp
from plancklens import utils
import numpy as np

class library_ftl:
    """ Library of a-posteriori re-scaled filtered CMB maps, for separate temperature and polarization filtering

    Args:
         ivfs : inverse filtering library instance.
         lmax (int) : defines the new healpy alm array shape (identical for temperature and polarization)
         lfilt_t (1d array): filtered temperature alms are rescaled by lfilt_t
         lfilt_e (1d array): filtered E-polarization alms are rescaled by lfilt_e
         lfilt_b (1d array): filtered B-polarization alms are rescaled by lfilt_b

    Wraps the input filtering instance *(ivfs)* methods to keep the same interface.

    Note:

        ftl fel fbl should eventually be taken off to be replaced by fal in all cases

    """
    def __init__(self, ivfs, lmax, lfilt_t, lfilt_e, lfilt_b):
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
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_tmliklm(idx), lmax=self.lmax), self.lfilt_t)

    def get_sim_emliklm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_emliklm(idx), lmax=self.lmax), self.lfilt_e)

    def get_sim_bmliklm(self, idx):
        return hp.almxfl(utils.alm_copy(self.ivfs.get_sim_bmliklm(idx), lmax=self.lmax), self.lfilt_b)


class library_shuffle:
    r"""A library of filtered sims with remapped indices.

        This is useful for lensing biases calculations, such as :math:`\hat N^{(0)}_L.`

        Args:
            ivfs : inverse-variance filtering library instance.
            idxs : index idx of this new instance points to idxs[idx] of the input *ivfs* instance.

        Wraps the input filtering instance *(ivfs)* methods to keep the same interface.

    """
    def __init__(self, ivfs, idxs):
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

