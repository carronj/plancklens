r"""CMB filtering utilities module.

This module collects some convenience wrapper libraries.

"""
import healpy as hp
from plancklens import utils
import numpy as np

def _alm_copy(alm, mmaxin:int or None, lmaxout:int, mmaxout:int):
    """Copies the healpy alm array, with the option to change its lmax

        Parameters
        ----------
        alm :ndarray
            healpy alm array to copy.
        mmaxin: int or None
            mmax parameter of input array (can be set to None or negative for default)
        lmaxout : int
            new alm lmax
        mmaxout: int
            new alm mmax


    """
    lmaxin = hp.Alm.getlmax(alm.size, mmaxin)
    if mmaxin is None or mmaxin < 0: mmaxin = lmaxin
    if (lmaxin == lmaxout) and (mmaxin == mmaxout):
        ret = np.copy(alm)
    else:
        ret = np.zeros(hp.Alm.getsize(lmaxout, mmaxout), dtype=complex)
        lmax_min = min(lmaxout, lmaxin)
        for m in range(0, min(mmaxout, mmaxin) + 1):
            idx_in =  m * (2 * lmaxin + 1 - m) // 2 + m
            idx_out = m * (2 * lmaxout+ 1 - m) // 2 + m
            ret[idx_out: idx_out + lmax_min + 1 - m] = alm[idx_in: idx_in + lmax_min + 1 - m]
    return ret

class library_ftl:
    """ Library of a-posteriori re-scaled filtered CMB maps, for separate temperature and polarization filtering

    Args:
         ivfs : inverse filtering library instance (e.g. one of those in plancklens.filt.filt_simple).
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
        self.mmax = lmax
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
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_tlm(idx), None, self.lmax, self.mmax), self.lfilt_t)

    def get_sim_elm(self, idx):
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_elm(idx), None, self.lmax, self.mmax), self.lfilt_e)

    def get_sim_blm(self, idx):
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_blm(idx), None, self.lmax, self.mmax), self.lfilt_b)

    def get_sim_tmliklm(self, idx):
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_tmliklm(idx), None, self.lmax, self.mmax), self.lfilt_t)

    def get_sim_emliklm(self, idx):
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_emliklm(idx), None, self.lmax, self.mmax), self.lfilt_e)

    def get_sim_bmliklm(self, idx):
        return hp.almxfl(_alm_copy(self.ivfs.get_sim_bmliklm(idx), None, self.lmax, self.mmax), self.lfilt_b)


class library_fml:
    def __init__(self, ivfs, lmax, mfilt_t, mfilt_e, mfilt_b):
        """Library of a-posteriori re-scaled filtered CMB maps

            This rescales the maps according to 'm' values, alm -> fm alm

        Args:
             ivfs : inverse filtering library instance (e.g. one of those in plancklens.filt.filt_simple).
             lmax (int) : defines the new healpy alm array shape (identical for temperature and polarization)
             mfilt_t (1d array): filtered temperature alms are rescaled by mfilt_t
             mfilt_e (1d array): filtered E-polarization alms are rescaled by mfilt_e
             mfilt_b (1d array): filtered B-polarization alms are rescaled by mfilt_b

        Wraps the input filtering instance *(ivfs)* methods to keep the same interface.


        """
        assert len(mfilt_t) > lmax and len(mfilt_e) > lmax and len(mfilt_b) > lmax
        self.ivfs = ivfs
        self.lmax = lmax
        self.mfilt_t = mfilt_t
        self.mfilt_e = mfilt_e
        self.mfilt_b = mfilt_b
        self.lib_dir = ivfs.lib_dir

    def hashdict(self):
        return {'ivfs': self.ivfs.hashdict(),
                'filt_t': utils.clhash(self.mfilt_t[:self.lmax + 1]),
                'filt_e': utils.clhash(self.mfilt_e[:self.lmax + 1]),
                'filt_b': utils.clhash(self.mfilt_b[:self.lmax + 1])}


    def get_fmask(self):
        return self.ivfs.get_fmask()

    @staticmethod
    def almxfm(alm, fm, lmax):
        ret = utils.alm_copy(alm, lmax=lmax)
        for m in range(lmax + 1):
            ret[hp.Alm.getidx(lmax, np.arange(m, lmax + 1, dtype=int), m)] *= fm[m]
        return ret

    def get_tal(self, a):
        return self.ivfs.get_tal(a)

    def get_ftl(self):
        m_rescal = 2 * np.cumsum(self.mfilt_t[:self.lmax + 1]) - self.mfilt_t[0]
        m_rescal /= (2 * np.arange(self.lmax + 1) + 1)
        return self.ivfs.get_ftl()[:self.lmax + 1] * np.sqrt(m_rescal) # root has better chance to work at the spectrum level

    def get_fel(self):
        m_rescal = 2 * np.cumsum(self.mfilt_e[:self.lmax + 1]) - self.mfilt_e[0]
        m_rescal /= (2 * np.arange(self.lmax + 1) + 1)
        return self.ivfs.get_fel()[:self.lmax + 1] * np.sqrt(m_rescal)

    def get_fbl(self):
        m_rescal = 2 * np.cumsum(self.mfilt_b[:self.lmax + 1]) - self.mfilt_b[0]
        m_rescal /= (2 * np.arange(self.lmax + 1) + 1)
        return self.ivfs.get_fbl()[:self.lmax + 1] * np.sqrt(m_rescal)

    def get_sim_tlm(self, idx):
        return self.almxfm(self.ivfs.get_sim_tlm(idx), self.mfilt_t, self.lmax)

    def get_sim_elm(self, idx):
        return self.almxfm(self.ivfs.get_sim_elm(idx), self.mfilt_t, self.lmax)

    def get_sim_blm(self, idx):
        return self.almxfm(self.ivfs.get_sim_blm(idx), self.mfilt_t, self.lmax)

    def get_sim_tmliklm(self, idx):
        return self.almxfm(self.ivfs.get_sim_tmliklm(idx), self.mfilt_t, self.lmax)

    def get_sim_emliklm(self, idx):
        return self.almxfm(self.ivfs.get_sim_emliklm(idx), self.mfilt_e, self.lmax)

    def get_sim_bmliklm(self, idx):
        return self.almxfm(self.ivfs.get_sim_bmliklm(idx), self.mfilt_b, self.lmax)



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
