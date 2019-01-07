"""Planck 2018 release simulation libraries.

"""
import healpy as hp

class smica_dx12:
    """ SMICA 2018 release simulation and data library at NERSC.

    """
    def __init__(self):
        self.cmbs = '/project/projectdirs/cmb/data/planck2018/ffp10/compsep/mc_cmb/dx12_v3_smica_cmb_mc_%05d_005a_2048.fits'
        self.noise = '/project/projectdirs/cmb/data/planck2018/ffp10/compsep/mc_noise/dx12_v3_smica_noise_mc_%05d_005a_2048.fits'
        self.data = '/project/projectdirs/cmb/data/planck2018/pr3/cmbmaps/dx12_v3_smica_cmb_005a_2048.fits'

    def hashdict(self):
        return {'cmbs':self.cmbs, 'noise':self.noise, 'data':self.data}

    def get_sim_tmap(self, idx):
        if idx == -1:
            return self.get_dat_tmap()
        return 1e6 * (hp.read_map(self.cmbs % idx, field=0) + hp.read_map(self.noise % idx, field=0))

    def get_dat_tmap(self):
        return 1e6 * hp.read_map(self.data, field=0)

    def get_sim_pmap(self, idx):
        if idx == -1:
            return self.get_dat_pmap()
        Q = 1e6 * (hp.read_map(self.cmbs % idx, field=1) + hp.read_map(self.noise % idx, field=1))
        U = 1e6 * (hp.read_map(self.cmbs % idx, field=2) + hp.read_map(self.noise % idx, field=2))
        return Q, U

    def get_dat_pmap(self):
        return 1e6 * hp.read_map(self.data, field=1), 1e6 * hp.read_map(self.data, field=2)


class cmb_len_ffp10:
    """ FFP10 input sim libraries, lensed alms.

    """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs, freq 0'}

    @staticmethod
    def get_sim_tlm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_lensed_scl_cmb_000_alm_mc_%04d.fits'%idx, hdu=3)


class cmb_unl_ffp10:
    """ FFP10 input sim libraries, unlensed alms.

    """
    def __init__(self):
        pass

    def hashdict(self):
        return {'sim_lib': 'ffp10 unlensed scalar cmb inputs'}

    @staticmethod
    def get_sim_tlm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx, hdu=1)

    @staticmethod
    def get_sim_elm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx, hdu=2)

    @staticmethod
    def get_sim_blm(idx):
        return 1e6 * hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx, hdu=3)

    @staticmethod
    def get_sim_plm(idx):
        return hp.read_alm('/project/projectdirs/cmb/data/generic/cmb/ffp10/mc/scalar/ffp10_unlensed_scl_cmb_000_tebplm_mc_%04d.fits'% idx, hdu=4)


