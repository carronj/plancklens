"""Planck 2018 release simulation libraries.

"""
import healpy as hp

class smica_dx12:
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

