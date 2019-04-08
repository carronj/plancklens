import os

import numpy as np

from plancklens2018 import utils

PL2018 = os.environ['PL2018']

def get_blbubc(bin_type):
    if bin_type == 'consext8':
        bins_l = np.array([8, 41, 85, 130, 175, 220, 265, 310, 355])
        bins_u = np.array([40, 84, 129, 174, 219, 264, 309, 354, 400])
    elif bin_type == 'agr2':
        bins_l = np.array([8, 21, 40, 66, 101, 145, 199, 264, 339, 426, 526, 638, 763, 902])
        bins_u = np.array([20, 39, 65, 100, 144, 198, 263, 338, 425, 525, 637, 762, 901, 2048])
    elif '_' in bin_type:
        edges = np.int_(bin_type.split('_'))
        bins_l = edges[:-1]
        bins_u = edges[1:] - 1
        bins_u[-1] += 1
    else:
        assert 0, bin_type + ' not implemented'
    return bins_l, bins_u, 0.5 * (bins_l + bins_u)

class ffp10_binner:
    def __init__(self, k1, k2, parfile, btype, ksource='p'):
        assert ksource == 'p', ksource +  ' source not implemented'
        lmaxphi = 2048
        clpp_fid =  utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lenspotentialCls.dat'))['pp'][:lmaxphi+1]
        kappaswitch = (np.arange(0, lmaxphi + 1, dtype=float) * (np.arange(1, lmaxphi + 2))) ** 2 / (2. * np.pi) * 1e7
        clkk_fid = clpp_fid * kappaswitch

        qc_resp = parfile.qresp_dd.get_response(k1, ksource) * parfile.qresp_dd.get_response(k2, ksource)
        bin_lmins, bin_lmaxs, bin_centers = get_blbubc(btype)
        vlpp_inv = qc_resp * (2 * np.arange(lmaxphi + 1) + 1) * (0.5 * parfile.qcls_dd.fsky1234)
        vlpp_inv[np.where(kappaswitch != 0)] /= kappaswitch[np.where(kappaswitch != 0)] ** 2
        vlpp_den = [np.sum([clkk_fid[l] ** 2 * vlpp_inv[l] for l in range(lmin, lmax + 1)]) for lmin, lmax in zip(bin_lmins, bin_lmaxs)]

        fid_bandpowers = np.ones(len(bin_centers))  # We will renormalize that as soon as l_av is calculated.

        def _get_BiL(i, L):  # Bin i window function to be applied to cLpp-like arrays as just described
            ret = (fid_bandpowers[i] / vlpp_den[i]) * vlpp_inv[L] * clkk_fid[L] * kappaswitch[L]
            ret *= (L >= bin_lmins[i]) * (L <= bin_lmaxs[i])
            return ret

        lav = np.zeros(len(bin_centers))
        for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
            w_lav = 1. / np.arange(lmin, lmax + 1) ** 2 / np.arange(lmin + 1, lmax + 2) ** 2
            lav[i] = np.sum(np.arange(lmin, lmax + 1) * w_lav * _get_BiL(i, np.arange(lmin, lmax + 1))) / np.sum(
                w_lav * _get_BiL(i, np.arange(lmin, lmax + 1)))

        self.k1 = k1
        self.k2 = k2
        self.ksource = ksource
        self.parfile = parfile

        self.fid_bandpowers =  np.interp(lav, np.arange(lmaxphi + 1, dtype=float), clkk_fid)
        self.bin_lmins = bin_lmins
        self.bin_lmaxs = bin_lmaxs
        self.bin_lavs = lav
        self.nbins = len(bin_centers)

        self.vlpp_den = vlpp_den
        self.vlpp_inv = vlpp_inv
        self.clkk_fid = clkk_fid
        self.kappaswitch = kappaswitch

    def _get_BiL(self, i, L):
        ret = (self.fid_bandpowers[i] / self.vlpp_den[i]) * self.vlpp_inv[L] * self.clkk_fid[L] * self.kappaswitch[L]
        ret *= (L >= self.bin_lmins[i]) * (L <= self.bin_lmaxs[i])
        return ret

    def _get_binnedcl(self, cl):
        ret = np.zeros(self.nbins)
        for i, (lmin, lmax) in enumerate(zip(self.bin_lmins, self.bin_lmaxs)):
            ret[i] = np.sum(self._get_BiL(i, np.arange(lmin, lmax + 1)) * cl[lmin:lmax + 1])
        return ret

    def get_mcn0(self):
        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * (2. * ss))

    def get_rdn0(self):
        ds = self.parfile.qcls_ds.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * (4 * ds - 2. * ss))

    def get_n1(self):
        """Analytical N1 caculation.

            This takes the analyical approximation to the QE pair filtering as input.

        """
        assert self.k1 == self.k2, 'check signs for qe''s of different spins'
        assert self.ksource[0] == 'p', 'check aniso source spectrum'
        # This implementation accepts 2 different qes but pairwise identical filtering on each qe leg.
        assert np.all(self.parfile.qcls_dd.qeA.f2map1.ivfs.get_ftl() == self.parfile.qcls_dd.qeA.f2map2.ivfs.get_ftl())
        assert np.all(self.parfile.qcls_dd.qeA.f2map1.ivfs.get_fel() == self.parfile.qcls_dd.qeA.f2map2.ivfs.get_fel())
        assert np.all(self.parfile.qcls_dd.qeA.f2map1.ivfs.get_fbl() == self.parfile.qcls_dd.qeA.f2map2.ivfs.get_fbl())
        assert np.all(self.parfile.qcls_dd.qeB.f2map1.ivfs.get_ftl() == self.parfile.qcls_dd.qeB.f2map2.ivfs.get_ftl())
        assert np.all(self.parfile.qcls_dd.qeB.f2map1.ivfs.get_fel() == self.parfile.qcls_dd.qeB.f2map2.ivfs.get_fel())
        assert np.all(self.parfile.qcls_dd.qeB.f2map1.ivfs.get_fbl() == self.parfile.qcls_dd.qeB.f2map2.ivfs.get_fbl())

        ivfsA = self.parfile.qcls_dd.qeA.f2map1.ivfs
        ivfsB = self.parfile.qcls_dd.qeB.f2map1.ivfs
        ftlA = ivfsA.get_ftl()
        felA = ivfsA.get_fel()
        fblA = ivfsA.get_fbl()
        ftlB = ivfsB.get_ftl()
        felB = ivfsB.get_fel()
        fblB = ivfsB.get_fbl()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        n1pp = self.parfile.n1_dd.get_n1(self.k1, self.ksource, self.clkk_fid *utils.cli(self.kappaswitch), ftlA, felA, fblA, len(qc_resp) - 1
                                    , kB=self.k2, ftlB=ftlB, felB=felB, fblB=fblB)
        return self._get_binnedcl(utils.cli(qc_resp) * n1pp)

    def get_bmmc(self):
        assert 0, 'FIXME'
        MC = get_binnedcl(getme.get_MC_pp_qcl(useN1=flags['useN1']))
        bmMC = 1. / (1 + MC / self.fid_bandpowers)  # Binned multiplicative MC correction.
        return binned(cl_ksource, btype) / cl

