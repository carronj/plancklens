"""Band-powers construction module.

    This is used to construct anistropy band-powers from a parameter file.

"""
import os
import numpy as np

import plancklens
from plancklens import utils
from plancklens import nhl


def get_blbubc(bin_type):
    if bin_type == 'consext8':
        bins_l = np.array([8, 41, 85, 130, 175, 220, 265, 310, 355])
        bins_u = np.array([40, 84, 129, 174, 219, 264, 309, 354, 400])
    elif bin_type == 'agr2':
        bins_l = np.array([8, 21, 40, 66, 101, 145, 199, 264, 339, 426, 526, 638, 763, 902])
        bins_u = np.array([20, 39, 65, 100, 144, 198, 263, 338, 425, 525, 637, 762, 901, 2048])
    elif bin_type == 'xdip':
        bins_l = np.array([8, 264, 902])
        bins_u = np.array([263, 901, 2048])
    elif bin_type == 'pdip':
        bins_l = np.array([8, 101, 426])
        bins_u = np.array([100, 425,  2048])
    elif bin_type == 'lowl':
        bins_l = np.array([2,7])
        bins_u = np.array([8,40])
    elif bin_type == '1_10_unb':
        bins_l = np.arange(1, 11)
        bins_u = bins_l
    elif '_' in bin_type:
        edges = np.int_(bin_type.split('_'))
        bins_l = edges[:-1]
        bins_u = edges[1:] - 1
        bins_u[-1] += 1
    else:
        assert 0, bin_type + ' not implemented'
    return bins_l, bins_u, 0.5 * (bins_l + bins_u)


class ffp10_binner:
    """Band-power construction library using the FFP10 fiducial cosmology.

        This combines lensing (or anisotropy) estimates spectra to build band-powers in the exact same way than
        the Planck 2018 lensing analysis.

        This uses the various QE and QE spectra libraries defined in a parameter file, in particular:
         - *qcls_dd*  (for data band-powers, to build covariance matrix, Monte-Carlo and point-source correction)
         - *qcls_ds*  (for RDN0 and point-source correction)
         - *qcls_ss*  (for MCN0, RDN0, Monte-Carlo and point-source correction)
         - *qresp_dd* (for the estimator normalization)
         - *n1_dd* (for the N1 bias subtraction)
         - *nhl_dd* (to build the semi-analytical covariance matrix)
         - *ivfs* (for the N1 bias and point-source correction)

         In each of the methods defined here (e.g. MCN0, RDN0...),  if the relevant QE, QE spectra, etc cannot be found
         precomputed, this will be performed on the fly. Hence in a realistic configuration it is always advisable
         to build them all previously (for example with *examples/run_qlms.py*)


        This library can be used to build the cross power spectra of two anisotropy estimators, calculates biases,
        obtain MC and point-source corrections and the covariance matrix.

        Args:
            k1: First quadratic estimator key. See the qest.py module for the key definitions.
            k2: Second quadratic estimator key. See the qest.py module for the key definitions.
            parfile: parameter file where the relevant QE libraries are defined
            btype: bin type descriptor ('consext8' or 'arg2' were the Planck 2018 lensing analysis defaults)
            ksource: anisotropy source (defaults to 'p', lensing)

    """
    def __init__(self, k1, k2, parfile, btype, ksource='p'):

        lmaxphi = 2048
        cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')

        if ksource == 'p':
            kswitch = (np.arange(0, lmaxphi + 1, dtype=float) * (np.arange(1, lmaxphi + 2))) ** 2 / (2. * np.pi) * 1e7
            if k1[0] == 'p' and k2[0] == 'p':
                clpp_fid = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))['pp'][:lmaxphi + 1]
            elif k1[0] == 'x' and k2[0] == 'x':
                clpp_fid = np.ones(lmaxphi + 1, dtype=float)
            else:
                assert 0, 'not implemented'
        else:
            kswitch  = np.ones(lmaxphi + 1, dtype=float)
            clpp_fid = np.ones(lmaxphi + 1, dtype=float)

        clkk_fid = clpp_fid * kswitch
        qc_resp = parfile.qresp_dd.get_response(k1, ksource)[:lmaxphi+1] * parfile.qresp_dd.get_response(k2, ksource)[:lmaxphi+1]
        bin_lmins, bin_lmaxs, bin_centers = get_blbubc(btype)
        vlpp_inv = qc_resp * (2 * np.arange(lmaxphi + 1) + 1) * (0.5 * getattr(parfile.qcls_dd, 'fsky1234', 1.)) # value irrelevant here
        vlpp_inv *= utils.cli(kswitch) ** 2
        vlpp_den = [np.sum(clkk_fid[slice(lmin, lmax + 1)] ** 2 * vlpp_inv[slice(lmin, lmax + 1)]) for lmin, lmax in zip(bin_lmins, bin_lmaxs)]

        fid_bandpowers = np.ones(len(bin_centers))  # We will renormalize that as soon as l_av is calculated.

        def _get_bil(i, L):  # Bin i window function to be applied to cLpp-like arrays as just described
            ret = (fid_bandpowers[i] / vlpp_den[i]) * vlpp_inv[L] * clkk_fid[L] * kswitch[L]
            ret *= (L >= bin_lmins[i]) * (L <= bin_lmaxs[i])
            return ret

        lav = np.zeros(len(bin_centers))
        for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
            w_lav = 1. / np.arange(lmin, lmax + 1) ** 2 / np.arange(lmin + 1, lmax + 2) ** 2
            lav[i] = np.sum(np.arange(lmin, lmax + 1) * w_lav * _get_bil(i, np.arange(lmin, lmax + 1))) / np.sum(
                w_lav * _get_bil(i, np.arange(lmin, lmax + 1)))

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
        self.kswitch = kswitch

        self.cls_path = cls_path

    def _get_bil(self, i, L):
        ret = (self.fid_bandpowers[i] / self.vlpp_den[i]) * self.vlpp_inv[L] * self.clkk_fid[L] * self.kswitch[L]
        ret *= (L >= self.bin_lmins[i]) * (L <= self.bin_lmaxs[i])
        return ret

    def _get_binnedcl(self, cl):
        assert len(cl) > self.bin_lmaxs[-1], (len(cl), self.bin_lmaxs[-1])
        ret = np.zeros(self.nbins)
        for i, (lmin, lmax) in enumerate(zip(self.bin_lmins, self.bin_lmaxs)):
            ret[i] = np.sum(self._get_bil(i, np.arange(lmin, lmax + 1)) * cl[lmin:lmax + 1])
        return ret

    def get_fid_bandpowers(self):
        """Returns Expected band-powers in the FFP10 fiducial cosmology.

        """
        return np.copy(self.fid_bandpowers)

    def get_dat_bandpowers(self):
        """Returns data raw band-powers, prior to any biases subtraction or correction.

        """
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * self.parfile.qcls_dd.get_sim_qcl(self.k1, -1, k2=self.k2))

    def get_mcn0(self):
        """Returns Monte-Carlo N0 lensing bias.

        """
        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * (2. * ss))

    def get_rdn0(self):
        """Returns realization-dependent N0 lensing bias RDN0.

        """
        ds = self.parfile.qcls_ds.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * (4 * ds - 2. * ss))

    def get_dat_nhl(self):
        """Returns N0 lensing bias, semi-analytical version.

            This is not highly accurate on the cut-sky

        """
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        return self._get_binnedcl(utils.cli(qc_resp) * self.parfile.nhl_dd.get_sim_nhl(-1, self.k1, self.k2))

    def get_n1(self, k1=None, k2=None, unnormed=False):
        """Returns analytical N1 lensing bias.

            This uses the analyical approximation to the QE pair filtering as input.

        """
        k1 = self.k1 if k1 is None else k1
        k2 = self.k2 if k2 is None else k2
        assert k1 == k2, 'check signs for qe''s of different spins'
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
        clpp_fid =  utils.camb_clfile(os.path.join(self.cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
        qc_resp = self.parfile.qresp_dd.get_response(k1, self.ksource) * self.parfile.qresp_dd.get_response(k2, self.ksource)
        n1pp = self.parfile.n1_dd.get_n1(k1, self.ksource, clpp_fid, ftlA, felA, fblA, len(qc_resp) - 1,
                                         kB=k2, ftlB=ftlB, felB=felB, fblB=fblB)
        return self._get_binnedcl(utils.cli(qc_resp) * n1pp) if not unnormed else n1pp

    def get_ps_data(self,lmin_ss_s4=100,lmax_ss_s4=2048,mc_sims_ss=None,mc_sims_ds=None):
        ks4 = 'stt'
        twolpo = 2 * np.arange(lmax_ss_s4 + 1) + 1.
        filt = np.ones(lmax_ss_s4 + 1, dtype=float)
        filt[:lmax_ss_s4] *= 0.

        dd_ptsrc = self.parfile.qcls_dd.get_sim_stats_qcl(ks4, self.parfile.mc_sims_var).mean()[:lmax_ss_s4 + 1]
        ds_ptsrc = self.parfile.qcls_ds.get_sim_stats_qcl(ks4, self.parfile.mc_sims_bias if mc_sims_ds is None else mc_sims_ds).mean()[:lmax_ss_s4 + 1]
        ss_ptsrc = self.parfile.qcls_ss.get_sim_stats_qcl(ks4, self.parfile.mc_sims_bias if mc_sims_ss is None else mc_sims_ss).mean()[:lmax_ss_s4 + 1]
        dat_ptsrc = self.parfile.qcls_dd.get_sim_qcl(ks4, -1)[:lmax_ss_s4 + 1]

        # This simple PS implementation accepts only identical filtering on each four legs.
        assert np.all(self.parfile.qcls_dd.qeA.f2map1.ivfs.get_ftl() == self.parfile.qcls_dd.qeA.f2map2.ivfs.get_ftl())
        assert np.all(self.parfile.qcls_dd.qeB.f2map1.ivfs.get_ftl() == self.parfile.qcls_dd.qeB.f2map2.ivfs.get_ftl())
        assert np.all(self.parfile.qcls_dd.qeA.f2map1.ivfs.get_ftl() == self.parfile.qcls_dd.qeB.f2map1.ivfs.get_ftl())
        ftl = self.parfile.qcls_dd.qeA.f2map1.ivfs.get_ftl()
        qc_resp_ptsrc = nhl.get_nhl(ks4, ks4, {}, {'tt':ftl}, len(ftl) - 1, len(ftl) - 1, lmax_out=lmax_ss_s4)[0] ** 2

        s4_band_norm = 4.0 / np.sum(4.0 * (twolpo[lmin_ss_s4:lmax_ss_s4 + 1] * qc_resp_ptsrc[lmin_ss_s4:lmax_ss_s4 + 1]))
        s4_cl_dat = s4_band_norm * twolpo * (dat_ptsrc - 4. * ds_ptsrc + 2. * ss_ptsrc)
        s4_cl_check = s4_band_norm * twolpo * (dd_ptsrc - 2. * ss_ptsrc)
        s4_cl_systs = s4_band_norm * twolpo * (4. * ds_ptsrc - 4. * ss_ptsrc)
        # phi-induced PS estimator N1
        clpp_fid =  utils.camb_clfile(os.path.join(self.cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
        s4_cl_clpp_n1 = s4_band_norm * twolpo * self.get_n1(k1=ks4, k2=ks4, unnormed=True)[:lmax_ss_s4+1]

        s4_cl_clpp_prim = s4_band_norm * twolpo * self.parfile.qresp_dd.get_response(ks4, self.ksource) [ :lmax_ss_s4 + 1] ** 2 * clpp_fid[:lmax_ss_s4 + 1]

        s4_band_dat = np.sum((s4_cl_dat - s4_cl_clpp_prim - s4_cl_clpp_n1)[lmin_ss_s4: lmax_ss_s4 + 1])
        s4_band_check = np.sum((s4_cl_check - s4_cl_clpp_prim - s4_cl_clpp_n1)[lmin_ss_s4: lmax_ss_s4 + 1])
        s4_band_syst = np.abs(np.sum(s4_cl_systs[lmin_ss_s4: lmax_ss_s4 + 1]))

        Cs2s2 = (s4_cl_dat - s4_cl_clpp_prim - s4_cl_clpp_n1) * utils.cli(twolpo) / s4_band_norm
        Cs2s2 *= utils.cli(qc_resp_ptsrc[:lmax_ss_s4 + 1])
        # reconstucted PS power (with correct normalization)
        s4_band_sim_stats = []

        for i, idx in utils.enumerate_progress(self.parfile.mc_sims_var):
            ts4_cl = s4_band_norm * twolpo[: lmax_ss_s4 + 1] * \
                     (self.parfile.qcls_dd.get_sim_qcl(ks4, idx)[:lmax_ss_s4 + 1] - 2. * ss_ptsrc)
            s4_band_sim_stats.append(np.sum((ts4_cl - s4_cl_clpp_prim - s4_cl_clpp_n1)[lmin_ss_s4: lmax_ss_s4 + 1]))

        print("ptsrc stats:")
        print('   fit range = [' + str(lmin_ss_s4) + ', ' + str(lmax_ss_s4) + ']')
        print('   sim avg has amplitude of ' + ('%.3g +- %0.3g (stat), discrepant from zero at %.3f sigma.' %
                                                (s4_band_check,
                                                 np.std(s4_band_sim_stats) / np.sqrt(len(self.parfile.mc_sims_var)),
                s4_band_check / np.std(s4_band_sim_stats) * np.sqrt(len(self.parfile.mc_sims_var)))))
        print('   dat has amplitude of ' + ('%.3g +- %0.3g (stat), signif of %.3f sigma.' %
                                            (s4_band_dat, np.std(s4_band_sim_stats),
                                             s4_band_dat / np.sqrt(np.var(s4_band_sim_stats)))))
        qc_resp =   self.parfile.qresp_dd.get_response(self.k1, self.ksource) \
                  * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        # PS spectrum response to ks4, using qe.key- source key symmetry of response functions.
        qlss = self.parfile.qresp_dd.get_response(ks4, self.k1[0]) * self.parfile.qresp_dd.get_response(ks4, self.k2[0])
        # Correction to apply to estimated spectrum :
        pp_cl_ps = s4_band_dat * utils.cli(qc_resp) * qlss
        return s4_band_dat, s4_band_check, s4_band_syst, s4_band_sim_stats, Cs2s2, pp_cl_ps

    def get_ps_corr(self, lmin_ss_s4=100, lmax_ss_s4=2048):
        """Returns point-source correction

        """
        return self._get_binnedcl(self.get_ps_data(lmin_ss_s4=lmin_ss_s4, lmax_ss_s4=lmax_ss_s4)[-1])

    def get_bamc(self, wn1=True):
        """Binned additive MC correction, with crude error bars.

            This compares the reconstruction on the simulations to the FFP10 input lensing spectrum.

            Note:
                the approximate error corrections to the additive MC correction variance follows Appendix C of
                https://arxiv.org/abs/1807.06210, check this for more details on its validity.

        """
        assert self.k1[0] == 'p' and self.k2[0] == 'p' and self.ksource == 'p', (self.k1, self.k2, self.ksource)
        ss2 = 2 * self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, self.parfile.mc_sims_var, k2=self.k2).mean()
        cl_pred = utils.camb_clfile(os.path.join(self.cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))['pp'][:len(ss2)]
        qc_norm = utils.cli(self.parfile.qresp_dd.get_response(self.k1, self.ksource)
                              * self.parfile.qresp_dd.get_response(self.k2, self.ksource))
        bp_stats = utils.stats(self.nbins)
        bp_n1 = self.get_n1() if wn1 else np.zeros(self.nbins, dtype=float)
        for i, idx in utils.enumerate_progress(self.parfile.mc_sims_var, label='collecting BP stats'):
            dd = self.parfile.qcls_dd.get_sim_qcl(self.k1, idx, k2=self.k2)
            bp_stats.add(self._get_binnedcl(qc_norm *(dd - ss2) - cl_pred) - bp_n1)
        NMF = len(self.parfile.qcls_dd.mc_sims_mf)
        if NMF == 0: NMF = np.inf
        NB = len(self.parfile.mc_sims_var)
        return bp_stats.mean(), bp_stats.sigmas_on_mean() * np.sqrt((1. + 1. + 2. / NMF + 2 * NB / (float(NMF * NMF))))
        # + 1 from MCN0 error, 2nd term MF linear error term, 3rd term from MF quadratic term (cancelled in data rec.)

    def get_bmmc(self, mc_sims_dd=None, mc_sims_ss=None, wN1=True):
        """Binned multiplicative MC correction.

            This compares the reconstruction on the simulations to the FFP10 input lensing spectrum.

        """
        assert self.k1[0] == 'p' and self.k2[0] == 'p' and self.ksource == 'p', (self.k1, self.k2, self.ksource)
        if mc_sims_dd is None: mc_sims_dd = self.parfile.mc_sims_var
        if mc_sims_ss is None: mc_sims_ss = self.parfile.mc_sims_var
        dd = self.parfile.qcls_dd.get_sim_stats_qcl(self.k1, mc_sims_dd, k2=self.k2).mean()
        ss = self.parfile.qcls_ss.get_sim_stats_qcl(self.k1, mc_sims_ss, k2=self.k2).mean()
        cl_pred =  utils.camb_clfile(os.path.join(self.cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))['pp']
        qc_resp = self.parfile.qresp_dd.get_response(self.k1, self.ksource) * self.parfile.qresp_dd.get_response(self.k2, self.ksource)
        bps = self._get_binnedcl(utils.cli(qc_resp) * (dd - 2 * ss) - cl_pred[:len(dd)])
        if wN1: bps -= self.get_n1()
        return 1. / (1 + bps / self.fid_bandpowers)

    def get_nhl_cov(self, mc_sims_dd=None):
        """Covariance matrix obtained from the semi-analytical N0 debiaser.

        """
        if mc_sims_dd is None: mc_sims_dd = self.parfile.mc_sims_var
        nhl_cov = utils.stats(self.nbins)
        qc_norm = utils.cli(self.parfile.qresp_dd.get_response(self.k1, self.ksource)
                              * self.parfile.qresp_dd.get_response(self.k2, self.ksource))
        for i, idx in utils.enumerate_progress(mc_sims_dd):
            dd = self.parfile.qcls_dd.get_sim_qcl(self.k1, idx, k2=self.k2)
            nhl_cov.add(self._get_binnedcl(qc_norm * (dd- self.parfile.nhl_dd.get_sim_nhl(int(idx), self.k1, self.k2))))
        return nhl_cov.cov()

    def get_mcn0_cov(self, mc_sims_dd=None):
        """Covariance matrix obtained from the realization-independent debiaser.

        """
        if mc_sims_dd is None: mc_sims_dd = self.parfile.mc_sims_var
        mcn0_cov = utils.stats(self.nbins)
        qc_norm = utils.cli(self.parfile.qresp_dd.get_response(self.k1, self.ksource)
                              * self.parfile.qresp_dd.get_response(self.k2, self.ksource))
        for i, idx in utils.enumerate_progress(mc_sims_dd):
            dd = self.parfile.qcls_dd.get_sim_qcl(self.k1, idx, k2=self.k2)
            mcn0_cov.add(self._get_binnedcl(qc_norm * dd))
        return mcn0_cov.cov()


    def get_ampl_x_input(self, mc_sims=None):
        """Returns cross-correlation of phi-maps to input lensing maps.

            Uses qlms_x_i library of parfile

        """
        qlmi = self.parfile.qlms_x_in
        if mc_sims is None: mc_sims = np.unique(np.concatenate([self.parfile.mc_sims_var, self.parfile.mc_sims_bias]))
        xin = utils.stats(self.nbins)
        qnorm = utils.cli(self.parfile.qresp_dd.get_response(self.k1, self.ksource))
        for i, idx in utils.enumerate_progress(mc_sims):
            qi = qlmi.get_sim_qcl(self.k1, idx)
            xin.add(self._get_binnedcl(qnorm * qi) / self.fid_bandpowers)
        return xin

