
import imp, os
import numpy as np
import pickle as pk

import argparse
import time
from subprocess import call

from plancklens2018 import utils
from plancklens2018 import mpi

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

VERSION = 'FFP10_vJan18'

if __name__ == '__main__':
    PL2018 = os.environ['PL2018']

    outdir_windows = PL2018 + "/likelihoods%s/%s/%s" % (VERSION, prefix, args.bin_type)
    outdir_bandpowers = PL2018 + "/likelihoods%s/%s/%s" % (VERSION, prefix, args.bin_type)
    #FIXME:
    dN1_cache = PL2018 + "/likelihoods%s/%s/temp/dmatsN1.pk" % (VERSION, prefix)  # bin_type indep.

    lmaxphi = 2048
    lmaxphi_N1 = 2500
    lmaxcmb = 2048
    kresp = 'p'
    # make the deltas.
    bin_lmins, bin_lmaxs,bins_center = get_blbubc(args.bin_type)
    nbins = len(bin_lmaxs)
    assert np.min(bin_lmins >= 2), 'check that this works OK'
    assert (lmaxphi, lmaxcmb, lmaxphi_N1) == (2048, 2048, 2500), (lmaxphi, lmaxcmb, lmaxphi_N1)

    cl_unl = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lenspotentialCls.dat'))
    cl_len = utils.camb_clfile(os.path.join(PL2018, 'inputs','cls','FFP10_wdipole_lensedCls.dat'))

    clpp_fid = np.copy(cl_unl.clpp[:lmaxphi + 1])
    cltt_fid = np.copy(cl_len.cltt[:lmaxcmb + 1])
    clte_fid = np.copy(cl_len.clte[:lmaxcmb + 1])
    clee_fid = np.copy(cl_len.clee[:lmaxcmb + 1])
    assert np.all(clpp_fid == par.cl_unl.clpp[:lmaxphi + 1])
    assert np.all(cltt_fid == par.cl_len.cltt[:lmaxcmb + 1])
    assert np.all(clte_fid == par.cl_len.clte[:lmaxcmb + 1])
    assert np.all(clee_fid == par.cl_len.clee[:lmaxcmb + 1])


    # Make fiducial bandpowers so that plots look like the usual thing
    # This acts on cpp (NB, this is not exactly kappa)
    kappaswitch = (np.arange(0, lmaxphi + 1, dtype=float) * (np.arange(1, lmaxphi + 2))) ** 2 / (2. * np.pi) * 1e7
    clkk_fid = clpp_fid * kappaswitch


    qlms_dd = par.qlms_dd
    qcls_dd = par.qcls_dd
    qc_resp = qcls_dd.get_response(args.kA, kresp)[:lmaxphi + 1]
    qc_norm = qcls_dd.get_response(args.kA, kresp).inverse()[:lmaxphi + 1]
    assert np.all(qc_resp == qlms_dd.get_response(args.kA, kresp)[:lmaxphi + 1] ** 2)

    # This is how DH built is fid. bandpowers:
    #norm = (np.arange(0, 2049) * (np.arange(0, 2049) + 1.)) ** 2 / (2. * np.pi)
    #bweight = clpp_fid[0:2049] * vlpp_inv[0:2049] / norm
    #fid_bandpowers = np.zeros((nbins))
    #for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
    #    den = np.sum(bweight[lmin:lmax + 1])
    #    fid_bandpowers[i] = np.sum(norm[lmin:lmax + 1] * (bweight[lmin:lmax + 1] / den) * clpp_fid[lmin:lmax + 1])

    # Make dcl bandpowers.  We want to be miminum variance estimate of the kappa amplitude, but acting on cpp
    # unbinned power.
    # Weights are ~ inverse kappa-like variance. The binning function (acting on cpp !) is Ckk * vlpp_inv * kappaswitch
    # With a normalization such that unity is expected for fiducial cpp.
    # We then multiply with fid_bandpowers to make it look like kappa.
    vlpp_inv = qc_resp * (2 * np.arange(lmaxphi + 1) + 1) * (0.5 * qcls_dd.fsky1234)
    vlpp_inv[np.where(kappaswitch != 0)] /= kappaswitch[np.where(kappaswitch != 0)] ** 2
    vlpp_den = [np.sum([clkk_fid[l] ** 2 * vlpp_inv[l] for l in range(lmin, lmax + 1)]) for lmin, lmax in
                zip(bin_lmins, bin_lmaxs)]

    fid_bandpowers = np.ones(nbins)  # We will renormalize that as soon as l_av is calculated.
    def get_BiL(i, L):  # Bin i window function to be applied to cLpp-like arrays as just described
        ret = (fid_bandpowers[i] / vlpp_den[i]) * vlpp_inv[L] * clkk_fid[L] * kappaswitch[L]
        ret *= (L >= bin_lmins[i]) * (L <= bin_lmaxs[i])
        return ret


    lav = np.zeros(len(bins_center))
    for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
        w_lav = 1. / np.arange(lmin, lmax + 1) ** 2 / np.arange(lmin + 1, lmax + 2) ** 2
        lav[i] = np.sum(np.arange(lmin, lmax + 1) * w_lav * get_BiL(i, np.arange(lmin, lmax + 1))) / np.sum(
            w_lav * get_BiL(i, np.arange(lmin, lmax + 1)))

    fid_bandpowers = np.interp(lav, np.arange(lmaxphi + 1, dtype=float), clkk_fid)


    def get_binnedcl(cl):
        assert len(cl) > max(bin_lmaxs)
        ret = np.zeros((nbins))
        for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
            ret[i] = np.sum(get_BiL(i, np.arange(lmin, lmax + 1)) * cl[lmin:lmax + 1])
        return ret


    if mpi.rank == 0 and args.do_windows:
        [dmat_cltt, dmat_clee, dmat_clte, dmat_clpp] = pk.load(open(dN1_cache, 'r'))
        print("Loaded cached ", dN1_cache)

        # The following window functions applies to Dl not Cl for the not-pp spectra and L ** 2 (L+1) ** 2 / (2 pi) for pp.
        lscmb = np.arange(2, lmaxcmb + 1)
        lsphi = np.arange(2, lmaxphi_N1 + 1)
        binned_dcls = {}
        for i, (lmin, lmax) in enumerate(zip(bin_lmins, bin_lmaxs)):
            binned_dcls[i] = np.zeros((4, max(lmaxphi_N1, lmaxcmb) + 1))
            for L in range(lmin, lmax + 1):
                binned_dcls[i][0, 2:lmaxcmb + 1] += (qc_norm[L] * dmat_cltt[L, 2:lmaxcmb + 1] * (2. * np.pi) / (
                lscmb * (lscmb + 1.))) * get_BiL(i, L)
                binned_dcls[i][1, 2:lmaxcmb + 1] += (qc_norm[L] * dmat_clee[L, 2:lmaxcmb + 1] * (2. * np.pi) / (
                lscmb * (lscmb + 1.))) * get_BiL(i, L)
                binned_dcls[i][2, 2:lmaxcmb + 1] += (qc_norm[L] * dmat_clte[L, 2:lmaxcmb + 1] * (2. * np.pi) / (
                lscmb * (lscmb + 1.))) * get_BiL(i, L)
                binned_dcls[i][3, 2:lmaxphi_N1 + 1] += (qc_norm[L] * dmat_clpp[L, 2:lmaxphi_N1 + 1] * (2. * np.pi) / (
                lsphi * (lsphi + 1.)) ** 2) * get_BiL(i, L)

        lscmb = np.arange(0, lmaxcmb + 1)
        lsphi = np.arange(0, lmaxphi_N1 + 1)
        linear_correction_fiducial = np.zeros(nbins, dtype=float)
        for i in range(0, nbins):
            linear_correction_fiducial[i] += np.sum(
                binned_dcls[i][0, :lmaxcmb + 1] * cltt_fid[0:lmaxcmb + 1] * lscmb * (lscmb + 1.) / (2. * np.pi))
            linear_correction_fiducial[i] += np.sum(
                binned_dcls[i][1, :lmaxcmb + 1] * clee_fid[0:lmaxcmb + 1] * lscmb * (lscmb + 1.) / (2. * np.pi))
            linear_correction_fiducial[i] += np.sum(
                binned_dcls[i][2, :lmaxcmb + 1] * clte_fid[0:lmaxcmb + 1] * lscmb * (lscmb + 1.) / (2. * np.pi))
            linear_correction_fiducial[i] += np.sum(
                binned_dcls[i][3, :lmaxphi_N1 + 1] * clpp_fid[0:lmaxphi_N1 + 1] * (lsphi * (lsphi + 1.)) ** 2 / (
                2. * np.pi))
        header = 'Lensing fiducial correction for \n %s \n bin type: %s\n' % (prefix, args.bin_type)
        header += '-------------- \n'
        header += time_str() + ' by ' + __file__ + '\n'
        header += '--------------'
        np.savetxt(outdir_windows + "/lensing_fiducial_correction.dat",
                   np.array([np.arange(1, nbins + 1), linear_correction_fiducial]).transpose(),
                   header=header, fmt=['%5i', '%.12e'])
        print("Cached ", outdir_windows + "/lensing_fiducial_correction.dat")
        for i in range(0, nbins):
            header = 'Window functions for \n %s \n bin type: %s\n' % (prefix, args.bin_type)
            header += '-------------- \n'
            header += time_str() + ' by ' + __file__ + '\n'
            header += '-------------- \n'
            header += 'Bin %s in %s : %s - %s \n' % (i + 1, nbins, bin_lmins[i], bin_lmaxs[i])
            header += 'L, dM / dDL TT, dM / dDL EE, dM / dDL TE, dM/dDL pp, BL'
            save_arr = np.zeros((6, lmaxphi_N1 + 1), dtype=float)
            save_arr[0] = np.arange(lmaxphi_N1 + 1)
            save_arr[1] = binned_dcls[i][0]
            save_arr[2] = binned_dcls[i][1]
            save_arr[3] = binned_dcls[i][2]
            save_arr[4] = binned_dcls[i][3]
            save_arr[5, :lmaxphi + 1] = get_BiL(i, np.arange(lmaxphi + 1))
            np.savetxt(outdir_windows + "/window%d.dat" % (i + 1), save_arr.transpose(), header=header,
                       fmt=['%5i'] + 5 * ['%.12e'])
            print("Cached ", outdir_windows + "/window%d.dat" % (i + 1))


    if mpi.rank == 0 and args.do_bandpowers:
        from utils import stats

        print(" **** Building bandpowers and covariance matrix")
        getme = lp.analysis_lib.get_me_things_from_database(args.parfile[0], args.kA)
        assert np.all(getme.qc_norm[:lmaxphi + 1] == qc_norm[:lmaxphi + 1])
        flags = {'dat_N0': 'RDN0', 'sim_N0': 'MCN0', 'cov_N0': 'Nhl', 'MC': 'bmMC','usePS': True, 'useN1': args.N1}
        datN0 = get_binnedcl(getme.get_dat_pp_rdn0())
        MCN0 = get_binnedcl(getme.get_MC_pp_n0())
        N1 = get_binnedcl(getme.get_N1_pp_qcl(useN1=flags['useN1']))
        MC = get_binnedcl(getme.get_MC_pp_qcl(useN1=flags['useN1']))
        bmMC = 1./ (1 + MC / fid_bandpowers) # Binned multiplicative MC correction.

        PS = get_binnedcl(getme.get_PS_corr(lmin_ss_s4=args.lmin_S4, lmax_ss_s4=args.lmax_S4))
        MF = get_binnedcl(getme.get_mean_field_cls(mc_sims=getme.mc_sims_bias))
        dat = (get_binnedcl(getme.get_dat_pp_qcl()) - datN0 - N1 - PS * flags['usePS']) * bmMC
        # Binned MC:
        Ahat = dat / fid_bandpowers
        S4_data = getme.get_S4(lmin_ss_s4=args.lmin_S4, lmax_ss_s4=args.lmax_S4)  # PS correction
        # get covariance with Nhl
        bi_stats = stats(nbins)
        for idx in getme.mc_sims_var:
            bi_stats.add(get_binnedcl(getme.get_sim_pp_qcl(idx) - getme.get_sim_pp_nhl(idx)))
        bias_hartlap = 1./ ((bi_stats.N - bi_stats.size - 2.) / (bi_stats.N - 1))  # Hartlap & al correction factor
        s2_MCs = 1. + (2. / len(getme.mc_sims_bias) + 9. / len(getme.mc_sims_var)) # Monte Carlo error due to finite number of sims: s2 ~ (2 / NMF + 9 / Nvar) * s2_BP

        def get_cov():
            ret = bi_stats.cov()
            ret *= s2_MCs # MC errors
            ret *= bias_hartlap # debiasing
            ret *= np.outer(bmMC,bmMC) #mulitplicative MC
            return ret
        cov = get_cov()
        devinsigmas = (Ahat - 1. * (args.kA[0] == 'p')) / np.sqrt(np.diag(cov)) * fid_bandpowers
        header = 'Band-powers and covariance for \n %s \n bin type: %s\n' % (prefix, args.bin_type)
        header += '-------------- \n'
        header += time_str() + ' by ' + __file__ + '\n'
        header += '-------------- \n'
        header += 'sims for mean field: %s\n' % len(getme.mc_sims_bias)
        header += 'sims for covariance: %s\n' % len(getme.mc_sims_var)
        header += 'Point-source amplitude S4: %.3g +- %0.3g (stat), signif of %.3f sigma\n' % (S4_data[0], np.std(S4_data[3]), S4_data[0] / np.std(S4_data[3]))
        header += 'S4 fit range %s - %s\n' % (args.lmin_S4, args.lmax_S4)
        header += 'Inverse cov. debiasing factor (incl) : %.5f\n' % bias_hartlap
        header += 'MCs bias est. error (incl) : %.5f\n' % s2_MCs
        header += 'Chi2 at fiducial: %.3f (full cov) \n'%(np.sum( (dat -fid_bandpowers) * np.dot(np.linalg.inv(cov),dat -fid_bandpowers)))
        header += 'Reconstruction flags:\n'
        header += str(flags) + '\n'
        header += '-------------- \n'
        header += 'Window functions cls (if produced) in ' + outdir_windows + '\n'
        header += '--------------\n'

        save_arr = np.zeros((nbins + len([dat, Ahat, devinsigmas, datN0, MCN0, N1, bmMC, PS, MF, fid_bandpowers]), nbins))
        for i, (bp, lab) in enumerate(zip([dat, Ahat, devinsigmas, datN0, MCN0, N1, bmMC, PS, MF, fid_bandpowers],
                                          ['BP', 'Ahat', 'sigdev', flags['dat_N0'], 'MCN0', 'N1', 'MC', 'PS', 'MF',
                                           'BPfid'])):
            header += '%2s' % lab if i == 0 else '%7s' % lab
            save_arr[i] = bp
        header += ' BP-covariance'
        save_arr[len(save_arr) - nbins:, :] = cov
        np.savetxt(outdir_bandpowers + '/bandpowers.dat', save_arr.transpose(),
                   fmt=(len(save_arr) - nbins) * ['%.4f'] + nbins * ['%.12e'], header=header)
        print("Cached ", outdir_bandpowers + '/bandpowers.dat')

    mpi.barrier()
    mpi.finalize()
