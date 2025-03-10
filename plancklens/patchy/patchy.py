"""This module contain methods for QE-related analytical predictions on data or filtering with inhomogeneous noise

"""
import healpy as hp
import numpy as np
from plancklens import utils, nhl, qresp
from plancklens.helpers import cachers
from plancklens.qcinv.util import read_map
def _read_map(m):
    return read_map(m)


def get_patchy_N0s(qekey_in, npatches, pixivmap_t, pixivmap_p, cls_unl, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin_ivf, lmax_ivf, lmax_qlm, transf,
                  rvmap_uKamin_t_data=None, rvmap_uKamin_p_data=None, joint_TP=False,
                  nlevt_fid=None, nlevp_fid=None, cacher=cachers.cacher_mem(), source='p', patch_method='percentiles', verbose=False):
    """Collects the effective reconstruction noise levels for different filtering and spectrum weighting schemes

        Args:
            qekey_in: QE anisotroy key
            npatches: the variance map will be split into this number of regions of equal sky areas
            pixivmap_t: inverse temperature noise pixel variance map used for the T. filtering
            pixivmap_p: inverse polarization noise pixel variance map used for the Pol. filtering
            cls_unl: unlensed CMB dict
            cls_cmb_dat: CMB spectra dict entering the data maps
            cls_cmb_filt: CMB spectra dict entering the filtering steps
            cls_weight: CMB spectra dict entering the QE weights (numerators)
            lmin_ivf: minimal CMB mutlipole
            lmax_ivf: maximal CMB multipole
            lmax_qlm: maximal QE multipole
            transf: CMB transfer function cl
            rvmap_uKamin_t_data (optional): set this to the data temperature noise map (in uK amin), if different from the one defining the filtering
            rvmap_uKamin_p_data (optional): set this to the data polarisation noise map (in uK amin), if different from the one defining the filtering
            joint_TP: set this to true if temperature and polarization are jointly filtered before building the QE
            nlevt_fid: set this to the fiducial temperature noise value to use for the single full-sky normalization
            nlevp_fid: set this to the fiducial polarisation noise value to use for the single full-sky normalization
            cacher: can use this to store results (descriptors only use the noise levels, joint_TP and qe_keys though)
            source: anistropy source for the responses calculations

        Returns:
            N0s: a dict of N0 arrays for different filtering and spectr weighting types
            MCcorr: prediction of the Monte-Carlo correction of the spectrum for inhom. filtering
            cMCcorr: Same for the cross-spectrum to the true lensing


    """

    assert qekey_in[0] in ['p', 'x'], 'fix curl fiducial and MC correction'
    qe_key = 'p' + qekey_in[1:]

    if (not joint_TP) and qe_key == 'ptt': # dont need pol partitioning here
        nlevst_ftl, nlevst_data, _nlevt_fid, fskiest, masks = mk_patches(npatches, pixivmap_t, rvmap_uKamin_data=rvmap_uKamin_t_data, method=patch_method, verbose=verbose)
        nlevsp_ftl, nlevsp_data, _nlevp_fid, fskiesp = (1e30 * np.ones_like(nlevst_ftl), 1e30 * np.copy(nlevst_data), 1e30, fskiest.copy())
    elif (not joint_TP) and qe_key == 'p_p':# dont need T  here
        nlevsp_ftl, nlevsp_data, _nlevp_fid, fskiesp, masks = mk_patches(npatches, pixivmap_p, rvmap_uKamin_data=rvmap_uKamin_p_data, method=patch_method, verbose=verbose)
        nlevst_ftl, nlevst_data, _nlevt_fid, fskiest = (1e30 * np.ones_like(nlevsp_ftl), 1e30 * np.copy(nlevsp_data), 1e30, fskiesp.copy())
    else:
        nlevst_ftl, nlevst_data, _nlevt_fid, fskiest, masks = mk_patches(npatches, pixivmap_t, rvmap_uKamin_data=rvmap_uKamin_t_data, method=patch_method, verbose=verbose)
        nlevsp_ftl, nlevsp_data, _nlevp_fid, fskiesp, masks = mk_patches(npatches, pixivmap_p, rvmap_uKamin_data=rvmap_uKamin_p_data, method=patch_method, verbose=verbose)
    if nlevt_fid is None: nlevt_fid = _nlevt_fid
    if nlevp_fid is None: nlevp_fid = _nlevp_fid

    assert np.allclose(fskiest, fskiesp, atol=1e-6), (np.array(fskiest)-np.array(fskiesp), fskiesp)
    fskies = fskiest

    cpp = cls_unl['pp'][:lmax_qlm+1]
    rid = 0 if qekey_in[0] == 'p' else 1
    if qekey_in[0] == 'x':
        cpp *= 0.

    rfid = get_responses(qe_key, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin_ivf, lmax_ivf, lmax_qlm, transf, [nlevt_fid], [nlevp_fid],
                  joint_TP=joint_TP, cacher=cacher, source=source)[0]
    resps = get_responses(qe_key, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin_ivf, lmax_ivf, lmax_qlm, transf, nlevst_ftl, nlevsp_ftl,
                  joint_TP=joint_TP, cacher=cacher, source=source)
    nhls_pds = get_nhls(qe_key, qe_key, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin_ivf, lmax_ivf, lmax_qlm, transf, nlevst_ftl, nlevst_data, nlevsp_ftl, nlevsp_data,
                    joint_TP=joint_TP,cacher=cacher)
    nhls_fds = get_nhls(qe_key, qe_key, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin_ivf, lmax_ivf, lmax_qlm, transf, [nlevt_fid] * npatches, nlevst_data, [nlevp_fid] * npatches, nlevsp_data,
                    joint_TP=joint_TP,cacher=cacher)

    labels = ['hom-filt, no-rew', 'hom-filt, mv-rew', 'inhom-filt, no-rew', 'inhom-filt, mv-rew']
    N0s = {q: np.zeros(lmax_qlm + 1, dtype=float) for q in labels}
    MCcorr_vmap = np.zeros(lmax_qlm + 1, dtype=float)
    cMCcorr_vmap = np.zeros(lmax_qlm + 1, dtype=float)

    fsky_tot = np.sum(fskies)
    print('total observed sky area %.3f'%fsky_tot)
    print('Using nlev fids %.3f %.3f'%(nlevt_fid, nlevp_fid))

    rfidi = utils.cli(rfid[rid])

    for i, (fsky, resp, nhl_pd, nhl_fd) in enumerate(zip(fskies, resps, nhls_pds, nhls_fds)):
        fp_f = fsky / fsky_tot
        Rp_Rf = resp[rid] * rfidi
        N0s['hom-filt, no-rew'] += fp_f * (cpp + nhl_fd[rid] * rfidi ** 2) ** 2
        # : spectrum of homo. filtered map without any spectra reweighting
        N0s['inhom-filt, no-rew'] += fp_f * (Rp_Rf ** 2 * cpp + nhl_pd[rid] * rfidi ** 2) ** 2
        # : spectrum of inhomo. filtered map without any spectra reweighting
        N0s['hom-filt, mv-rew'] += fp_f * utils.cli((cpp + nhl_fd[rid] * rfidi ** 2) ** 2)
        # : inverse variance weighting of homog filtered map
        N0s['inhom-filt, mv-rew'] += fp_f * utils.cli((cpp + nhl_pd[rid] * rfidi ** 2 * utils.cli(Rp_Rf ** 2)) ** 2)
        # : inverse variance weighting of inhomog filtered map

        MCcorr_vmap += fp_f * Rp_Rf ** 2
        cMCcorr_vmap += fp_f * Rp_Rf

    N0s['hom-filt, mv-rew'] = utils.cli(N0s['hom-filt, mv-rew'])
    N0s['inhom-filt, mv-rew'] = utils.cli(N0s['inhom-filt, mv-rew'])
    N0s['inhom-filt, no-rew'] *= utils.cli(MCcorr_vmap ** 2)
    for spec in N0s.values():
        spec[:] = np.sqrt(spec) - cpp
    return N0s, MCcorr_vmap, cMCcorr_vmap

def mk_patches(Np, pix_ivmap, rvmap_uKamin_data=None, ret_masks=False, method='percentiles', verbose=False):
    """Splits the variance maps into equal-area regions with different noise levels

        Args:
            Np: desired number of patches
            pix_ivmap: input inverse pixel variance map used for the filtering
            rvmap_uKamin_data: root variance map in uK amin of the data (if different from pix_ivmap)
            ret_masks: returns the defined series of masks if set
            method: which method should be used to perform the calculation
                     percentiles: equal sky areas regions
                     linear: equally spaced nlevs in uK (this is best in terms of convergence towards an integral)


    """

    mask = _read_map(pix_ivmap) > 0
    npix = mask.size
    nside = hp.npix2nside(npix)
    nlev_map = utils.cli(np.sqrt(_read_map(pix_ivmap))) * np.sqrt(hp.nside2pixarea(nside)) / np.pi * 60 * 180.
    nlev_map_mask = nlev_map # map used to define the sky areas
    if np.unique(nlev_map_mask[np.where(mask)]).size <= 1 :
        assert rvmap_uKamin_data is not None, ('uniform map ? this wont work')
        nlev_map_mask = _read_map(rvmap_uKamin_data)
        mask = _read_map(nlev_map_mask) > 0
        assert np.unique(nlev_map_mask[np.where(mask)]).size > 1
    if method == 'percentiles':
        edges = np.percentile(nlev_map_mask[np.where(mask)], np.linspace(0, 100, Np + 1))
    elif method == 'linear':
        edges = np.linspace(np.min(nlev_map_mask[np.where(mask)]), np.max(nlev_map_mask[np.where(mask)]), Np + 1)
    elif method == 'linear_vmap':
        ninv = _read_map(pix_ivmap)
        edges = np.linspace(np.min(ninv[np.where(mask)]), np.max(ninv[np.where(mask)]), Np + 1)
        del ninv
        edges = 1./np.sqrt(edges[::-1])  * np.sqrt(hp.nside2pixarea(nside)) / np.pi * 60 * 180.
    else:
        assert 0, 'method ' + method + ' not implemented'
    edges[0] = -1.
    edges[-1] = 10000
    nlevs = []  # from filtering variance map
    nlevs_data = []  # from data variance map
    fskies = []
    masks = []
    for i in range(1, Np + 1):
        this_mask = (nlev_map > edges[i - 1]) & (nlev_map <= edges[i])
        this_fsky = np.mean(mask * this_mask)
        if this_fsky > 0:
            nlevs.append(np.mean(nlev_map[mask * this_mask]))
            fskies.append(this_fsky)
            if rvmap_uKamin_data is not None:
                nlevs_data.append(np.mean(_read_map(rvmap_uKamin_data[mask * this_mask])))
            if ret_masks:
                masks.append(this_mask * mask)
    if rvmap_uKamin_data is None:
        nlevs_data = nlevs
    nlev_fid = np.sqrt(4. * np.pi / npix / np.sum(_read_map(pix_ivmap)) * np.sum(mask)) * 180. * 60. / np.pi
    if verbose:
        for nf, nd in zip(nlevs, nlevs_data):
            print('%.2f (ftl)   %.2f (dat) uKamin' % (nf, nd))
        print('%.2f (fid)' %nlev_fid)
    return nlevs, nlevs_data, nlev_fid, fskies, masks

def get_nlev_fid(pix_ivmap):
    mask = _read_map(pix_ivmap) > 0
    nlev_fid = np.sqrt(4. * np.pi / mask.size / np.sum(_read_map(pix_ivmap)) * np.sum(mask)) * 180. * 60. / np.pi
    return nlev_fid

def get_fal(a, cl_len, nlev, transf, lmin, lmax):
    """Simple diagonal isotropic filter 

    """
    fal = utils.cli(cl_len.get(a + a)[:lmax + 1] + (nlev / 60. / 180. * np.pi) ** 2 / transf[:lmax + 1] ** 2)
    fal[:lmin] *= 0.
    return fal


def get_ivf_cls(cls_cmb_dat, cls_cmb_filt, lmin, lmax, nlevt_f, nlevp_f, nlevt_m, nlevp_m, transf, jt_tp=False):
    """inverse filtered spectra (spectra of Cov^-1 X) for CMB inverse-variance filtering


        Args:
            cls_cmb_dat: dict of cmb cls of the data maps
            cls_cmb_filt: dict of cmb cls used in the filtering matrix
            lmin: minimum multipole considered
            lmax: maximum multipole considered
            nlevt_f: fiducial temperature noise level used in the filtering in uK-amin
            nlevp_f: fiducial polarization noise level used in the filtering in uK-amin
            nlevt_m: temperature noise level of the data in uK-amin
            nlevp_m: polarization noise level of the data in uK-amin
            transf: CMB transfer function
            jt_tp: if set joint temperature-polarization filtering is performed. If not they are filtered independently

        Returns:
            dict of inverse-variance filtered maps spectra (for N0 calcs.)
            dict of filtering matrix spectra (for response calcs. This has no dependence on the data parts of the inputs)


    """
    ivf_cls = {}
    if not jt_tp:
        filt_cls_i = {}
        for a in ['t']:
            ivf_cls[a + a] = get_fal(a, cls_cmb_filt, nlevt_f, transf, lmin, lmax) ** 2 * utils.cli(get_fal(a, cls_cmb_dat, nlevt_m, transf, 0, lmax))
            filt_cls_i[a + a] = get_fal(a, cls_cmb_filt, nlevt_f, transf, lmin, lmax)
        for a in ['e', 'b']:
            ivf_cls[a + a] = get_fal(a, cls_cmb_filt, nlevp_f, transf, lmin, lmax) ** 2 * utils.cli(get_fal(a, cls_cmb_dat, nlevp_m, transf, 0, lmax))
            filt_cls_i[a + a] = get_fal(a, cls_cmb_filt, nlevp_f, transf, lmin, lmax)
        ivf_cls['te'] = cls_cmb_dat['te'][:lmax + 1] * get_fal('e', cls_cmb_filt, nlevp_f, transf, lmin, lmax) * get_fal('t', cls_cmb_filt,  nlevt_f, transf,   lmin, lmax)
        return ivf_cls, filt_cls_i
    else:
        filt_cls = np.zeros((3, 3, lmax + 1), dtype=float)
        dat_cls = np.zeros((3, 3, lmax + 1), dtype=float)
        filt_cls[0, 0] = utils.cli(get_fal('t', cls_cmb_filt, nlevt_f, transf, lmin, lmax))
        filt_cls[1, 1] = utils.cli(get_fal('e', cls_cmb_filt, nlevp_f, transf, lmin, lmax))
        filt_cls[2, 2] = utils.cli(get_fal('b', cls_cmb_filt, nlevp_f, transf, lmin, lmax))
        filt_cls[0, 1, lmin:] = cls_cmb_filt['te'][lmin:lmax + 1]
        filt_cls[1, 0, lmin:] = cls_cmb_filt['te'][lmin:lmax + 1]
        dat_cls[0, 0] = utils.cli(get_fal('t', cls_cmb_dat, nlevt_m, transf, lmin, lmax))
        dat_cls[1, 1] = utils.cli(get_fal('e', cls_cmb_dat, nlevp_m, transf, lmin, lmax))
        dat_cls[2, 2] = utils.cli(get_fal('b', cls_cmb_dat, nlevp_m, transf, lmin, lmax))
        dat_cls[0, 1, lmin:] = cls_cmb_dat['te'][lmin:lmax + 1]
        dat_cls[1, 0, lmin:] = cls_cmb_dat['te'][lmin:lmax + 1]
        filt_cls_i = np.linalg.pinv(filt_cls.swapaxes(0, 2)).swapaxes(0, 2)
        return cls_dot(filt_cls_i, dat_cls, lmin, lmax), \
               {'tt':filt_cls_i[0,0], 'ee':filt_cls_i[1, 1], 'bb':filt_cls_i[2, 2], 'te':filt_cls_i[0, 1]}

def cls_dot(cls_fidi, cls_dat, lmin, lmax):
    zro = np.zeros(lmax + 1, dtype=float)
    ret = {'tt':zro.copy(), 'te':zro.copy(), 'ee':zro.copy(), 'bb':zro.copy()}
    for i in range(3):
        for j in range(3):
            ret['tt'] += cls_fidi[0, i] * cls_fidi[0, j] * cls_dat[i, j]
            ret['te'] += cls_fidi[0, i] * cls_fidi[1, j] * cls_dat[i, j]
            ret['ee'] += cls_fidi[1, i] * cls_fidi[1, j] * cls_dat[i, j]
            ret['bb'] += cls_fidi[2, i] * cls_fidi[2, j] * cls_dat[i, j]
    for cl in ret.values():
        cl[:lmin] *= 0
    return ret

def get_responses(qe_key, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin, lmax, lmax_qlm, transf, nlevts_filt, nlevps_filt,
                  joint_TP=False, cacher=cachers.cacher_mem(), source='p'):
    """Collects estimator responses for a list of filtering noise levels


        Args:
            qe_key: QE estimator key
            cls_cmb_dat: CMB cls of the data maps
            cls_cmb_filt: CMB cls used for the filtering
            cls_weight: CMB cls in the QE weights
            lmin: minimum CMB multipole considered
            lmax: maximum CMB multipole considered
            lmax_qlm: QE output lmax
            transf: CMB transfer function
            nlevts_filt: list or array of filtering temperature noise levels
            nlevps_filt: list or array of filtering polarization noise levels
            joint_TP: uses joint temperature and polarization filtering if set, separate if not
            cacher: can be used to store results
            source: QE response anisotropy source (defaults to lensing)

        Returns:
            lists of responses (GG, CC, GC CG for spin-weight QE)

        Note:
            Results may be stored with the cacher but only the filtering noise levels, QE keys and joint_TP are differentiated in the filename

    """
    resps = []
    for i, (nlevt_f, nlevp_f) in utils.enumerate_progress(list(zip(nlevts_filt, nlevps_filt)), 'collecting responses'):
        fname = 'vmapresps%s_%s_%s' % ('jTP' * joint_TP, qe_key, qe_key) + utils.clhash(np.array([nlevt_f, nlevp_f]), dtype=np.float32)
        if not cacher.is_cached(fname):
            cls_filt_i = get_ivf_cls(cls_cmb_dat, cls_cmb_filt, lmin, lmax, nlevt_f, nlevp_f, nlevt_f, nlevp_f, transf, jt_tp=joint_TP)[1]
            this_resp =  qresp.get_response(qe_key, lmax, source, cls_weight, cls_cmb_dat, cls_filt_i, lmax_qlm=lmax_qlm)
            cacher.cache(fname, this_resp)
        resps.append(np.array(cacher.load(fname)))
    return np.array(resps)

def get_nhls(qe_key1, qe_key2, cls_cmb_dat, cls_cmb_filt, cls_weight, lmin, lmax, lmax_qlm, transf, nlevts_filt, nlevts_map, nlevps_filt, nlevps_map,
             joint_TP=False, cacher=cachers.cacher_mem()):
    """Collects unnormalized estimator noise levels for a list of filtering noise levels and data map noise levels


        Args:
            qe_key1: first QE estimator key
            qe_key2: second QE estimator key
            cls_cmb_dat: CMB cls of the data maps
            cls_cmb_filt: CMB cls used for the filtering
            cls_weight: CMB cls in the QE weights
            lmin: minimum CMB multipole considered
            lmax: maximum CMB multipole considered
            lmax_qlm: QE output lmax
            transf: CMB transfer function
            nlevts_filt: list or array of filtering temperature noise levels
            nlevts_map: list or array of data map temperature noise levels
            nlevps_filt: list or array of filtering polarization noise levels
            nlevps_map: list or array of data maptemperature noise levels
            joint_TP: uses joint temperature and polarization filtering if set, separate if not
            cacher: can be used to store results

        Returns:
            lists of reconstruction noise levels (GG, CC, GC CG for spin-weight QE)

        Note:
            Results may be stored with the cacher but only the filtering and data noise levels, QE keys and joint_TP are differentiated in the filename

    """
    Nhls = []
    for i, (nlevt_f, nlevt_m, nlevp_f, nlevp_m) in utils.enumerate_progress(list(zip(nlevts_filt, nlevts_map, nlevps_filt, nlevps_map)), 'collecting nhls'):
        fname = 'vmapnhl%s_%s_%s' % ('jTP' * joint_TP, qe_key1, qe_key2) + utils.clhash(np.array([nlevt_f, nlevt_m, nlevp_f, nlevp_m]))
        if not cacher.is_cached(fname):
            ivf_cls = get_ivf_cls(cls_cmb_dat, cls_cmb_filt, lmin, lmax, nlevt_f, nlevp_f, nlevt_m, nlevp_m, transf, jt_tp=joint_TP)[0]
            this_nhl = nhl.get_nhl(qe_key1, qe_key2, cls_weight, ivf_cls, lmax, lmax, lmax_out=lmax_qlm)
            cacher.cache(fname, this_nhl)
        Nhls.append(np.array(cacher.load(fname)))
    return np.array(Nhls)

