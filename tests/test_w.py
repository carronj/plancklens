from __future__ import print_function

import numpy as np
import healpy as hp
import os

import plancklens
from plancklens import utils
from plancklens import nhl, qresp
from plancklens.wigners import wigners

def test_w():
    cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')

    lmax_ivf = 500
    lmin_ivf = 100
    nlev_t = 35.
    nlev_p = 35. * np.sqrt(2.)
    beam_fwhm = 6.
    lmax_qlm = lmax_ivf

    for ksource in ['p', 'f']:
        if ksource in ['p', 'f']:
            qe_keys = [ksource + 'tt', ksource+'_p', ksource]
        elif ksource in ['a', 'a_p', 'stt']:
            qe_keys = [ksource]
        else:
            assert 0

        transf = hp.gauss_beam(beam_fwhm / 60. / 180. * np.pi, lmax=lmax_ivf)

        cls_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
        cls_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))

        fal_sepTP =  {'tt': utils.cli(cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2),
                      'ee': utils.cli(cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2),
                      'bb': utils.cli(cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)}


        cls_ivfs_sepTP = {'tt':fal_sepTP['tt'].copy(),
                          'ee':fal_sepTP['ee'].copy(),
                          'bb':fal_sepTP['bb'].copy(),
                          'te':cls_len['te'][:lmax_ivf + 1] * fal_sepTP['tt'] * fal_sepTP['ee']}
        cls_dat = {
            'tt': (cls_len['tt'][:lmax_ivf + 1] + (nlev_t / 60. /180. * np.pi) ** 2 / transf ** 2),
            'ee': (cls_len['ee'][:lmax_ivf + 1] + (nlev_p / 60. /180. * np.pi) ** 2 / transf ** 2),
            'bb': (cls_len['bb'][:lmax_ivf + 1] + (nlev_p / 60. /180. * np.pi) ** 2 / transf ** 2),
            'te':  np.copy(cls_len['te'][:lmax_ivf + 1]) }

        fal_jtTP = utils.cl_inverse(cls_dat)
        cls_ivfs_jtTP = utils.cl_inverse(cls_dat)

        for cls in [fal_sepTP, fal_jtTP, cls_ivfs_sepTP, cls_ivfs_jtTP]:
            for cl in cls.values():
                cl[:max(1, lmin_ivf)] *= 0.

        for qe_key in qe_keys:
            NG, NC, NGC, NCG = nhl.get_nhl(qe_key, qe_key, cls_weight, cls_ivfs_sepTP, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)
            RG, RC, RGC, RCG = qresp.get_response(qe_key, lmax_ivf, ksource, cls_weight, cls_len, fal_sepTP, lmax_qlm=lmax_qlm)
            if qe_key[1:] in ['tt', '_p']:
                assert np.allclose(NG[1:], RG[1:], rtol=1e-6), qe_key
                assert np.allclose(NC[2:], RC[2:], rtol=1e-6), qe_key
            assert np.all(NCG == 0.) and np.all(NGC == 0.) # for these keys
            assert np.all(RCG == 0.) and np.all(RGC == 0.)

        NG, NC, NGC, NCG = nhl.get_nhl(ksource, ksource, cls_weight, cls_ivfs_jtTP, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)
        RG, RC, RGC, RCG = qresp.get_response(ksource, lmax_ivf, ksource, cls_weight, cls_len, fal_jtTP, lmax_qlm=lmax_qlm)
        assert np.allclose(NG[1:], RG[1:], rtol=1e-6), ksource
        assert np.allclose(NC[2:], RC[2:], rtol=1e-6), ksource
        assert np.all(NCG == 0.) and np.all(NGC == 0.) # for these keys
        assert np.all(RCG == 0.) and np.all(RGC == 0.)

