# opfilt_pp
#
# operations and filters for polarization only c^-1
# S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}
#FIXME: hashes
import hashlib
import numpy  as np
import healpy as hp
from . import util
from . import util_alm
from . import template_removal
from . import dense


def calc_prep(maps, s_cls, n_inv_filt):
    qmap, umap = np.copy(maps[0]), np.copy(maps[1])
    assert (len(qmap) == len(umap))
    lmax = len(n_inv_filt.b_transf) - 1
    npix = len(qmap)

    n_inv_filt.apply_map([qmap, umap])
    elm, blm = hp.map2alm_spin([qmap, umap], 2, lmax=lmax)
    elm *= npix / (4. * np.pi)
    blm *= npix / (4. * np.pi)
    hp.almxfl(elm, n_inv_filt.b_transf, inplace=True)
    hp.almxfl(blm, n_inv_filt.b_transf, inplace=True)
    return util_alm.eblm([elm, blm])


def apply_fini(alm, s_cls, n_inv_filt):
    sfilt = alm_filter_sinv(s_cls)
    ret = sfilt.calc(alm)
    alm.elm[:] = ret.elm[:]
    alm.blm[:] = ret.blm[:]

class dot_op:
    def __init__(self, lmax=None):
        self.lmax = lmax

    def __call__(self, alm1, alm2):
        lmax1 = alm1.lmax
        lmax2 = alm2.lmax

        if self.lmax is not None:
            lmax = self.lmax
        else:
            assert lmax1 == lmax2
            lmax = lmax1

        tcl = util_alm.alm_cl_cross(alm1.elm, alm2.elm) + util_alm.alm_cl_cross(alm1.blm, alm2.blm)

        return np.sum(tcl[2:] * (2. * np.arange(2, lmax + 1) + 1))


class fwd_op:
    def __init__(self, s_cls, n_inv_filt):
        self.s_inv_filt = alm_filter_sinv(s_cls)
        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'s_inv_filt': self.s_inv_filt.hashdict(),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, alm):
        return self.calc(alm)

    def calc(self, alm):
        nlm = alm * 1.0
        self.n_inv_filt.apply_alm(nlm)
        slm = self.s_inv_filt.calc(alm)
        return nlm + slm


# ===

class pre_op_diag:
    def __init__(self, s_cls, n_inv_filt):
        s_inv_filt = alm_filter_sinv(s_cls)
        assert ((s_inv_filt.lmax + 1) >= len(n_inv_filt.b_transf))

        ninv_fel, ninv_fbl = n_inv_filt.get_febl()

        lmax = len(n_inv_filt.b_transf) - 1

        flmat = s_inv_filt.slinv[0:lmax + 1, :, :]

        for l in range(lmax + 1):
            flmat[l, 0, 0] += ninv_fel[l]
            flmat[l, 1, 1] += ninv_fbl[l]
            flmat[l, :, :] = np.linalg.pinv(flmat[l, :, :].reshape((2, 2)))
        self.flmat = flmat

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, alm):
        tmat = self.flmat
        relm = hp.almxfl(alm.elm, tmat[:, 0, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 0, 1], inplace=False)
        rblm = hp.almxfl(alm.elm, tmat[:, 1, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 1, 1], inplace=False)
        return util_alm.eblm([relm, rblm])


def pre_op_dense(lmax, fwd_op, cache_fname=None):
    return dense.pre_op_dense_pp(lmax, fwd_op, cache_fname=cache_fname)

class alm_filter_sinv:
    def __init__(self, s_cls):
        lmax = s_cls.lmax
        slmat = np.zeros((lmax + 1, 2, 2))  # matrix of EB correlations at each l.
        slmat[:, 0, 0] = s_cls.get('ee', np.zeros(lmax + 1))
        slmat[:, 0, 1] = s_cls.get('eb', np.zeros(lmax + 1))
        slmat[:, 1, 1] = s_cls.get('bb', np.zeros(lmax + 1))
        slmat[:, 1, 0] = slmat[:, 0, 1]

        slinv = np.zeros((lmax + 1, 2, 2))
        for l in range(0, lmax + 1):
            slinv[l, :, :] = np.linalg.pinv(slmat[l])

        self.lmax = lmax
        self.slinv = slinv

    def calc(self, alm):
        tmat = self.slinv

        relm = hp.almxfl(alm.elm, tmat[:, 0, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 0, 1], inplace=False)
        rblm = hp.almxfl(alm.elm, tmat[:, 1, 0], inplace=False) + hp.almxfl(alm.blm, tmat[:, 1, 1], inplace=False)
        return util_alm.eblm([relm, rblm])

    def hashdict(self):
        return {'slinv': hashlib.sha1(self.slinv.flatten().view(np.uint8)).hexdigest()}


class alm_filter_ninv(object):
    def __init__(self, n_inv, b_transf, marge_maps=[]):
        self.n_inv = []
        for i, tn in enumerate(n_inv):
            if isinstance(tn, list):
                n_inv_prod = util.load_map(tn[0][:])
                if (len(tn) > 1):
                    for n in tn[1:]:
                        n_inv_prod = n_inv_prod * util.load_map(n[:])
                self.n_inv.append(n_inv_prod)
            else:
                self.n_inv.append(util.load_map(n_inv[i]))
        n_inv = self.n_inv

        assert (len(n_inv) == 1) or len(n_inv) == 3, len(n_inv)

        if len(n_inv) == 3:
            assert len(marge_maps) == 0
            # Did not check if the template calculation works for len(ninv) == 3.

        npix = len(n_inv[0])
        nside = hp.npix2nside(npix)
        for n in n_inv[1:]:
            assert (len(n) == npix)

        templates_p = [];
        templates_p_hash = []
        for tmap in [util.load_map(m) for m in marge_maps]:
            assert (npix == len(tmap))
            templates_p.append(template_removal.template_map_p(tmap))
            templates_p_hash.append(hashlib.sha1(tmap.view(np.uint8)).hexdigest())

        if len(templates_p) != 0:
            nmodes = np.sum([t.nmodes for t in templates_p])
            modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(templates_p)]))
            modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in templates_p]))

            Pt_Nn1_P = np.zeros((nmodes, nmodes))
            for ir in range(0, nmodes):
                tmap = np.copy(n_inv[0])
                templates_p[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates_p[0:modes_idx_t[ir] + 1]:
                    Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(tmap)
                    Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                    ic += tc.nmodes

            self.Pt_Nn1_P_inv = np.linalg.inv(Pt_Nn1_P)

        self.n_inv = n_inv
        self.b_transf = b_transf[:]

        self.templates_p = templates_p
        self.templates_p_hash = templates_p_hash

        self.npix = npix
        self.nside = nside

    def get_febl(self):
        if False:
            pass
        elif len(self.n_inv) == 1:  # TT, 1/2(QQ+UU)
            n_inv_cl_p = np.sum(self.n_inv[0]) / (4.0 * np.pi) * self.b_transf ** 2

            return n_inv_cl_p, n_inv_cl_p
        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            n_inv_cl_p = np.sum(0.5 * (self.n_inv[0] + self.n_inv[2])) / (4.0 * np.pi) * self.b_transf ** 2

            return n_inv_cl_p, n_inv_cl_p
        else:
            assert 0

    def hashdict(self):
        return {'n_inv': [hashlib.sha1(n.view(np.uint8)).hexdigest() for n in self.n_inv],
                'b_transf': hashlib.sha1(self.b_transf.view(np.uint8)).hexdigest(),
                'templates_p_hash': self.templates_p_hash}

    def degrade(self, nside):
        if nside == self.nside:
            return self
        else:
            marge_maps_p = []
            n_marge_maps_p = len(self.templates_p)
            if n_marge_maps_p > 0:
                marge_maps_p = [hp.ud_grade(ti.map, nside) for ti in self.templates_p[0:n_marge_maps_p]]

            return alm_filter_ninv([hp.ud_grade(n, nside, power=-2) for n in self.n_inv], self.b_transf, marge_maps_p)

    def apply_alm(self, alm):
        # applies Y^T N^{-1} Y
        lmax = alm.lmax

        hp.almxfl(alm.elm, self.b_transf, inplace=True)
        hp.almxfl(alm.blm, self.b_transf, inplace=True)
        qmap, umap = hp.alm2map_spin((alm.elm, alm.blm), self.nside, 2, lmax)

        self.apply_map([qmap, umap])  # applies N^{-1}
        npix = len(qmap)

        telm, tblm = hp.map2alm_spin([qmap, umap], 2, lmax=lmax)
        alm.elm[:] = telm * (npix / (4. * np.pi))
        alm.blm[:] = tblm * (npix / (4. * np.pi))

        hp.almxfl(alm.elm, self.b_transf, inplace=True)
        hp.almxfl(alm.blm, self.b_transf, inplace=True)

    def apply_map(self, amap):
        [qmap, umap] = amap

        # applies N^{-1}
        if False:
            pass
        elif len(self.n_inv) == 1:  # TT, QQ=UU
            qmap *= self.n_inv[0]
            umap *= self.n_inv[0]
        elif len(self.n_inv) == 3:  # TT, QQ, QU, UU
            qmap_copy = qmap.copy()

            qmap *= self.n_inv[0]
            qmap += self.n_inv[1] * umap

            umap *= self.n_inv[2]
            umap += self.n_inv[1] * qmap_copy

            del qmap_copy
        else:
            assert 0

        if len(self.templates_p) != 0:
            # FIXME :
            coeffs = np.concatenate(([t.dot(tmap) for t in self.templates_p]))
            coeffs = np.dot(self.Pt_Nn1_P_inv, coeffs)

            pmodes = np.zeros(len(self.n_inv[0]))
            im = 0
            for t in self.templates_p:
                t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv[0]
            tmap -= pmodes
