from __future__ import absolute_import
from __future__ import print_function
# opfilt_tt
#
# operations and filters for temperature only c^-1
# S^{-1} (S^{-1} + Y^t N^{-1} Y)^{-1} Y^t N^{-1}
# TODO : could redefine units of ninv to avoid the npix / (4 pi everywhere)
import hashlib
import numpy  as np
import healpy as hp

from . import util
from . import util_alm
from . import template_removal
from . import dense


# ===

def calc_prep(map, s_cls, n_inv_filt):
    tmap = np.copy(map)
    n_inv_filt.apply_map(tmap)

    lmax = len(n_inv_filt.b_transf) - 1
    npix = len(map)

    alm = hp.map2alm(tmap, lmax=lmax, iter=0)
    alm *= npix / (4. * np.pi)

    hp.almxfl(alm, n_inv_filt.b_transf, inplace=True)
    return alm


def apply_fini(alm, s_cls, n_inv_filt):
    cltt = s_cls.cltt[:]
    cltt_inv = np.zeros(len(cltt))
    cltt_inv[np.where(cltt != 0)] = 1.0 / cltt[np.where(cltt != 0)]

    hp.almxfl(alm, cltt_inv, inplace=True)


# ===

class dot_op:
    def __init__(self, lmax=None):
        self.lmax = lmax

    def __call__(self, alm1, alm2):
        lmax1 = util_alm.nlm2lmax(len(alm1))
        lmax2 = util_alm.nlm2lmax(len(alm2))

        if self.lmax != None:
            lmax = self.lmax
        else:
            assert (lmax1 == lmax2)
            lmax = lmax1

        tcl = util_alm.alm_cl_cross(alm1, alm2)

        return np.sum(tcl * (2. * np.arange(0, lmax + 1) + 1))


class fwd_op:
    def __init__(self, s_cls, n_inv_filt):
        cltt = s_cls.cltt[:]
        self.cltt_inv = np.zeros(len(cltt))
        self.cltt_inv[np.where(cltt != 0)] = 1.0 / cltt[np.where(cltt != 0)]

        self.n_inv_filt = n_inv_filt

    def hashdict(self):
        return {'cltt_inv': hashlib.sha1(self.cltt_inv.view(np.uint8)).hexdigest(),
                'n_inv_filt': self.n_inv_filt.hashdict()}

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        if np.all(talm == 0):  # do nothing if zero
            return talm

        alm = np.copy(talm)
        self.n_inv_filt.apply_alm(alm)
        alm += hp.almxfl(talm, self.cltt_inv)

        return alm


# ===

class pre_op_diag:
    def __init__(self, s_cls, n_inv_filt):
        cltt = s_cls.cltt[:]

        assert len(cltt) >= len(n_inv_filt.b_transf)

        n_inv_cl = np.sum(n_inv_filt.n_inv) / (4.0 * np.pi)

        lmax = len(n_inv_filt.b_transf) - 1
        assert lmax <= (len(cltt) - 1)

        filt = np.zeros(lmax + 1)

        filt[np.where(cltt[0:lmax + 1] != 0)] += 1.0 / cltt[np.where(s_cls.cltt[0:lmax + 1] != 0)]
        filt[np.where(n_inv_filt.b_transf[0:lmax + 1] != 0)] += n_inv_cl * n_inv_filt.b_transf[np.where(
            n_inv_filt.b_transf[0:lmax + 1] != 0)] ** 2

        filt[np.where(filt != 0)] = 1.0 / filt[np.where(filt != 0)]

        self.filt = filt

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        return hp.almxfl(talm, self.filt)

# ===
def pre_op_dense(lmax, fwd_op, cache_fname=None):
    return dense.pre_op_dense_tt(lmax, fwd_op, cache_fname=cache_fname)

class alm_filter_ninv(object):
    def __init__(self, n_inv, b_transf, marge_monopole=False, marge_dipole=False, marge_uptolmin=-1, marge_maps=[]):
        """
            turn the input 'ninv' into a actual inverse variance maps.
            input 'n_inv' is a list of paths, or maps.
            The attribute n_inv is then the product of the masks map in the paths. 
            If several paths / maps are given they are all multiplied together.

            For planck 2015 n_inv is typically a list with [ n_invt, paths_to_masks]
            where n_invt is a scalar 3. / n_lev ** 2 (pixel variance)

            marge_monopole and marge_dipole ignored if marge_uptolmin is set.
        """
        if isinstance(n_inv, list):
            # The following line does nothing is if the first element of n_inv is not a string.
            n_inv_prod = util.load_map(n_inv[0][:])
            if (len(n_inv) > 1):
                for n in n_inv[1:]:
                    n_inv_prod = n_inv_prod * util.load_map(n[:])
            n_inv = n_inv_prod
            print("std = ", np.std(n_inv[np.where(n_inv[:] != 0.0)]) / np.average(n_inv[np.where(n_inv[:] != 0.0)]))
            # jcarron commented this :
            # assert (np.std(n_inv[np.where(n_inv[:] != 0.0)]) / np.average(n_inv[np.where(n_inv[:] != 0.0)]) < 1.e-7)
        else:
            n_inv = util.load_map(n_inv[:])
        templates = [];
        templates_hash = []
        for tmap in [util.load_map(m) for m in marge_maps]:
            assert (len(n_inv) == len(tmap))
            templates.append(template_removal.template_map(tmap))
            templates_hash.append(hashlib.sha1(tmap.view(np.uint8)).hexdigest())

        if marge_uptolmin >= 0:
            templates.append(template_removal.template_uptolmin(marge_uptolmin))
        else:
            if (marge_monopole): templates.append(template_removal.template_monopole())
            if (marge_dipole): templates.append(template_removal.template_dipole())

        if len(templates) != 0:
            nmodes = np.sum([t.nmodes for t in templates])
            modes_idx_t = np.concatenate(([t.nmodes * [int(im)] for im, t in enumerate(templates)]))
            modes_idx_i = np.concatenate(([range(0, t.nmodes) for t in templates]))
            print("   Building %s - %s template projection matrix" % (nmodes, nmodes))
            Pt_Nn1_P = np.zeros((nmodes, nmodes))
            for ir in range(0, nmodes):
                if np.mod(ir, int(0.1 * nmodes)) == 0: print ("   filling TNiT: %4.1f" % (100. * ir / nmodes)), "%"
                tmap = np.copy(n_inv)
                templates[modes_idx_t[ir]].apply_mode(tmap, int(modes_idx_i[ir]))

                ic = 0
                for tc in templates[0:modes_idx_t[ir] + 1]:
                    Pt_Nn1_P[ir, ic:(ic + tc.nmodes)] = tc.dot(tmap)
                    Pt_Nn1_P[ic:(ic + tc.nmodes), ir] = Pt_Nn1_P[ir, ic:(ic + tc.nmodes)]
                    ic += tc.nmodes
            eigv, eigw = np.linalg.eigh(Pt_Nn1_P)
            eigv_inv = 1.0 / eigv
            self.Pt_Nn1_P_inv = np.dot(np.dot(eigw, np.diag(eigv_inv)), np.transpose(eigw))

        self.n_inv = n_inv
        self.b_transf = b_transf[:]
        self.npix = len(self.n_inv)

        self.marge_monopole = marge_monopole
        self.marge_dipole = marge_dipole
        self.marge_uptolmin = marge_uptolmin
        self.templates = templates
        self.templates_hash = templates_hash

    def hashdict(self):
        hash = {'n_inv': hashlib.sha1(self.n_inv.view(np.uint8)).hexdigest(),
                'b_transf': hashlib.sha1(self.b_transf.view(np.uint8)).hexdigest(),
                'marge_monopole': self.marge_monopole,
                'marge_dipole': self.marge_dipole,
                'templates_hash': self.templates_hash}
        if self.marge_uptolmin > 0:  # for compability
            hash['marge_uptolmin'] = self.marge_uptolmin
        return hash

    def degrade(self, nside):
        if nside == hp.npix2nside(len(self.n_inv)):
            return self
        else:
            print("DEGRADING WITH NO MARGE MAPS")
            marge_maps = []
            # n_marge_maps = len(self.templates) - (self.marge_monopole + self.marge_dipole)
            # if ( n_marge_maps > 0 ):
            #    marge_maps = [hp.ud_grade(ti.map, nside) for ti in self.templates[0:n_marge_maps]]
            return alm_filter_ninv(hp.ud_grade(self.n_inv, nside, power=-2), self.b_transf, self.marge_monopole,
                                   self.marge_dipole, self.marge_uptolmin, marge_maps)

    def apply_alm(self, alm):
        # applies Y^T N^{-1} Y
        npix = len(self.n_inv)

        hp.almxfl(alm, self.b_transf, inplace=True)

        tmap = hp.alm2map(alm, hp.npix2nside(npix), verbose=False)

        self.apply_map(tmap)

        alm[:] = hp.map2alm(tmap, lmax=util_alm.nlm2lmax(len(alm)), iter=0)
        alm[:] *= (npix / (4. * np.pi))

        hp.almxfl(alm, self.b_transf, inplace=True)


    def apply_map(self, tmap):
        # applies N^{-1}

        tmap *= self.n_inv
        if len(self.templates) != 0:
            coeffs = np.concatenate(([t.dot(tmap) for t in self.templates]))
            coeffs = np.dot(self.Pt_Nn1_P_inv, coeffs)

            pmodes = np.zeros(len(self.n_inv))
            im = 0
            for t in self.templates:
                t.accum(pmodes, coeffs[im:(im + t.nmodes)])
                im += t.nmodes
            pmodes *= self.n_inv
            tmap -= pmodes