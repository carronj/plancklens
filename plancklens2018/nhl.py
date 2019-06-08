"""This module to calculate semi-analytical noise biases.

"""
from __future__ import print_function

import os
import pickle as pk
import numpy as np
import healpy as hp

from plancklens2018 import utils
from plancklens2018 import utils_spin as uspin
from plancklens2018 import mpi
from plancklens2018 import sql
from plancklens2018 import qresp


def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivf1, lmax_ivf2,
            lmax_out=None, cls_ivfs_bb=None, cls_ivfs_ab=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

        Args:
            qe_key1: QE key 1
            qe_key2: QE key 2
            cls_weights: dictionary with the CMB spectra entering the QE weights.
                        (expected are 'tt', 'te', 'ee' when/if relevant)
            cls_ivfs: dictionary with the inverse-variance filtered CMB spectra.
                        (expected are 'tt', 'te', 'ee', 'bb', 'tb', 'eb' when/if relevant)
            lmax_ivf1: QE 1 uses CMB multipoles down to lmax_ivf1.
            lmax_ivf2: QE 2 uses CMB multipoles down to lmax_ivf2.
            lmax_out(optional): outputs are calculated down to lmax_out. Defaults to lmax_ivf1 + lmax_ivf2

        Outputs:
            4-tuple of gradient (G) and curl (C) mode Gaussian noise co-variances GG, CC, GC, CG.

    """
    qes1 = qresp.get_qes(qe_key1, lmax_ivf1, cls_weights)
    qes2 = qresp.get_qes(qe_key2, lmax_ivf2, cls_weights)
    if lmax_out is None:
        lmax_out = lmax_ivf1 + lmax_ivf2
    return  _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=cls_ivfs_bb, cls_ivfs_ab=cls_ivfs_ab)

def _get_nhl(qes1, qes2, cls_ivfs, lmax_out, cls_ivfs_bb=None, cls_ivfs_ab=None):
    GG_N0 = np.zeros(lmax_out + 1, dtype=float)
    CC_N0 = np.zeros(lmax_out + 1, dtype=float)
    GC_N0 = np.zeros(lmax_out + 1, dtype=float)
    CG_N0 = np.zeros(lmax_out + 1, dtype=float)

    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs_ab

    for qe1 in qes1:
        cL1 = qe1.cL(np.arange(lmax_out + 1))
        for qe2 in qes2:
            cL2 = qe2.cL(np.arange(lmax_out + 1))
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)
            sgn_R = (-1) ** (so + to + uo + vo)

            clsu = utils.joincls([qe1.leg_a.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([qe1.leg_b.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(ti, vi, cls_ivfs_bb)])
            R_sutv = sgn_R * utils.joincls(
                [uspin.wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = utils.joincls([qe1.leg_a.cl, qe2.leg_b.cl.conj(), uspin.spin_cls(si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([qe1.leg_b.cl, qe2.leg_a.cl.conj(), uspin.spin_cls(ti, ui, cls_ivfs_ba)])
            R_sutv = R_sutv + sgn_R * utils.joincls(
                [uspin.wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_out), cL1, cL2])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = utils.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = sgn_R * utils.joincls(
                [uspin.wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_out), cL1, cL2])

            clsv = utils.joincls([sgnms * qe1.leg_a.cl.conj(), qe2.leg_b.cl.conj(), uspin.spin_cls(-si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([sgnmt * qe1.leg_b.cl.conj(), qe2.leg_a.cl.conj(), uspin.spin_cls(-ti, ui, cls_ivfs_ba)])
            R_msmtuv = R_msmtuv + sgn_R * utils.joincls(
                [uspin.wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_out), cL1, cL2])

            GG_N0 +=  0.5 * R_sutv.real
            GG_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv.real

            CC_N0 += 0.5 * R_sutv.real
            CC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.real

            GC_N0 -= 0.5 * R_sutv.imag
            GC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

            CG_N0 += 0.5 * R_sutv.imag
            CG_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

    return GG_N0, CC_N0, GC_N0, CG_N0


class nhl_lib_simple:
    """Semi-analytical unnormalized N0 library.

    NB: This version only for 4 identical legs, and with simple 1/fsky spectrum estimator.

    """
    def __init__(self, lib_dir, ivfs, cls_weight, lmax_qlm):
        self.lmax_qlm = lmax_qlm
        self.cls_weight = cls_weight
        self.ivfs = ivfs
        fn_hash = os.path.join(lib_dir, 'nhl_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(fn_hash, 'rb')), self.hashdict())

        self.lib_dir = lib_dir
        self.npdb = sql.npdb(os.path.join(lib_dir, 'npdb.db'))
        self.fsky = np.mean(self.ivfs.get_fmask())

    def hashdict(self):
        ret = {k: utils.clhash(self.cls_weight[k]) for k in self.cls_weight.keys()}
        ret['ivfs']  = self.ivfs.hashdict()
        ret['lmax_qlm'] = self.lmax_qlm
        return ret

    def get_sim_nhl(self, idx, k1, k2, recache=False):
        assert idx == -1 or idx >= 0, idx
        s1, GC1, s1ins, ksp1 = qresp.qe_spin_data(k1)
        s2, GC2, s2ins, ksp2 = qresp.qe_spin_data(k2)
        fn = 'anhl_qe_' + ksp1 + k1[1:] + '_qe_' + ksp2 +  k2[1:] + GC1 + GC2
        suf =  ('sim%04d'%idx) * (int(idx) >= 0) +  'dat' * (idx == -1)
        if self.npdb.get(fn + suf) is None or recache:
            assert s1 >= 0 and s2 >= 0, (s1, s2)
            cls_ivfs, lmax_ivf = self._get_cls(idx, np.unique(np.concatenate([s1ins, s2ins])))
            GG, CC, GC, CG = get_nhl(k1, k2, self.cls_weight, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=self.lmax_qlm)
            fns = [('G', 'G', GG) ] + [('C', 'G', CG)] * (s1 > 0) + [('G', 'C', GC)] * (s2 > 0) + [('C', 'C', CC)] * (s1 > 0) * (s2 > 0)
            if recache and self.npdb.get(fn + suf) is not None:
                for GC1, GC2, N0 in fns:
                    self.npdb.remove('anhl_qe_' + ksp1 +  k1[1:] + '_qe_'+ ksp2 + k2[1:] + GC1 + GC2 + suf)
            for GC1, GC2, N0 in fns:
                self.npdb.add('anhl_qe_' + ksp1 + k1[1:] + '_qe_' + ksp2 + k2[1:] + GC1 + GC2 + suf, N0)
        return self.npdb.get(fn + suf)

    def _get_cls(self, idx, spins):
        assert np.all(spins >= 0), spins
        ret = {}
        if 0 in spins:
            ret['tt'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx)) / self.fsky
        if 2 in spins:
            ret['ee'] = hp.alm2cl(self.ivfs.get_sim_elm(idx)) / self.fsky
            ret['bb'] = hp.alm2cl(self.ivfs.get_sim_blm(idx)) / self.fsky
            ret['eb'] = hp.alm2cl(self.ivfs.get_sim_elm(idx), alms2=self.ivfs.get_sim_blm(idx)) / self.fsky
        if 0 in spins and 2 in spins:
            ret['te'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx), alms2=self.ivfs.get_sim_elm(idx)) / self.fsky
            ret['tb'] = hp.alm2cl(self.ivfs.get_sim_tlm(idx), alms2=self.ivfs.get_sim_blm(idx)) / self.fsky
        lmaxs = [len(cl) for cl in ret.values()]
        assert len(np.unique(lmaxs)) == 1, lmaxs
        return ret, lmaxs[0]