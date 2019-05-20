"""This module to calculate semi-analytical noise biases.

#FIXME: do version with non-zero empirical TB EB (-> complex coupling and nonzero GC-CG term)
"""
from __future__ import print_function

import os
import pickle as pk
import numpy as np
import healpy as hp

from plancklens2018 import utils
from plancklens2018 import mpi
from plancklens2018 import sql
from plancklens2018.qresp import get_qes, wignerc, get_coupling

class nhl_lib_simple:
    """Semi-analytical unnormalized N0 library.

    NB: This version only for 4 identical legs. Simple 1/fsky spectrum estimator.

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
                pk.dump(self.hashdict(), open(fn_hash, 'wb'))
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
        GC1 = '_C' if k1[0] == 'x' else '_G'
        GC2 = '_C' if k2[0] == 'x' else '_G'
        if GC1 != GC2:
            return np.zeros(self.lmax_qlm + 1, dtype=float)
        fn = 'anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + GC1
        suf =  ('sim%04d'%idx) * (idx >= 0) +  'dat' * (idx == -1)
        if self.npdb.get(fn + suf) is None or recache:
            cls_ivfs, lmax_ivf = self._get_cls(idx)
            G, C = get_nhl(k1, k2, self.cls_weight, cls_ivfs, lmax_ivf, lmax_out=self.lmax_qlm)
            if recache and self.npdb.get(fn) is not None:
                self.npdb.remove('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_G' + suf)
                self.npdb.remove('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_C' + suf)
            self.npdb.add('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_G' + suf, G)
            self.npdb.add('anhl_qe_' + k1[1:] + '_qe_' + k2[1:] + '_C' + suf, C)
        return self.npdb.get(fn + suf)

    def _get_cls(self, idx):
        #FIXME avoid computing unnecessary filtered maps.
        ret =  {'tt':  hp.alm2cl(self.ivfs.get_sim_tlm(idx)) / self.fsky,
                'ee':  hp.alm2cl(self.ivfs.get_sim_elm(idx)) / self.fsky,
                'bb':  hp.alm2cl(self.ivfs.get_sim_blm(idx)) / self.fsky,
                'te':  hp.alm2cl(self.ivfs.get_sim_tlm(idx), alms2=self.ivfs.get_sim_elm(idx)) / self.fsky}
        lmaxs = [len(cl) for cl in ret.values()]
        assert len(np.unique(lmaxs)) == 1, lmaxs
        return ret, lmaxs[0]


def get_nhl(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivfs, lmax_out=None, cls_ivfs_bb=None, cls_ivfs_ab=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

    #FIXME: explain cls_ivfs here

    """
    qes1 = get_qes(qe_key1, lmax_ivfs, cls_weights)
    qes2 = get_qes(qe_key2, lmax_ivfs, cls_weights)
    return  _get_nhl(qes1, qes2, cls_ivfs, lmax_ivfs,
                     lmax_out=lmax_out, cls_ivfs_bb=cls_ivfs_bb, cls_ivfs_ab=cls_ivfs_ab)

def get_nhl_cplx(qe_key1, qe_key2, cls_weights, cls_ivfs, lmax_ivfs, lmax_out=None, cls_ivfs_bb=None, cls_ivfs_ab=None):
    """(Semi-)Analytical noise level calculation for the cross-spectrum of two QE keys.

    #FIXME: explain cls_ivfs here

    """
    qes1 = get_qes(qe_key1, lmax_ivfs, cls_weights)
    qes2 = get_qes(qe_key2, lmax_ivfs, cls_weights)
    return  _get_nhl_cplx(qes1, qes2, cls_ivfs, lmax_ivfs,
                     lmax_out=lmax_out, cls_ivfs_bb=cls_ivfs_bb, cls_ivfs_ab=cls_ivfs_ab)

def _get_nhl(qes1, qes2, cls_ivfs, lmax_qe, lmax_out=None, cls_ivfs_bb=None, cls_ivfs_ab=None):

    lmax_out = 2 * lmax_qe if lmax_out is None else lmax_out
    G_N0 = np.zeros(lmax_out + 1, dtype=float)
    C_N0 = np.zeros(lmax_out + 1, dtype=float)
    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs_ab

    for qe1 in qes1:
        for qe2 in qes2:
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)
            sgn_R = (-1) ** (uo + vo + uo + vo)

            clsu = utils.joincls([qe1.leg_a.cl, qe2.leg_a.cl, get_coupling(si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([qe1.leg_b.cl, qe2.leg_b.cl, get_coupling(ti, vi, cls_ivfs_bb)])
            R_sutv = sgn_R * utils.joincls(
                [wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            clsv = utils.joincls([qe1.leg_a.cl, qe2.leg_b.cl, get_coupling(si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([qe1.leg_b.cl, qe2.leg_a.cl, get_coupling(ti, ui, cls_ivfs_ba)])
            R_sutv += sgn_R * utils.joincls(
                [wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = utils.joincls([sgnms * qe1.leg_a.cl, qe2.leg_a.cl, get_coupling(-si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([sgnmt * qe1.leg_b.cl, qe2.leg_b.cl, get_coupling(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = sgn_R * utils.joincls(
                [wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            clsv = utils.joincls([sgnms * qe1.leg_a.cl, qe2.leg_b.cl, get_coupling(-si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([sgnmt * qe1.leg_b.cl, qe2.leg_a.cl, get_coupling(-ti, ui, cls_ivfs_ba)])
            R_msmtuv += sgn_R * utils.joincls(
                [wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            G_N0 +=  0.5 * R_sutv
            G_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv

            C_N0 += 0.5 * R_sutv
            C_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv
    return G_N0, C_N0

def _get_nhl_cplx(qes1, qes2, cls_ivfs, lmax_qe, lmax_out=None, cls_ivfs_bb=None, cls_ivfs_ab=None):

    lmax_out = 2 * lmax_qe if lmax_out is None else lmax_out
    GG_N0 = np.zeros(lmax_out + 1, dtype=float)
    CC_N0 = np.zeros(lmax_out + 1, dtype=float)
    GC_N0 = np.zeros(lmax_out + 1, dtype=float)
    CG_N0 = np.zeros(lmax_out + 1, dtype=float)


    cls_ivfs_aa = cls_ivfs
    cls_ivfs_bb = cls_ivfs if cls_ivfs_bb is None else cls_ivfs_bb
    cls_ivfs_ab = cls_ivfs if cls_ivfs_ab is None else cls_ivfs_ab
    cls_ivfs_ba = cls_ivfs_ab

    for qe1 in qes1:
        for qe2 in qes2:
            si, ti, ui, vi = (qe1.leg_a.spin_in, qe1.leg_b.spin_in, qe2.leg_a.spin_in, qe2.leg_b.spin_in)
            so, to, uo, vo = (qe1.leg_a.spin_ou, qe1.leg_b.spin_ou, qe2.leg_a.spin_ou, qe2.leg_b.spin_ou)
            assert so + to >= 0 and uo + vo >= 0, (so, to, uo, vo)
            sgn_R = (-1) ** (uo + vo + uo + vo)

            clsu = utils.joincls([qe1.leg_a.cl, qe2.leg_a.cl, get_spin_coupling(si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([qe1.leg_b.cl, qe2.leg_b.cl, get_spin_coupling(ti, vi, cls_ivfs_bb)])
            R_sutv = sgn_R * utils.joincls(
                [wignerc(clsu, cltv, so, uo, to, vo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            clsv = utils.joincls([qe1.leg_a.cl, qe2.leg_b.cl, get_spin_coupling(si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([qe1.leg_b.cl, qe2.leg_a.cl, get_spin_coupling(ti, ui, cls_ivfs_ba)])
            R_sutv += sgn_R * utils.joincls(
                [wignerc(clsv, cltu, so, vo, to, uo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            # we now need -s-t uv
            sgnms = (-1) ** (si + so)
            sgnmt = (-1) ** (ti + to)
            clsu = utils.joincls([sgnms * qe1.leg_a.cl, qe2.leg_a.cl, get_spin_coupling(-si, ui, cls_ivfs_aa)])
            cltv = utils.joincls([sgnmt * qe1.leg_b.cl, qe2.leg_b.cl, get_spin_coupling(-ti, vi, cls_ivfs_bb)])
            R_msmtuv = sgn_R * utils.joincls(
                [wignerc(clsu, cltv, -so, uo, -to, vo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            clsv = utils.joincls([sgnms * qe1.leg_a.cl, qe2.leg_b.cl, get_spin_coupling(-si, vi, cls_ivfs_ab)])
            cltu = utils.joincls([sgnmt * qe1.leg_b.cl, qe2.leg_a.cl, get_spin_coupling(-ti, ui, cls_ivfs_ba)])
            R_msmtuv += sgn_R * utils.joincls(
                [wignerc(clsv, cltu, -so, vo, -to, uo, lmax_out=lmax_out), qe1.cL, qe2.cL])

            GG_N0 +=  0.5 * R_sutv.real
            GG_N0 +=  0.5 * (-1) ** (to + so) * R_msmtuv.real

            CC_N0 += 0.5 * R_sutv.real
            CC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.real

            GC_N0 -= 0.5 * R_sutv.imag
            GC_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

            CG_N0 += 0.5 * R_sutv.imag
            CG_N0 -= 0.5 * (-1) ** (to + so) * R_msmtuv.imag

    return GG_N0, CC_N0, GC_N0, CG_N0


def get_spin_coupling(s1, s2, cls):
    """<_{s1}X_{lm} _{s2}X^*{lm}>

    Note:
        This uses the spin-field conventions where _0X_{lm} = -T_{lm}

    """
    if s1 < 0:
        return (-1) ** (s1 + s2) * np.conjugate(get_coupling(-s1, -s2, cls))
    assert s1 in [0, -2, 2] and s2 in [0, -2, 2], (s1, s2, 'not implemented')
    if s1 == 0:
        if s2 == 0:
            return cls['tt']
        return - cls['te'] + 1j * np.sign(s2) * cls.get('tb', 0.)
    elif s1 == 2:
        if s2 == 0:
            return -cls['te'] - 1j * cls.get('tb', 0.)
        elif s2 == 2:
            return cls['ee'] + cls['bb']
        elif s2 == -2:
            return cls['ee'] - cls['bb'] + 2j * cls.get('eb', 0.)
        else:
            assert 0






