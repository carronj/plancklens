from __future__ import print_function

import os
import healpy as hp
import numpy as np
import pickle as pk

from plancklens.helpers import mpi, sql
from plancklens import utils


class library(object):
    r"""QE estimators (cross-)power spectra library template.

        This combines two QE estimators instances and a set of mean-field estimates to produce raw power spectra.

        Args:
            lib_dir: spectra will be cached in lib_dir
            qeA: first  QE instance (from *plancklens.qest*)
            qeB: second QE instance
            mc_sims_mf: simulation indices to use for the mean-field subtraction.
               *mc_sims_mf[0::2]* are used on the first spectrum leg, *mc_sims_mf[1::2]* on the second.

        This simple implementation produces spectra as :math:`\frac{1}{(2L + 1)f_{\rm sky}} \sum_{M} \hat \phi_{LM}^A \hat \phi_{LM}^{B\dagger}`
        after mean-field subtraction.


    """
    def __init__(self, lib_dir, qeA, qeB, mc_sims_mf):
        self.lib_dir = lib_dir
        self.prefix = lib_dir
        self.qeA = qeA
        self.qeB = qeB
        self.mc_sims_mf = mc_sims_mf
        fsname = os.path.join(lib_dir, 'fskies.dat')
        hname = os.path.join(self.lib_dir, 'qcl_sim_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir): os.makedirs(lib_dir)
            if not os.path.exists(fsname):
                print('Caching sky fractions...')
                ms = {1: self.qeA.get_mask(1), 2: self.qeA.get_mask(2),
                      3: self.qeB.get_mask(1), 4: self.qeB.get_mask(2)}
                assert np.all([m.shape == ms[1].shape for m in ms.values()]), (m.shape for m in ms.values())
                fskies = {}
                for i in [1, 2, 3, 4]:
                    for j in [1, 2, 3, 4][i - 1:]:
                        fskies[10 * i + j] = np.mean(ms[i] * ms[j])
                fskies[1234] = np.mean(ms[1] * ms[2] * ms[3] * ms[4])
                with open(fsname, 'w') as f:
                    for lab, _f in zip(np.sort(list(fskies.keys())), np.array(list(fskies.values()))[np.argsort(list(fskies.keys()))]):
                        f.write('%4s %.5f \n' % (lab, _f))
            if not os.path.exists(hname):
                if not os.path.exists(self.lib_dir):
                    os.makedirs(self.lib_dir)
                pk.dump(self.hashdict(), open(hname, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(hname, 'rb')), self.hashdict(), fn=hname)
        self.npdb = sql.npdb(os.path.join(lib_dir, 'cldb.db'))
        fskies = {}
        with open(fsname) as f:
            for line in f:
                (key, val) = line.split()
                fskies[int(key)] = float(val)
        self.fskies = fskies
        self.fsky1234 = fskies[1234]
        self.fsky11 = fskies[11]
        self.fsky12 = fskies[12]
        self.fsky22 = fskies[22]

    def hashdict(self):
        return {'qeA': self.qeA.hashdict(),
                'qeB': self.qeB.hashdict(),
                'mc_sims_mf': self._mcmf_hash()}

    def _mcmf_hash(self):
        return utils.mchash(self.mc_sims_mf)

    def get_lmaxqcl(self, k1, k2):
        return min(self.qeA.get_lmax_qlm(k1), self.qeB.get_lmax_qlm(k2))

    def load_sim_qcl(self, k1, idx, k2=None, lmax=None):
        """Same as get_sim_qcl without triggering its calculation"""
        return self.get_sim_qcl(k1, idx, k2=k2, lmax=lmax, calc=False)

    def get_sim_qcl(self, k1, idx, k2=None, lmax=None, recache=False, calc=True):
        """Returns QE (cross-)power spectrum for a simulation index.

            Args:
                k1: QE anisotropy key 1
                idx: simulation index
                k2: QE anisotropy key 2 (defaults to k1)
                lmax: optionally reduce the output to multipole lmax
                calc: calculates it if not done already. Otherwise throws an error if set to False

            Returns:
               QE power spectrum (1d array)

        """
        if k2 is None: k2 = k1
        assert k1 in self.qeA.keys and k2 in self.qeB.keys, (k1, k2)
        assert idx not in self.mc_sims_mf, idx
        lmax_qcl = self.get_lmaxqcl(k1, k2)
        lmax_out = lmax or lmax_qcl
        assert lmax_out <= lmax_qcl
        if idx >= 0:
            fname = os.path.join(self.lib_dir, 'sim_qcl_k1%s_k2%s_lmax%s_%04d_%s.dat' % (k1, k2, lmax_qcl, idx, self._mcmf_hash()))
        else:
            assert idx ==-1
            fname = os.path.join(self.lib_dir, 'sim_qcl_k1%s_k2%s_lmax%s_dat_%s.dat' % (k1, k2, lmax_qcl, self._mcmf_hash()))
        if calc:
            recache=False
        if calc and (self.npdb.get(fname) is None or recache):
            qlmA = self.qeA.get_sim_qlm(k1, idx, lmax=lmax_qcl)
            if (k1 == k2) and (self.qeA is self.qeB):
                qlmB = np.copy(qlmA)
            else:
                qlmB = self.qeB.get_sim_qlm(k2, idx, lmax=lmax_qcl)
            qlmA -= self.qeA.get_sim_qlm_mf(k1, self.mc_sims_mf[0::2], lmax=lmax_qcl)
            qlmB -= self.qeB.get_sim_qlm_mf(k2, self.mc_sims_mf[1::2], lmax=lmax_qcl)
            if recache and self.npdb.get(fname) is not None:
                self.npdb.remove(fname)
            self.npdb.add(fname, self._alm2clfsky1234(qlmA, qlmB, k1, k2))
            del qlmA, qlmB
        return self.npdb.get(fname)[:lmax_out + 1] / self.fskies[1234]

    def get_sim_stats_qcl(self, k1, mc_sims, k2=None, recache=False):
        """Returns the average of QE power spectra

            Args:
                k1: QE anisotropy key 1
                mc_sims: the simulation indices to average the spectra over
                k2: QE anisotropy key 2 (defaults to k1)

            Returns:
                *plancklens.utils.stats* instance

        """
        if k2 is None: k2 = k1
        tfname = os.path.join(self.lib_dir, 'sim_qcl_stats_%s_%s_%s.pk' % (k1, k2, utils.mchash(mc_sims)))
        if not os.path.exists(tfname) or recache:
            stats_qcl = utils.stats(self.get_lmaxqcl(k1, k2) + 1, docov=False)
            for i, idx in utils.enumerate_progress(mc_sims, label='sim_stats qcl (k1,k2)=' + str((k1, k2))):
                stats_qcl.add(self.get_sim_qcl(k1, idx, k2=k2))
            pk.dump(stats_qcl, open(tfname, 'wb'), protocol=2)
        return pk.load(open(tfname, 'rb'))

    def _alm2clfsky1234(self, qlm1, qlm2, k1, k2):
        return hp.alm2cl(qlm1, alms2=qlm2)


class average:
    """From a list of QE spectra libraries, produces a qecl average library.

        Args:
            lib_dir: spectra will be cached in lib_dir
            qcls_lib: list of *plancklens.qecl* instances

    """
    def __init__(self, lib_dir, qcls_lib):
        self.lib_dir = lib_dir
        self.qclibs = qcls_lib
        hname = os.path.join(lib_dir, 'qeclav_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(hname):
                pk.dump(self.hashdict(), open(hname, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(hname, 'rb')), self.hashdict(), fn=hname)
        self.mc_sims_mf = np.sort(np.unique(np.concatenate([qcl.mc_sims_mf for qcl in self.qclibs])))

    def hashdict(self):
        return {'qcl_lib %s'%i : qclib.hashdict() for i, qclib in enumerate(self.qclibs)}

    def get_lmaxqcl(self, k1, k2):
        return np.min([qclib.get_lmaxqcl(k1, k2) for qclib in self.qclibs])

    def get_sim_qcl(self, k1, idx, k2=None, lmax=None):
        """Returns QE (cross-)power spectrum average for a simulation index.

            Args:
                k1: QE anisotropy key 1
                idx: simulation index
                k2: QE anisotropy key 2 (defaults to k1)
                lmax: optionally reduce the output to multipole lmax

            Returns:
               QE power spectrum average (1d array)

        """
        if lmax is None: lmax = self.get_lmaxqcl(k1, k2)
        ret = self.qclibs[0].get_sim_qcl(k1, idx, k2=k2,lmax=lmax)
        for qclib in self.qclibs[1:]:
            ret += qclib.get_sim_qcl(k1, idx, k2=k2,lmax=lmax)
        return ret / len(self.qclibs)

    def get_dat_qcl(self, k1, k2=None, lmax=None):
        if lmax is None: lmax = self.get_lmaxqcl(k1, k2)
        ret = self.qclibs[0].get_dat_qcl(k1, k2=k2,lmax=lmax)
        for qclib in self.qclibs[1:]:
            ret += qclib.get_dat_qcl(k1, k2=k2,lmax=lmax)
        return ret / len(self.qclibs)

    def get_sim_stats_qcl(self, k1, mc_sims, k2=None, recache=False, lmax=None):
        """Returns the sim-average of the input *plancklens.qecl* list QE power spectra average

            Args:
                k1: QE anisotropy key 1
                mc_sims: the simulation indices to average the spectra over
                k2: QE anisotropy key 2 (defaults to k1)

            Returns:
                *plancklens.utils.stats* instance

        """
        if k2 is None: k2 = k1
        if lmax is None: lmax = self.get_lmaxqcl(k1, k2)
        tfname = os.path.join(self.lib_dir, 'sim_qcl_stats_%s_%s_%s_%s.pk' % (k1, k2, lmax, utils.mchash(mc_sims)))
        if not os.path.exists(tfname) or recache:
            stats_qcl = utils.stats(lmax + 1, docov=False)
            for i, idx in utils.enumerate_progress(mc_sims, label='building sim_stats qcl (k1,k2)=' + str((k1, k2))):
                stats_qcl.add(self.get_sim_qcl(k1, idx, k2=k2, lmax=lmax))
            pk.dump(stats_qcl, open(tfname, 'wb'), protocol=2)
        return pk.load(open(tfname, 'rb'))