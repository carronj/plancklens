from __future__ import print_function

import os
import healpy as hp
import numpy as np
import pickle as pk

from plancklens2018.helpers import mpi, sql
from plancklens2018 import utils


class library(object):
    """Lensing estimator power spectra library.

    """
    def __init__(self, lib_dir, qeA, qeB, mc_sims_mf):
        """ qeA and qeB are two quadratic estimator instances. """
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
                fskies[1234] = np.mean(ms[1] * ms[2] * ms[3] * ms[3])
                with open(fsname, 'w') as f:
                    for lab, _f in zip(np.sort(list(fskies.keys())), np.array(list(fskies.values()))[np.argsort(list(fskies.keys()))]):
                        f.write('%4s %.5f \n' % (lab, _f))
            if not os.path.exists(hname):
                if not os.path.exists(self.lib_dir):
                    os.makedirs(self.lib_dir)
                pk.dump(self.hashdict(), open(hname, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(hname, 'rb')), self.hashdict())
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

    def get_sim_qcl(self, k1, idx, k2=None, lmax=None, recache=False):
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
        if self.npdb.get(fname) is None or recache:
            qlmA = self.qeA.get_sim_qlm(k1, idx, lmax=lmax_qcl)
            qlmA -= self.qeA.get_sim_qlm_mf(k1, self.mc_sims_mf[0::2], lmax=lmax_qcl)
            qlmB = self.qeB.get_sim_qlm(k2, idx, lmax=lmax_qcl)
            qlmB -= self.qeB.get_sim_qlm_mf(k2, self.mc_sims_mf[1::2], lmax=lmax_qcl)
            if recache and self.npdb.get(fname) is not None:
                self.npdb.remove(fname)
            self.npdb.add(fname, self._alm2clfsky1234(qlmA, qlmB, k1, k2))
            del qlmA, qlmB
        return self.npdb.get(fname)[:lmax_out + 1] / self.fskies[1234]

    def get_sim_stats_qcl(self, k1, mc_sims, k2=None, recache=False):
        if k2 is None: k2 = k1
        tfname = os.path.join(self.lib_dir, 'sim_qcl_stats_%s_%s_%s.pk' % (k1, k2, utils.mchash(mc_sims)))
        if not os.path.exists(tfname) or recache:
            stats_qcl = utils.stats(self.get_lmaxqcl(k1, k2) + 1, docov=False)
            for i, idx in utils.enumerate_progress(mc_sims, label='sim_stats qcl (k1,k2)=' + str((k1, k2))):
                stats_qcl.add(self.get_sim_qcl(k1, idx, k2=k2))
            pk.dump(stats_qcl, open(tfname, 'wb'), protocol=2)
        return pk.load(open(tfname, 'rb'))

    def get_response(self, k1, s1, k2=None, s2=None):
        """
        k1 = estimator key for qeA.
        s1 = key defining source of anisotropy for the qeA estimator. defaults to k1.
        (optional) k2 = estimator key for qeB. defaults to k1.
        (optional) s2 = key defining source of anisotropy for the qeB estimator. defaults to s1.
        """
        if k2 is None: k2 = k1
        if s2 is None: s2 = s1
        return self.qeA.get_response(k1, s1) * self.qeB.get_response(k2, s2)

    def _alm2clfsky1234(self, qlm1, qlm2, k1, k2):
        return hp.alm2cl(qlm1, alms2=qlm2)


class library_MSC(library):
    """
    Exact same as above except spectra are mask deconv. spectra (on a single mask) instead of 1/fsky recipe.
    """

    def __init__(self, lib_dir, qeA, qeB, mc_sims_mf, lmax_cl, apomask_path, nhl_lib=None):
        # FIXME:
        from libaml import cut_sky_cl
        mask = hp.read_map(apomask_path)
        self.nside = hp.npix2nside(mask.size)
        self.MSC = cut_sky_cl.MaskedSkyCoupling(lmax_cl, mask)
        self.lmax_cl = lmax_cl
        super(library_MSC, self).__init__(lib_dir, qeA, qeB, mc_sims_mf, nhl_lib=nhl_lib)

    def _cli(self, cl):
        ret = np.zeros_like(cl)
        ret[np.where(cl != 0)] = 1. / cl[np.where(cl > 0)]
        return ret

    def hashdict(self):
        MSChash = self.MSC.cache_path(create=True)
        assert MSChash is not None
        return {'qeA': self.qeA.hashdict(), 'qeB': self.qeB.hashdict(),
                'apomask': MSChash, 'mc_sims_mf': self._mcmf_hash(), 'lmax_cl': self.lmax_cl}

    def _alm2clfsky1234(self, qlm1, qlm2, k1, k2):
        """ We first renormalized to something sensible and take take MSC spectra, and rescale back """
        assert not k1 is None and not k2 is None, (k1, k2)
        # FIXME hack
        f1 = k1[0] + '_teb' if k1[0] != 's' else 'stt'
        f2 = k2[0] + '_teb' if k2[0] != 's' else 'stt'
        norm1 = self._cli(self.qeA.get_response(k1, f1)[:self.lmax_cl + 1])
        norm2 = self._cli(self.qeA.get_response(k2, f2)[:self.lmax_cl + 1])
        if k1[0] in ['x', 'p']:
            norm1 *= np.arange(0, self.lmax_cl + 1, dtype=float) * np.arange(1, self.lmax_cl + 2, dtype=float)
        if k2[0] in ['x', 'p']:
            norm2 *= np.arange(0, self.lmax_cl + 1, dtype=float) * np.arange(1, self.lmax_cl + 2, dtype=float)
        if np.all(qlm2 == qlm1) and np.all(norm1 == norm2):
            q1 = hp.alm2map(hp.almxfl(qlm1, norm1), self.nside)
            return self.MSC.map2cl(q1, cache=False)[:self.lmax_cl + 1] * self._cli(norm1 * norm2) * self.fskies[1234]
        else:
            q1 = hp.alm2map(hp.almxfl(qlm1, norm1), self.nside)
            q2 = hp.alm2map(hp.almxfl(qlm2, norm2), self.nside)
            return self.MSC.map2cl(q1, _map2=q2, cache=False)[:self.lmax_cl + 1] * self._cli(norm1 * norm2) * \
                   self.fskies[1234]


class average:
    def __init__(self, lib_dir, qcls_lib):
        """Qcl average library. Hashes only for qecl sim stats

        """
        self.lib_dir = lib_dir
        self.qclibs = qcls_lib
        hname = os.path.join(lib_dir, 'qeclav_hash.pk')
        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
            if not os.path.exists(hname):
                pk.dump(self.hashdict(), open(hname, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(pk.load(open(hname, 'rb')), self.hashdict())

    def hashdict(self):
        return {'qcl_lib %s'%i : qclib.hashdict() for i, qclib in enumerate(self.qclibs)}

    def get_lmaxqcl(self, k1, k2):
        return np.min([qclib.get_lmaxqcl(k1, k2) for qclib in self.qclibs])

    def get_sim_qcl(self, k1, idx, k2=None, lmax=None):
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
        if k2 is None: k2 = k1
        if lmax is None: lmax = self.get_lmaxqcl(k1, k2)
        tfname = os.path.join(self.lib_dir, 'sim_qcl_stats_%s_%s_%s_%s.pk' % (k1, k2, lmax, utils.mchash(mc_sims)))
        if not os.path.exists(tfname) or recache:
            stats_qcl = utils.stats(lmax + 1, docov=False)
            for i, idx in utils.enumerate_progress(mc_sims, label='building sim_stats qcl (k1,k2)=' + str((k1, k2))):
                stats_qcl.add(self.get_sim_qcl(k1, idx, k2=k2, lmax=lmax))
            pk.dump(stats_qcl, open(tfname, 'wb'), protocol=2)
        return pk.load(open(tfname, 'rb'))