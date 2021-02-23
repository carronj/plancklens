from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys, re, copy
import numpy as np

from plancklens.qcinv import util, util_alm
from plancklens.qcinv import cd_solve
from plancklens.qcinv import cd_monitors

class multigrid_stage(object):
    def __init__(self, ids, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache):
        self.depth = ids
        self.pre_ops_descr = pre_ops_descr
        self.lmax = lmax
        self.nside = nside
        self.iter_max = iter_max
        self.eps_min = eps_min
        self.tr = tr
        self.cache = cache
        self.pre_ops = []


class multigrid_chain:
    def __init__(self, opfilt, chain_descr, s_cls, n_inv_filt, debug_log_prefix=None, plogdepth=0):
        self.debug_log_prefix = debug_log_prefix
        self.plogdepth = plogdepth

        self.opfilt = opfilt
        self.chain_descr = chain_descr

        self.s_cls = s_cls
        self.n_inv_filt = n_inv_filt

        stages = {}
        for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache] in self.chain_descr:
            stages[id] = multigrid_stage(id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache)
            for pre_op_descr in pre_ops_descr:  # recursively add all stages to stages[0]
                stages[id].pre_ops.append(parse_pre_op_descr(pre_op_descr, opfilt=self.opfilt,
                                                             s_cls=self.s_cls, n_inv_filt=self.n_inv_filt,
                                                             stages=stages, lmax=lmax, nside=nside, chain=self))
        self.bstage = stages[0]  # these are the pre_ops called in cd_solve

    def solve(self, soltn, tpn_map, apply_fini='', dot_op=None):
        assert hasattr(self.opfilt, 'apply_fini%s' % apply_fini)
        finifunc = getattr(self.opfilt, 'apply_fini%s' % apply_fini)
        if apply_fini != '':
            print("*********** ")
            print("multigrid solve: using %s finiop " % apply_fini)
            print("*********** ")
        self.watch = util.stopwatch()
        self.iter_tot = 0
        self.prev_eps = None
        if dot_op is None:
            dot_op = self.opfilt.dot_op()
        logger = (lambda iter, eps, stage=self.bstage, **kwargs:
                  self.log(stage, iter, eps, **kwargs))

        tpn_alm = self.opfilt.calc_prep(tpn_map, self.s_cls, self.n_inv_filt)
        monitor = cd_monitors.monitor_basic(dot_op, logger=logger, iter_max=self.bstage.iter_max,
                                        eps_min=self.bstage.eps_min, d0=dot_op(tpn_alm, tpn_alm))

        fwd_op = self.opfilt.fwd_op(self.s_cls, self.n_inv_filt)

        cd_solve.cd_solve(soltn, tpn_alm,
                          fwd_op, self.bstage.pre_ops, dot_op, monitor,
                          tr=self.bstage.tr, cache=self.bstage.cache)
        finifunc(soltn, self.s_cls, self.n_inv_filt)

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage.depth > self.plogdepth:
            return

        log_str = '   ' * stage.depth + '(%4d, %04d) [%s] (%d, %.8f)' % (
        stage.nside, stage.lmax, str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

        if self.debug_log_prefix is not None:
            log = open(self.debug_log_prefix + 'stage_all.dat', 'a')
            log.write(log_str)
            log.close()

            if stage.depth == 0:
                f_handle = self.debug_log_prefix + 'stage_soltn_' + str(stage.depth) + '_%04d'%iter +'.npy'
                np.save(f_handle,  kwargs['soltn'])

                #f_handle = self.debug_log_prefix + 'stage_resid_' + str(stage.depth) + '.npy'
                #np.save(f_handle, kwargs['resid']]])
                #f_handle.close()

            log_str = '%05d %05d %10.6e %05d %s\n' % (self.iter_tot, int(elapsed), eps, iter, str(elapsed))
            log = open(self.debug_log_prefix + 'stage_' + str(stage.depth) + '.dat', 'a')
            log.write(log_str)
            log.close()

            if (self.prev_eps is not None) and (self.prev_stage.depth > stage.depth):
                log_final_str = '%05d %05d %10.6e %s\n' % (
                self.iter_tot - 1, int(self.prev_elapsed), self.prev_eps, str(self.prev_elapsed))

                log = open(self.debug_log_prefix + 'stage_final_' + str(self.prev_stage.depth) + '.dat', 'a')
                log.write(log_final_str)
                log.close()

            self.prev_stage = stage
            self.prev_eps = eps
            self.prev_elapsed = elapsed


def parse_pre_op_descr(pre_op_descr, **kwargs):
    if re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr):
        (low_descr, lsplit, hgh_descr) = re.match("split\((.*),\s*(.*),\s*(.*)\)\Z", pre_op_descr).groups()
        print('creating split preconditioner ', (low_descr, lsplit, hgh_descr))

        lsplit = int(lsplit)

        kwargs_low = copy.copy(kwargs);
        kwargs_low['lmax'] = lsplit
        kwargs_hgh = copy.copy(kwargs);
        kwargs_hgh['lmin'] = lsplit + 1
        pre_op_low = parse_pre_op_descr(low_descr, **kwargs_low)
        pre_op_hgh = parse_pre_op_descr(hgh_descr, **kwargs_hgh)

        return pre_op_split(lsplit, kwargs['lmax'], pre_op_low, pre_op_hgh)
    elif re.match("diag_cl\Z", pre_op_descr):
        return kwargs['opfilt'].pre_op_diag(kwargs['s_cls'], kwargs['n_inv_filt'])
    elif re.match("dense\Z", pre_op_descr):
        # FIXME: remove this option in favor of dense() below.
        print('creating dense preconditioner. (nside = %d, lmax = %d)' % (kwargs['nside'], kwargs['lmax']))

        fwd_op = kwargs['opfilt'].fwd_op(kwargs['s_cls'], kwargs['n_inv_filt'].degrade(kwargs['nside']))

        return kwargs['opfilt'].pre_op_dense(kwargs['lmax'], fwd_op)
    elif re.match("dense\((.*)\)\Z", pre_op_descr):
        (dense_cache_fname,) = re.match("dense\((.*)\)\Z", pre_op_descr).groups()
        if dense_cache_fname == '': dense_cache_fname = None

        print('creating dense preconditioner. (nside = %d, lmax = %d, cache = %s)' % (
        kwargs['nside'], kwargs['lmax'], dense_cache_fname))
        fwd_op = kwargs['opfilt'].fwd_op(kwargs['s_cls'], kwargs['n_inv_filt'].degrade(kwargs['nside']))
        return kwargs['opfilt'].pre_op_dense(kwargs['lmax'], fwd_op, cache_fname=dense_cache_fname)
    elif re.match("stage\(.*\)\Z", pre_op_descr):
        (stage_id,) = re.match("stage\((.*)\)\Z", pre_op_descr).groups()
        print('creating multigrid preconditioner: stage_id = ', stage_id)

        stage = kwargs['stages'][int(stage_id)]
        logger = (lambda iter, eps, stage=stage, chain=kwargs['chain'], **kwargs:
                  chain.log(stage, iter, eps, **kwargs))

        assert (stage.lmax == kwargs['lmax'])

        return pre_op_multigrid(kwargs['opfilt'], stage.lmax, stage.nside,
                                kwargs['s_cls'], kwargs['n_inv_filt'].degrade(stage.nside),
                                stage.pre_ops, logger, stage.tr, stage.cache,
                                stage.iter_max, stage.eps_min)
    else:
        assert 0, 'pre_op_descr' +  pre_op_descr +  ' is unrecognized!'


class pre_op_split:
    def __init__(self, lsplit, lmax, pre_op_low, pre_op_hgh):
        self.lsplit = lsplit
        self.lmax = lmax

        self.pre_op_low = pre_op_low
        self.pre_op_hgh = pre_op_hgh

        self.iter = 0

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        self.iter += 1

        talm_low = self.pre_op_low(util_alm.alm_copy(talm, lmax=self.lsplit))
        talm_hgh = self.pre_op_hgh(util_alm.alm_copy(talm, lmax=self.lmax))

        return util_alm.alm_splice(talm_low, talm_hgh, self.lsplit)


class pre_op_multigrid:
    def __init__(self, opfilt, lmax, nside, s_cls, n_inv_filt, pre_ops,
                 logger, tr, cache, iter_max, eps_min):
        self.opfilt = opfilt
        self.fwd_op = opfilt.fwd_op(s_cls, n_inv_filt)

        self.lmax = lmax
        self.nside = nside

        self.s_cls = s_cls
        self.pre_ops = pre_ops

        self.logger = logger

        self.tr = tr
        self.cache = cache

        self.iter_max = iter_max
        self.eps_min = eps_min

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, talm):
        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op(),
                            iter_max=self.iter_max, eps_min=self.eps_min, logger=self.logger)
        soltn = talm * 0.0
        cd_solve.cd_solve(soltn, util_alm.alm_copy(talm, lmax=self.lmax),
                          self.fwd_op, self.pre_ops, self.opfilt.dot_op(), monitor, tr=self.tr, cache=self.cache)

        return util_alm.alm_splice(soltn, talm, self.lmax)
