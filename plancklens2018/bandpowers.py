import numpy as np


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


def get_rdn0(parfile, k1, k2, ksource):
    ds = parfile.qcls_ds.get_sim_stats_qcl(k1, parfile.mc_sims_var, k2=k2).mean()
    ss = parfile.qcls_ss.get_sim_stats_qcl(k1, parfile.mc_sims_var, k2=k2).mean()
    qc_norm = parfile.qresp_dd.get_response(k1, ksource) * parfile.qresp_dd.get_response(k2, ksource)
    return qc_norm * (4. * ds - 2. * ss)

def get_mcn0(parfile, k1, k2, ksource):
    ss = parfile.qcls_ss.get_sim_stats_qcl(k1, parfile.mc_sims_var, k2=k2).mean()
    qc_norm = parfile.qresp_dd.get_response(k1, ksource) * parfile.qresp_dd.get_response(k2, ksource)
    return qc_norm * (2. * ss)

def get_n1(parfile, k1, k2, ksource):
    """Analytical N1 caculation.

        This takes the analyical approximation to the QE pair filtering as input.

    """
    assert k1 == k2, 'check signs for qe''s of different spins'
    assert ksource[0] == 'p', 'check aniso source spectrum'
    # This implementation accepts 2 different qes but pairwise identical filtering on each qe leg.
    assert np.all(parfile.qcls_dd.qeA.f2map1.ivfs.get_ftl() == parfile.qcls_dd.qeA.f2map2.ivfs.get_ftl())
    assert np.all(parfile.qcls_dd.qeA.f2map1.ivfs.get_fel() == parfile.qcls_dd.qeA.f2map2.ivfs.get_fel())
    assert np.all(parfile.qcls_dd.qeA.f2map1.ivfs.get_fbl() == parfile.qcls_dd.qeA.f2map2.ivfs.get_fbl())
    assert np.all(parfile.qcls_dd.qeB.f2map1.ivfs.get_ftl() == parfile.qcls_dd.qeB.f2map2.ivfs.get_ftl())
    assert np.all(parfile.qcls_dd.qeB.f2map1.ivfs.get_fel() == parfile.qcls_dd.qeB.f2map2.ivfs.get_fel())
    assert np.all(parfile.qcls_dd.qeB.f2map1.ivfs.get_fbl() == parfile.qcls_dd.qeB.f2map2.ivfs.get_fbl())

    ivfsA = parfile.qcls_dd.qeA.f2map1.ivfs
    ivfsB = parfile.qcls_dd.qeB.f2map1.ivfs
    ftlA = ivfsA.get_ftl()
    felA = ivfsA.get_fel()
    fblA = ivfsA.get_fbl()
    ftlB = ivfsB.get_ftl()
    felB = ivfsB.get_fel()
    fblB = ivfsB.get_fbl()
    qc_norm = parfile.qresp_dd.get_response(k1, ksource) * parfile.qresp_dd.get_response(k2, ksource)
    n1pp = parfile.n1_dd.get_n1(k1, ksource, parfile.cl_unl.clpp, ftlA, felA, fblA, len(qc_norm) - 1
                                , kB=k2, ftlB=ftlB, felB=felB, fblB=fblB)
    return qc_norm * n1pp