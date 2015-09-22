__author__ = 'pratik'

import argparse
import os
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import brewer2mpl



def estimate_coeff(measurements, basis):
    coeff = np.zeros((measurements.shape[0], basis.shape[0]))
    for j in xrange(0, coeff.shape[0]):
        res = opt.nnls(basis.T, measurements[j, :])
        coeff[j, :] = res[0]
    return coeff


def estimate_basis(measurements, coeff):
    return estimate_coeff(measurements.T, coeff.T).T


def get_node_sense_matrix(E, E_ones):
    node_sense_matrix = []

    for r in xrange(E.shape[0]):
        m = []
        for s in xrange(E.shape[1]):
            m.append((E[r][s] / E_ones[0][s]))
        node_sense_matrix.append(m)

    return np.asarray(node_sense_matrix)


def get_primary_role(node_id, node_role, node_seq):
    row = None
    for c, n in enumerate(node_seq):
        if n == node_id:
            row = node_role[c, :]
            break
    if row is not None:
        reversed_sorted_indices = row.argsort()[-2:][::-1]
        primary_role = reversed_sorted_indices[0]
        return primary_role


def get_primary_role_for_all(node_role, node_seq):
    primary_node_role = {}
    for idx, node_id in enumerate(node_seq):
        row = node_role[idx, :]
        reversed_sorted_indices = row.argsort()[-2:][::-1]
        primary_role = reversed_sorted_indices[0]
        primary_node_role[node_id] = primary_role
    return primary_node_role


def load_name_mapping(_file):
    names = {}
    for line in open(_file):
        line = line.strip().split('\t')
        names[line[0]] = int(line[1])
    return names


if __name__ == '__main__':
    # from matplotlib import rcParams
    # rcParams['text.usetex'] = True
    argument_parser = argparse.ArgumentParser(prog='Author Visuals')
    argument_parser.add_argument('-n', '--author_name', help='author name', required=True)
    argument_parser.add_argument('-nw', '--network', help='author name', required=True)
    argument_parser.add_argument('-m', '--method', help='role discovery method', required=True)
    #
    args = argument_parser.parse_args()

    author_name = args.author_name
    network = args.network
    method = args.method

    method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
    method_fig_label = {'riders_r': 'EERs Right Sparse', 'riders': 'EERs', 'rolx': 'RolX', 'sparse': 'GLRD-S', 'diverse': 'GLRD-D'}
    nw_fig_label = {'_05_09': '2009', '_05_10': '2010', '_05_13': '2013'}

    final_values = np.zeros((6, 6))

    names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/CIKM%s_Graph_mapping.txt' % (network))
    names_icdm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/ICDM%s_Graph_mapping.txt' % (network))
    names_kdd = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/KDD%s_Graph_mapping.txt' % (network))
    names_sdm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/SDM%s_Graph_mapping.txt' % (network))
    names_sigmod = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/SIGMOD%s_Graph_mapping.txt' % (network))
    names_vldb = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/VLDB%s_Graph_mapping.txt' % (network))

    id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/%s%s/out-CIKM%s-ids.txt' % (method_node_ids[method], network, network))
    id_icdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/%s%s/out-ICDM%s-ids.txt' % (method_node_ids[method], network, network))
    id_kdd = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/%s%s/out-KDD%s-ids.txt' % (method_node_ids[method], network, network))
    id_sdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/%s%s/out-SDM%s-ids.txt' % (method_node_ids[method], network, network))
    id_sigmod = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/%s%s/out-SIGMOD%s-ids.txt' % (method_node_ids[method], network, network))
    id_vldb = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/%s%s/out-VLDB%s-ids.txt' % (method_node_ids[method], network, network))

    m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/properties%s/measurements.txt' % (network), delimiter=',')
    m_icdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/properties%s/measurements.txt' % (network), delimiter=',')
    m_kdd = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/properties%s/measurements.txt' % (network), delimiter=',')
    m_sdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/properties%s/measurements.txt' % (network), delimiter=',')
    m_sigmod = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/properties%s/measurements.txt' % (network), delimiter=',')
    m_vldb = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/properties%s/measurements.txt' % (network), delimiter=',')

    nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/%s%s/out-CIKM%s-nodeRoles.txt' % (method, network, network))
    nr_icdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/%s%s/out-ICDM%s-nodeRoles.txt' % (method, network, network))
    nr_kdd = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/%s%s/out-KDD%s-nodeRoles.txt' % (method, network, network))
    nr_sdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/%s%s/out-SDM%s-nodeRoles.txt' % (method, network, network))
    nr_sigmod = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/%s%s/out-SIGMOD%s-nodeRoles.txt' % (method, network, network))
    nr_vldb = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/%s%s/out-VLDB%s-nodeRoles.txt' % (method, network, network))


    all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                              'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                              'Degree', 'Wt_Degree', 'Clus_Coeff']

    labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

    measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt. Degree', 'Clustering Coeff']
    pruned_labels = dict((x, y) for x, y in zip(measurement_labels, range(len(measurement_labels))))

    cikm = []
    cikm_node_mapping = {}
    for i, node_id in enumerate(id_cikm):
        cikm_node_mapping[node_id] = i
        cikm.append(m_cikm[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    cikm = np.asarray(cikm)
    cikm_normalized = normalize(cikm, norm='l2', axis=0)

    E_cikm = estimate_basis(cikm_normalized, normalize(nr_cikm))
    G_ones_cikm = np.ones((cikm.shape[0], 1))
    E_ones_cikm = estimate_basis(cikm_normalized, G_ones_cikm)


    E_cikm = np.asarray(E_cikm)
    E_ones_cikm = np.asarray(E_ones_cikm)

    all_values = []
    for r in xrange(E_cikm.shape[0]):
        m = []
        for s in xrange(E_cikm.shape[1]):
            a = E_cikm[r][s] / E_ones_cikm[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_cikm = get_primary_role_for_all(nr_cikm, id_cikm)
    primary_role_idx = primary_roles_cikm[names_cikm[author_name]]

    final_values[0, :] = all_values[primary_role_idx, :]

    primary_role_between_cikm = []
    primary_role_closeness_cikm = []
    primary_role_bcc_cikm = []
    primary_role_degree_cikm = []
    primary_role_wt_degree_cikm = []
    primary_role_clus_coeff_cikm = []

    for nid, rid in primary_roles_cikm.iteritems():
        if primary_role_idx == rid:
            primary_role_between_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_cikm.append(cikm_normalized[cikm_node_mapping[nid], pruned_labels['Clustering Coeff']])


    ### ICDM
    icdm = []
    icdm_node_mapping = {}
    for i, node_id in enumerate(id_icdm):
        icdm_node_mapping[node_id] = i
        icdm.append(m_icdm[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    icdm = np.asarray(icdm)
    icdm_normalized = normalize(icdm, norm='l2', axis=0)

    E_icdm = estimate_basis(icdm_normalized, normalize(nr_icdm))
    G_ones_icdm = np.ones((icdm.shape[0], 1))
    E_ones_icdm = estimate_basis(icdm_normalized, G_ones_icdm)


    E_icdm = np.asarray(E_icdm)
    E_ones_icdm = np.asarray(E_ones_icdm)

    all_values = []
    for r in xrange(E_icdm.shape[0]):
        m = []
        for s in xrange(E_icdm.shape[1]):
            a = E_icdm[r][s] / E_ones_icdm[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_icdm = get_primary_role_for_all(nr_icdm, id_icdm)
    primary_role_idx = primary_roles_icdm[names_icdm[author_name]]

    final_values[1, :] = all_values[primary_role_idx, :]

    primary_role_between_icdm = []
    primary_role_closeness_icdm = []
    primary_role_bcc_icdm = []
    primary_role_degree_icdm = []
    primary_role_wt_degree_icdm = []
    primary_role_clus_coeff_icdm = []

    for nid, rid in primary_roles_icdm.iteritems():
        if primary_role_idx == rid:
            primary_role_between_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_icdm.append(icdm_normalized[icdm_node_mapping[nid], pruned_labels['Clustering Coeff']])


    ### KDD
    kdd = []
    kdd_node_mapping = {}
    for i, node_id in enumerate(id_kdd):
        kdd_node_mapping[node_id] = i
        kdd.append(m_kdd[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    kdd = np.asarray(kdd)
    kdd_normalized = normalize(kdd, norm='l2', axis=0)

    E_kdd = estimate_basis(kdd_normalized, normalize(nr_kdd))
    G_ones_kdd = np.ones((kdd.shape[0], 1))
    E_ones_kdd = estimate_basis(kdd_normalized, G_ones_kdd)


    E_kdd = np.asarray(E_kdd)
    E_ones_kdd = np.asarray(E_ones_kdd)

    all_values = []
    for r in xrange(E_kdd.shape[0]):
        m = []
        for s in xrange(E_kdd.shape[1]):
            a = E_kdd[r][s] / E_ones_kdd[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_kdd = get_primary_role_for_all(nr_kdd, id_kdd)
    primary_role_idx = primary_roles_kdd[names_kdd[author_name]]

    final_values[2, :] = all_values[primary_role_idx, :]

    primary_role_between_kdd = []
    primary_role_closeness_kdd = []
    primary_role_bcc_kdd = []
    primary_role_degree_kdd = []
    primary_role_wt_degree_kdd = []
    primary_role_clus_coeff_kdd = []

    for nid, rid in primary_roles_kdd.iteritems():
        if primary_role_idx == rid:
            primary_role_between_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_kdd.append(kdd_normalized[kdd_node_mapping[nid], pruned_labels['Clustering Coeff']])


    ### SDM
    sdm = []
    sdm_node_mapping = {}
    for i, node_id in enumerate(id_sdm):
        sdm_node_mapping[node_id] = i
        sdm.append(m_sdm[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    sdm = np.asarray(sdm)
    sdm_normalized = normalize(sdm, norm='l2', axis=0)

    E_sdm = estimate_basis(sdm_normalized, normalize(nr_sdm))
    G_ones_sdm = np.ones((sdm.shape[0], 1))
    E_ones_sdm = estimate_basis(sdm_normalized, G_ones_sdm)


    E_sdm = np.asarray(E_sdm)
    E_ones_sdm = np.asarray(E_ones_sdm)

    all_values = []
    for r in xrange(E_sdm.shape[0]):
        m = []
        for s in xrange(E_sdm.shape[1]):
            a = E_sdm[r][s] / E_ones_sdm[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_sdm = get_primary_role_for_all(nr_sdm, id_sdm)
    primary_role_idx = primary_roles_sdm[names_sdm[author_name]]

    final_values[3, :] = all_values[primary_role_idx, :]


    primary_role_between_sdm = []
    primary_role_closeness_sdm = []
    primary_role_bcc_sdm = []
    primary_role_degree_sdm = []
    primary_role_wt_degree_sdm = []
    primary_role_clus_coeff_sdm = []

    for nid, rid in primary_roles_sdm.iteritems():
        if primary_role_idx == rid:
            primary_role_between_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_sdm.append(sdm_normalized[sdm_node_mapping[nid], pruned_labels['Clustering Coeff']])


    ### SIGMOD
    sigmod = []
    sigmod_node_mapping = {}
    for i, node_id in enumerate(id_sigmod):
        sigmod_node_mapping[node_id] = i
        sigmod.append(m_sigmod[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    sigmod = np.asarray(sigmod)
    sigmod_normalized = normalize(sigmod, norm='l2', axis=0)

    E_sigmod = estimate_basis(sigmod_normalized, normalize(nr_sigmod))
    G_ones_sigmod = np.ones((sigmod.shape[0], 1))
    E_ones_sigmod = estimate_basis(sigmod_normalized, G_ones_sigmod)


    E_sigmod = np.asarray(E_sigmod)
    E_ones_sigmod = np.asarray(E_ones_sigmod)

    all_values = []
    for r in xrange(E_sigmod.shape[0]):
        m = []
        for s in xrange(E_sigmod.shape[1]):
            a = E_sigmod[r][s] / E_ones_sigmod[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_sigmod = get_primary_role_for_all(nr_sigmod, id_sigmod)
    primary_role_idx = primary_roles_sigmod[names_sigmod[author_name]]

    final_values[4, :] = all_values[primary_role_idx, :]


    primary_role_between_sigmod = []
    primary_role_closeness_sigmod = []
    primary_role_bcc_sigmod = []
    primary_role_degree_sigmod = []
    primary_role_wt_degree_sigmod = []
    primary_role_clus_coeff_sigmod = []

    for nid, rid in primary_roles_sigmod.iteritems():
        if primary_role_idx == rid:
            primary_role_between_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_sigmod.append(sigmod_normalized[sigmod_node_mapping[nid], pruned_labels['Clustering Coeff']])


    ### VLDB
    vldb = []
    vldb_node_mapping = {}
    for i, node_id in enumerate(id_vldb):
        vldb_node_mapping[node_id] = i
        vldb.append(m_vldb[node_id, [labels['Betweenness'], labels['Closeness'],
                                     labels['#BCC'], labels['Degree'],
                                     labels['Wt_Degree'], labels['Clus_Coeff']]])


    vldb = np.asarray(vldb)
    vldb_normalized = normalize(vldb, norm='l2', axis=0)

    E_vldb = estimate_basis(vldb_normalized, normalize(nr_vldb))
    G_ones_vldb = np.ones((vldb.shape[0], 1))
    E_ones_vldb = estimate_basis(vldb_normalized, G_ones_vldb)


    E_vldb = np.asarray(E_vldb)
    E_ones_vldb = np.asarray(E_ones_vldb)

    all_values = []
    for r in xrange(E_vldb.shape[0]):
        m = []
        for s in xrange(E_vldb.shape[1]):
            a = E_vldb[r][s] / E_ones_vldb[0][s]
            m.append(np.abs(a))
        all_values.append(m)

    all_values = np.asarray(all_values)
    all_values = normalize(all_values, norm='l2', axis=0)  # role X measurements
    primary_roles_vldb = get_primary_role_for_all(nr_vldb, id_vldb)
    primary_role_idx = primary_roles_vldb[names_vldb[author_name]]

    final_values[5, :] = all_values[primary_role_idx, :]

    primary_role_between_vldb = []
    primary_role_closeness_vldb = []
    primary_role_bcc_vldb = []
    primary_role_degree_vldb = []
    primary_role_wt_degree_vldb = []
    primary_role_clus_coeff_vldb = []

    for nid, rid in primary_roles_vldb.iteritems():
        if primary_role_idx == rid:
            primary_role_between_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['Betweenness']])
            primary_role_closeness_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['Closeness']])
            primary_role_bcc_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['#BCC']])
            primary_role_degree_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['Degree']])
            primary_role_wt_degree_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['Wt. Degree']])
            primary_role_clus_coeff_vldb.append(vldb_normalized[vldb_node_mapping[nid], pruned_labels['Clustering Coeff']])


    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    width = 0.12
    gap = 0.1
    ind = np.arange(6)

    between_values = final_values[:, 0]
    closeness_values = final_values[:, 1]
    bcc_values = final_values[:, 2]
    degree_values = final_values[:, 3]
    wdegree_values = final_values[:, 4]
    clustering_values = final_values[:, 5]

    fig = plt.figure()

    ax = fig.add_subplot(2,3,1)
    rects1 = ax.bar(gap+ind, between_values, width, color=colors[0], label=measurement_labels[0])
    rects2 = ax.bar(gap+ind+width, closeness_values, width, color=colors[1], label=measurement_labels[1])
    rects3 = ax.bar(gap+ind+2*width, bcc_values, width, color=colors[2], label=measurement_labels[2])
    rects4 = ax.bar(gap+ind+3*width, degree_values, width, color=colors[3], label=measurement_labels[3])
    rects5 = ax.bar(gap+ind+4*width, wdegree_values, width, color=colors[4], label=measurement_labels[4])
    rects6 = ax.bar(gap+ind+5*width, clustering_values, width, color=colors[5], label=measurement_labels[5])

    conf_labels = ['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB']
    ax.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax.set_xticklabels(conf_labels, rotation=20, ha='left')
    ax.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax.set_ylabel('Normalized Measurement Scores')  #, size=18)
    # ax.set_title(r'%s' % author_name)  #, size=22)
    fig.suptitle(r'%s, %s Roles on DBLP 2005-%s' % (author_name, method_fig_label[method], nw_fig_label[network]), size=18)  #, size=22)

    ax.legend(loc=1, ncol=3, fontsize='small')

    ax1 = fig.add_subplot(2, 3, 2)
    between_boxplot_data = [primary_role_between_cikm,primary_role_between_icdm, primary_role_between_kdd,
                            primary_role_between_sdm, primary_role_between_sigmod, primary_role_between_vldb]
    ax1.boxplot(between_boxplot_data,
                 notch=True, sym='kD', patch_artist=True,
                 boxprops={'facecolor': colors[0], 'edgecolor': colors[0]}, showmeans=True, meanline=True,
                 meanprops={'color': 'black', 'lw': 2.0},
                 medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})
    ax1.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax1.set_xticklabels(conf_labels, rotation=90, ha='left')
    ax1.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax1.set_ylabel('Normalized Betweenness Scores')  #, size=18)

    ax2 = fig.add_subplot(2, 3, 3)
    closeness_boxplot_data = [primary_role_closeness_cikm,primary_role_closeness_icdm, primary_role_closeness_kdd,
                       primary_role_closeness_sdm, primary_role_closeness_sigmod, primary_role_closeness_vldb]
    ax2.boxplot(closeness_boxplot_data,
                       notch=True, sym='kD', patch_artist=True,
                       boxprops={'facecolor': colors[1], 'edgecolor': colors[1]}, showmeans=True, meanline=True,
                       meanprops={'color': 'black', 'lw': 2.0},
                       medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})
    ax2.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax2.set_xticklabels(conf_labels, rotation=90, ha='left')
    ax2.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax2.set_ylabel('Normalized Closeness Scores')  #, size=18)

    ax3 = fig.add_subplot(2, 3, 4)
    bcc_boxplot_data = [primary_role_bcc_cikm,primary_role_bcc_icdm, primary_role_bcc_kdd,
                       primary_role_bcc_sdm, primary_role_bcc_sigmod, primary_role_bcc_vldb]
    ax3.boxplot(bcc_boxplot_data,
                       notch=True, sym='kD', patch_artist=True,
                       boxprops={'facecolor': colors[2], 'edgecolor': colors[2]}, showmeans=True, meanline=True,
                       meanprops={'color': 'black', 'lw': 2.0},
                       medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})
    ax3.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax3.set_xticklabels(conf_labels, rotation=90, ha='left')
    ax3.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax3.set_ylabel('Normalized #BCC Scores')  #, size=18)

    ax4 = fig.add_subplot(2, 3, 5)
    degree_boxplot_data = [primary_role_degree_cikm,primary_role_degree_icdm, primary_role_degree_kdd,
                       primary_role_degree_sdm, primary_role_degree_sigmod, primary_role_degree_vldb]
    ax4.boxplot(degree_boxplot_data,
                       notch=True, sym='kD', patch_artist=True,
                       boxprops={'facecolor': colors[3], 'edgecolor': colors[3]}, showmeans=True, meanline=True,
                       meanprops={'color': 'black', 'lw': 2.0},
                       medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})
    ax4.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax4.set_xticklabels(conf_labels, rotation=90, ha='left')
    ax4.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax4.set_ylabel('Normalized Degree Scores')  #, size=18)

    # ax2 = fig.add_subplot(3, 3, 5)
    # wt_degree_boxplot_data = [primary_role_wt_degree_cikm,primary_role_wt_degree_icdm, primary_role_wt_degree_kdd,
    #                    primary_role_wt_degree_sdm, primary_role_wt_degree_sigmod, primary_role_wt_degree_vldb]
    # ax2.boxplot(wt_degree_boxplot_data,
    #                    notch=True, sym='kD', patch_artist=True,
    #                    boxprops={'facecolor': colors[4], 'edgecolor': colors[4]}, showmeans=True, meanline=True,
    #                    meanprops={'color': 'black', 'lw': 2.0},
    #                    medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})

    ax6 = fig.add_subplot(2, 3, 6)
    clus_coeff_boxplot_data = [primary_role_clus_coeff_cikm,primary_role_clus_coeff_icdm, primary_role_clus_coeff_kdd,
                       primary_role_clus_coeff_sdm, primary_role_clus_coeff_sigmod, primary_role_clus_coeff_vldb]
    ax6.boxplot(clus_coeff_boxplot_data,
                       notch=True, sym='kD', patch_artist=True,
                       boxprops={'facecolor': colors[5], 'edgecolor': colors[5]}, showmeans=True, meanline=True,
                       meanprops={'color': 'black', 'lw': 2.0},
                       medianprops={'lw': 2.0, 'color': 'black'}, flierprops={'alpha': 0.7})
    ax6.set_xticks(ind, conf_labels)#, rotation='vertical')
    ax6.set_xticklabels(conf_labels, rotation=90, ha='left')
    ax6.tick_params(axis='x', length=50, which='major', direction='out', top='off')  #, labelsize=20)
    ax6.set_ylabel('Normalized Clustering Coeff. Scores')  #, size=18)

    plt.show()