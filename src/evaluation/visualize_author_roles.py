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


def get_primary_role(node_id, node_role):
    row = node_role[node_id, :]
    reversed_sorted_indices = row.argsort()[-2:][::-1]
    primary_role = reversed_sorted_indices[0]
    return primary_role


if __name__ == '__main__':
    # argument_parser = argparse.ArgumentParser(prog='Author Visuals')
    # argument_parser.add_argument('-id', '--author_id', help='author node id', required=True)
    # argument_parser.add_argument('-m', '--method', help='role discovery method', required=True)
    #
    # args = argument_parser.parse_args()
    #
    # a_id = int(args.author_id)
    # method = args.method

    # conf_ids = ['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB']
    # nw = '_05_10'
    #
    #
    # method_node_ids = {'riders_s': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
    final_values = np.zeros((6, 6))

    m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/properties_05_10/measurements.txt', delimiter=',')
    m_icdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/properties_05_10/measurements.txt', delimiter=',')
    m_kdd = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/properties_05_10/measurements.txt', delimiter=',')
    m_sdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/properties_05_10/measurements.txt', delimiter=',')
    m_sigmod = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/properties_05_10/measurements.txt', delimiter=',')
    m_vldb = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/properties_05_10/measurements.txt', delimiter=',')

    nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/riders_05_10/out-CIKM_05_10-nodeRoles.txt')
    nr_icdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/riders_05_10/out-ICDM_05_10-nodeRoles.txt')
    nr_kdd = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/riders_05_10/out-KDD_05_10-nodeRoles.txt')
    nr_sdm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/riders_05_10/out-SDM_05_10-nodeRoles.txt')
    nr_sigmod = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/riders_05_10/out-SIGMOD_05_10-nodeRoles.txt')
    nr_vldb = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/riders_05_10/out-VLDB_05_10-nodeRoles.txt')


    all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                              'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                              'Degree', 'Wt_Degree', 'Clus_Coeff']

    labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

    measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt. Degree', 'Clustering Coeff']

    cikm = []
    for node_id in xrange(m_cikm.shape[0]):
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
    primary_role_idx = get_primary_role(729, nr_cikm)

    final_values[0, :] = all_values[primary_role_idx, :]


    ### ICDM
    icdm = []
    for node_id in xrange(m_icdm.shape[0]):
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
    primary_role_idx = get_primary_role(271, nr_icdm)

    final_values[1, :] = all_values[primary_role_idx, :]

    ### KDD
    kdd = []
    for node_id in xrange(m_kdd.shape[0]):
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
    primary_role_idx = get_primary_role(13, nr_kdd)

    final_values[2, :] = all_values[primary_role_idx, :]

    ### SDM
    sdm = []
    for node_id in xrange(m_sdm.shape[0]):
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
    primary_role_idx = get_primary_role(25, nr_sdm)

    final_values[3, :] = all_values[primary_role_idx, :]

    ### SIGMOD
    sigmod = []
    for node_id in xrange(m_sigmod.shape[0]):
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
    primary_role_idx = get_primary_role(109, nr_sigmod)

    final_values[4, :] = all_values[primary_role_idx, :]

    ### VLDB
    vldb = []
    for node_id in xrange(m_vldb.shape[0]):
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
    primary_role_idx = get_primary_role(73, nr_vldb)

    final_values[5, :] = all_values[primary_role_idx, :]

    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    width = 0.12
    gap = 0.1
    ind = np.arange(6)

    fig, ax = plt.subplots()

    between_values = final_values[:, 0]
    closeness_values = final_values[:, 1]
    bcc_values = final_values[:, 2]
    degree_values = final_values[:, 3]
    wdegree_values = final_values[:, 4]
    clustering_values = final_values[:, 5]

    rects1 = ax.bar(gap+ind, between_values, width, color=colors[0], label=measurement_labels[0])
    rects2 = ax.bar(gap+ind+width, closeness_values, width, color=colors[1], label=measurement_labels[1])
    rects3 = ax.bar(gap+ind+2*width, bcc_values, width, color=colors[2], label=measurement_labels[2])
    rects4 = ax.bar(gap+ind+3*width, degree_values, width, color=colors[3], label=measurement_labels[3])
    rects5 = ax.bar(gap+ind+4*width, wdegree_values, width, color=colors[4], label=measurement_labels[4])
    rects6 = ax.bar(gap+ind+5*width, clustering_values, width, color=colors[5], label=measurement_labels[5])

    plt.xticks(ind, ['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB'], rotation='vertical')
    ax.set_xticks(ind)
    ax.set_xticklabels(['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB'], rotation=20, ha='left')
    ax.tick_params(axis='x', length=50, which='major', direction='out', top='off', labelsize=20)
    ax.set_ylabel('Normalized Role Measurement Scores', size=18)
    plt.title(r'Jiawei Han', size=22)
    #plt.title(r'CoR$\varepsilon$X-R Absolute Deviations from Baseline NodeSense on ICDM Co-Authorship Network, Year 2005-2009', size=16)

    plt.legend(loc=1, ncol=3)
    plt.show()
