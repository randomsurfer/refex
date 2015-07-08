__author__ = 'pratik'

import sys
import argparse
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import brewer2mpl
from sklearn.preprocessing import normalize
from matplotlib import rcParams
from scipy.stats import powerlaw as pl

def estimate_coeff(measurements, basis):
    coeff = np.zeros((measurements.shape[0], basis.shape[0]))
    for j in xrange(0, coeff.shape[0]):
        res = opt.nnls(basis.T, measurements[j, :])
        coeff[j, :] = res[0]
    return coeff


def estimate_basis(measurements, coeff):
    return estimate_coeff(measurements.T, coeff.T).T


def get_random_role_assignment(num_nodes, num_roles, seed=1000):
    random_role_assignment = np.zeros((num_nodes, num_roles))
    import random
    random.seed(seed)
    value = 1.0
    for node in xrange(num_nodes):
        role = random.randint(0, num_roles - 1)
        random_role_assignment[node][role] = value
    return random_role_assignment


def get_powerlaw_random_role_assignment(num_nodes, num_roles, alpha=3.0, seed=1000):
    random_role_assignment = np.zeros((num_nodes, num_roles))
    np.random.seed(seed=seed)
    simulated_data = pl.rvs(alpha, size=num_nodes)
    hist, bins = np.histogram(simulated_data, bins=num_roles-1)
    default_value = 1.0
    test = []
    roles = np.digitize(simulated_data, bins)
    for node, role in zip(xrange(num_nodes), roles):
        test.append(role)
        random_role_assignment[node][role - 1] = default_value
    return random_role_assignment


def get_node_sense_matrix(E, E_ones):
    node_sense_matrix = []

    for r in xrange(E.shape[0]):
        m = []
        for s in xrange(E.shape[1]):
            m.append((E[r][s] / E_ones[0][s]))
        node_sense_matrix.append(m)

    return np.asarray(node_sense_matrix)


if __name__ == '__main__':
    # rcParams['text.usetex'] = True
    argument_parser = argparse.ArgumentParser(prog='node sense plots')
    argument_parser.add_argument('-nr', '--node-role', help='node-role matrix', required=True)
    argument_parser.add_argument('-m', '--measurements', help='measurements for graph', required=True)
    argument_parser.add_argument('-i', '--id-file', help='node-ids', required=True)

    args = argument_parser.parse_args()

    nr_file = args.node_role
    measurements_file = args.measurements
    id_file = args.id_file

    node_id_seq = np.loadtxt(id_file)
    node_roles = np.loadtxt(nr_file)[:, 0:5]
    # random_node_roles = get_random_role_assignment(node_id_seq.shape[0], 26, 101)[:, 0:5]
    random_node_roles = get_powerlaw_random_role_assignment(node_id_seq.shape[0], 5)#[:, 0:5]
    node_measurements = np.loadtxt(measurements_file, delimiter=',')

    # Node-Measurement matrix has node_id in the first column
    # and is sorted according to node ids too
    # Node-Role matrix is aligned with reference to the out-ids file
    # Hence, the alignment of the measurement matrix to the node ids in the node role matrix

    all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                              'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                              'Degree', 'Wt_Degree', 'Clus_Coeff']

    labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

    aligned_node_measurements = []
    for node_id in node_id_seq:
        aligned_node_measurements.append(node_measurements[node_id, [labels['Betweenness'], labels['Closeness'],
                                                                     labels['#BCC'], labels['Degree'],
                                                                     labels['Wt_Degree'], labels['Clus_Coeff']]])

    aligned_node_measurements = np.asarray(aligned_node_measurements)
    normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

    E = estimate_basis(normalized_measurements, normalize(node_roles))
    E_ran = estimate_basis(normalized_measurements, normalize(random_node_roles))
    G_ones = np.ones((aligned_node_measurements.shape[0], 1))
    E_ones = estimate_basis(normalized_measurements, G_ones)

    measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt. Degree', 'Clustering Coeff']

    E = np.asarray(E)
    E_ran = np.asarray(E_ran)
    E_ones = np.asarray(E_ones)

    all_values = []
    for r in xrange(E.shape[0]):
        m = []
        for s in xrange(E.shape[1]):
            a = E[r][s] / E_ones[0][s]
            b = E_ran[r][s] / E_ones[0][s]
            # m.append(np.abs(a))
            m.append(np.abs(b))
            # m.append(np.abs(a-b))
        all_values.append(m)

    all_values = np.asarray(all_values)
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors

    N = E.shape[0]
    width = 0.12
    gap = 0.1
    ind = np.arange(N)

    fig, ax = plt.subplots()

    between_values = all_values[:, 0]
    closeness_values = all_values[:, 1]
    bcc_values = all_values[:, 2]
    degree_values = all_values[:, 3]
    wdegree_values = all_values[:, 4]
    clustering_values = all_values[:, 5]

    rects1 = ax.bar(gap+ind, between_values, width, color=colors[0], label=measurement_labels[0])
    rects2 = ax.bar(gap+ind+width, closeness_values, width, color=colors[1], label=measurement_labels[1])
    rects3 = ax.bar(gap+ind+2*width, bcc_values, width, color=colors[2], label=measurement_labels[2])
    rects4 = ax.bar(gap+ind+3*width, degree_values, width, color=colors[3], label=measurement_labels[3])
    rects5 = ax.bar(gap+ind+4*width, wdegree_values, width, color=colors[4], label=measurement_labels[4])
    rects6 = ax.bar(gap+ind+5*width, clustering_values, width, color=colors[5], label=measurement_labels[5])

    plt.xticks(ind, ['Role '+ str(i) for i in ind], rotation='vertical')
    ax.set_xticks(ind)
    ax.set_xticklabels(['Role '+ str(i+1) for i in ind], rotation=20, ha='left')
    ax.tick_params(axis='x', length=50, which='major', direction='out', top='off', labelsize=24)
    ax.set_ylabel('NodeSense Scores', size=18)
    #plt.title(r'CoR$\varepsilon$X-R Absolute Deviations from Baseline NodeSense on ICDM Co-Authorship Network, Year 2005-2009', size=16)

    plt.legend(loc=1, ncol=3)
    plt.show()