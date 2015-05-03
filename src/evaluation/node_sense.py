__author__ = 'pratik'

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import brewer2mpl
from sklearn.preprocessing import normalize


'''
V ~ GF
G is the basis matrix
F is the coeff matrix
---
Node Sense
G.E ~ M
E.T G.T ~ M.T
(m x r) (r x n) ~ m x n
M - node measurement matrix node x |measurements|
G - node-role matrix n x r
E - role-measurement matrix r x |measurements| (role contribution to node measurements)

def estimate_W(V, H):
    W = np.zeros((V.shape[0], H.shape[0]))
    print V.shape, H.shape
    for j in xrange(0, W.shape[0]):
        res = opt.nnls(H.T, V[j, :])
        W[j, :] = res[0]
    return W


'''


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
    value = 1.0 / num_nodes
    for node in xrange(num_nodes):
        role = random.randint(0, num_roles - 1)
        random_role_assignment[node][role] = value
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
    try:
        node_role_file = sys.argv[1]
        node_measurement_file = sys.argv[2]
        node_id_file = sys.argv[3]
    except IndexError:
        print 'usage: python %s <node-role-file> <node-measurements> <node-id-file>' % sys.argv[0]
        sys.exit(1)

    node_id_seq = []
    for line in open(node_id_file):
        line = line.strip()
        val = int(float(line))
        node_id_seq.append(val)

    try:
        node_roles = np.loadtxt(node_role_file)
        node_measurements = np.loadtxt(node_measurement_file, delimiter=',')
    except ValueError as ve:
        print ve.args

    # Node-Measurement matrix has node_id in the first column
    # and is sorted according to node ids too
    # Node-Role matrix is aligned with reference to the out-ids file
    # Hence, the alignment of the measurement matrix to the node ids in the node role matrix

    aligned_node_measurements = []
    for node_id in node_id_seq:
        aligned_node_measurements.append(node_measurements[node_id, 1:])

    aligned_node_measurements = np.asarray(aligned_node_measurements)
    normalized_measurements = normalize(aligned_node_measurements, norm='l1', axis=0)

    E = estimate_basis(normalized_measurements, normalize(node_roles, norm='l1', axis=0))
    G_ones = np.ones((aligned_node_measurements.shape[0], 1))
    E_ones = estimate_basis(normalized_measurements, G_ones)

    measurement_labels = ['Betweenness', 'Closeness', 'Degree', 'Clustering Coeff', '#BCC', 'Wt. Degree']
    # print '\t'.join(measurement_labels)
    E = np.asarray(E)
    E_ones = np.asarray(E_ones)

    all_values = []
    for r in xrange(E.shape[0]):
        m = []
        for s in xrange(E.shape[1]):
            m.append((E[r][s] / E_ones[0][s]))
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
    degree_values = all_values[:, 2]
    clustering_values = all_values[:, 3]
    bcc_values = all_values[:, 4]
    wdegree_values = all_values[:, 5]

    rects1 = ax.bar(gap+ind, between_values, width, color=colors[0], label=measurement_labels[0])
    rects2 = ax.bar(gap+ind+width, closeness_values, width, color=colors[1], label=measurement_labels[1])
    rects3 = ax.bar(gap+ind+2*width, degree_values, width, color=colors[2], label=measurement_labels[2])
    rects4 = ax.bar(gap+ind+3*width, clustering_values, width, color=colors[3], label=measurement_labels[3])
    rects5 = ax.bar(gap+ind+4*width, bcc_values, width, color=colors[4], label=measurement_labels[4])
    rects6 = ax.bar(gap+ind+5*width, wdegree_values, width, color=colors[5], label=measurement_labels[5])

    plt.xticks(ind, ['Role '+ str(i) for i in ind], rotation='vertical')
    ax.set_xticks(ind)
    ax.set_xticklabels(['Role '+ str(i) for i in ind], rotation=40, ha='left')
    ax.tick_params(axis='x', length=50, which='major', direction='out', top='off', labelsize=18)
    ax.set_ylabel('Scores', size=16)
    plt.title('GLRD-Diverse on CIKM Co-Authorship Network, Year 2005-2009', size=16)

    plt.legend(loc=1, ncol=3)
    plt.show()