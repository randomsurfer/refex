__author__ = 'pratik'

import nimfa
import numpy as np
import argparse
import scipy.optimize as opt
from sklearn.preprocessing import normalize
from collections import defaultdict
import mdl

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


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='eval sparsity nmf')
    argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    argument_parser.add_argument('-nm', '--node-measurement', help='node-measure matrix file', required=True)

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    node_measurement_file = args.node_measurement
    refex_features = np.loadtxt(node_feature, delimiter=',')
    node_measurements = np.loadtxt(node_measurement_file, delimiter=',')

    actual_fx_matrix = refex_features[:, 1:]
    node_id_seq = [int(node) for node in refex_features[:, 0]]

    aligned_node_measurements = []
    for node_id in node_id_seq:
        aligned_node_measurements.append(node_measurements[node_id, 1:])

    aligned_node_measurements = np.asarray(aligned_node_measurements)
    normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

    measurement_labels = ['Betweenness', 'Closeness', 'Degree', 'Clustering Coeff', '#BCC', 'Wt. Degree']

    G_ones = np.ones((aligned_node_measurements.shape[0], 1))
    E_ones = estimate_basis(normalized_measurements, G_ones)
    E_ones = np.asarray(E_ones)

    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    sparsity_threshold = [i for i in xrange(5, 20)]

    random_role_assignment = get_random_role_assignment(n, 10, 1000)
    E_ran = estimate_basis(normalized_measurements, normalize(random_role_assignment))
    E_ran = np.asarray(E_ran)

    random_sense_matrix = get_node_sense_matrix(E_ran, E_ones)

    mdlo = mdl.MDL(int(np.log2(n)))

    for sparsity in sparsity_threshold:
        snmf = nimfa.Snmf(actual_fx_matrix, seed="random_vcol", version='r', rank=10, beta=sparsity, max_iter=50)
        snmf_fit = snmf()
        node_roles = np.asarray(snmf_fit.basis())
        F = np.asarray(snmf_fit.coef())

        E = estimate_basis(normalized_measurements, normalize(node_roles))
        E = np.asarray(E)

        node_sense_matrix = get_node_sense_matrix(E, E_ones)

        aads = defaultdict(list)
        a = []

        err = snmf_fit.distance(metric='kl')

        # estimated_matrix = np.asarray(np.dot(node_roles, F))
        # loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        for j, label in enumerate(measurement_labels):
            label_measurement = node_sense_matrix[:, j]
            random_label_measurement = random_sense_matrix[:, j]
            aad = np.mean(np.abs(label_measurement - random_label_measurement))
            aads[label].append(aad)
            a.append(aad)

        final_str = ''
        for label in measurement_labels:
            val = '%.2f' % np.mean(aads[label])
            final_str += val + '\t'
        # print '\t'.join(measurement_labels)
        print final_str
        print ''
        print 'Spar: %s, err: %.2f, aad: %.2f' % (sparsity, err, np.mean(a))
        print ''
