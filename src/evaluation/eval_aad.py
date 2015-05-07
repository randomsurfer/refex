__author__ = 'pratik'

import nimfa
import numpy as np
import argparse
import scipy.optimize as opt
from sklearn.preprocessing import normalize
import os
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
    argument_parser = argparse.ArgumentParser(prog='eval aad')
    argument_parser.add_argument('-if', '--input-folder', help='input folder', required=True)
    argument_parser.add_argument('-nm', '--node-measurement', help='node-measure matrix file', required=True)

    args = argument_parser.parse_args()

    input_folder = args.input_folder
    node_measurement_file = args.node_measurement
    node_measurements = np.loadtxt(node_measurement_file, delimiter=',')

    methods = ['rolx', 'sparse', 'diverse', 'coriders', 'csparse', 'rsparse', 'riders']
    measurement_labels = ['Betweenness', 'Closeness', 'Degree', 'Clustering Coeff', '#BCC', 'Wt. Degree']

    between_aad = np.zeros((7, 11))
    degree_aad = np.zeros((7, 11))
    bcc_aad = np.zeros((7, 11))
    aad_all = np.zeros((7, 11))

    for jdx, rank in enumerate(xrange(10, 21)):
        for idx, method in enumerate(methods):
            aads = defaultdict(list)
            a = []
            fname = '%s-%s-nodeRoles.txt' % (rank, method)
            fname = os.path.join(input_folder, fname)
            fname_id = 'out-%s-ids.txt' % (method)
            fname_id = os.path.join(input_folder, fname_id)

            node_roles = np.loadtxt(fname)
            node_ids = np.loadtxt(fname_id)
            node_id_seq = [int(node) for node in node_ids]

            aligned_node_measurements = []
            for node_id in node_id_seq:
                aligned_node_measurements.append(node_measurements[node_id, 1:])

            aligned_node_measurements = np.asarray(aligned_node_measurements)
            normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

            G_ones = np.ones((aligned_node_measurements.shape[0], 1))
            E_ones = estimate_basis(normalized_measurements, G_ones)
            E_ones = np.asarray(E_ones)

            n, r = node_roles.shape

            E = estimate_basis(normalized_measurements, normalize(node_roles))
            E = np.asarray(E)

            node_sense_matrix = get_node_sense_matrix(E, E_ones)

            for i in xrange(30):
                random_role_assignment = get_random_role_assignment(node_measurements.shape[0], rank, 1000 + i)

                E_ran = estimate_basis(normalized_measurements, normalize(random_role_assignment))
                E_ran = np.asarray(E_ran)
                random_sense_matrix = get_node_sense_matrix(E_ran, E_ones)

                for j, label in enumerate(measurement_labels):
                    label_measurement = node_sense_matrix[:, j]
                    random_label_measurement = random_sense_matrix[:, j]
                    aad = np.mean(np.abs(label_measurement - random_label_measurement))
                    aads[label].append(aad)
                    a.append(aad)

            between_aad[idx][jdx] = np.mean(aads['Betweenness'])
            degree_aad[idx][jdx] = np.mean(aads['Degree'])
            bcc_aad[idx][jdx] = np.mean(aads['#BCC'])
            aad_all[idx][jdx] = np.mean(a)

    # plot here
    np.savetxt('aad', aad_all)
    np.savetxt('bcc', bcc_aad)
    np.savetxt('between', between_aad)
    np.savetxt('degree', degree_aad)

    from pandas import *
    # import pylab
    from matplotlib import pyplot as plt

    df = pandas.DataFrame(bcc_aad, columns=[str(i) for i in xrange(10, 21)])

    vals = np.around(df.values, 2)
    normal = plt.normalize(vals.min() - 1, vals.max() + 1)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, xticks=[], yticks=[])

    the_table=plt.table(cellText=vals, rowLabels=methods, colLabels=df.columns,
                        colWidths = [0.03]*vals.shape[1], loc='center',
                        cellColours=plt.cm.RdYlGn(normal(vals)))

    plt.show()
    # pylab.savefig('colorscaletest.pdf', bbox_inches=0)