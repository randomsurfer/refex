__author__ = 'pratik'

import argparse
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import normalize
import os
from collections import defaultdict


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

    methods = ['coridex', 'csparse', 'rsparse', 'riders', 'rolx', 'sparse', 'diverse']
    methods_id = {'coridex': 'coridex', 'csparse': 'coridex', 'rsparse': 'coridex',
                  'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
    global_measurement_labels = ['Betweenness', 'Closeness', '#BCC']
    neighborhood_measurement_labels = ['Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt']
    local_measurement_labels = ['Degree', 'Wt. Degree', 'Clustering Coeff']

    between_aad = np.zeros((7, 11))
    closeness_aad = np.zeros((7, 11))
    bcc_aad = np.zeros((7, 11))

    ego_zero_deg_aad = np.zeros((7, 11))
    ego_zero_wt_aad = np.zeros((7, 11))
    ego_one_deg_aad = np.zeros((7, 11))
    ego_one_wt_aad = np.zeros((7, 11))

    degree_aad = np.zeros((7, 11))
    wt_degree_aad = np.zeros((7, 11))
    clus_coeff_aad = np.zeros((7, 11))

    aad_global = np.zeros((7, 11))
    aad_neighbourhood = np.zeros((7, 11))
    aad_local = np.zeros((7, 11))

    for idx, method in enumerate(methods):
        aads_global_loop = defaultdict(list)
        aads_neighbours_loop = defaultdict(list)
        aads_local_loop = defaultdict(list)
        gbl = []
        ll = []
        nl = []
        for jdx, rank in enumerate(xrange(10, 21)):
            for itr in xrange(1, 6):
                fname = '%s-%s-%s-nodeRoles.txt' % (rank, itr, method)
                fname = os.path.join(input_folder, fname)

                fname_id = 'out-%s-ids.txt' % (methods_id[method])
                fname_id = os.path.join(input_folder, fname_id)

                node_roles = np.loadtxt(fname)
                node_roles[node_roles <= 0.0] = 0.0

                node_ids = np.loadtxt(fname_id)
                node_id_seq = [int(node) for node in node_ids]

                aligned_node_measurements = []
                for node_id in node_id_seq:
                    aligned_node_measurements.append(node_measurements[node_id, 1:])

                aligned_node_measurements = np.asarray(aligned_node_measurements)
                normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

                global_measurements = normalized_measurements[:, 0:3]
                neighborhood_measurements = normalized_measurements[:, 3:7]
                local_measurements = normalized_measurements[:, 7:]

                G_ones = np.ones((aligned_node_measurements.shape[0], 1))
                E_g_ones = np.asarray(estimate_basis(global_measurements, G_ones))
                E_n_ones = np.asarray(estimate_basis(neighborhood_measurements, G_ones))
                E_l_ones = np.asarray(estimate_basis(local_measurements, G_ones))

                normalized_node_roles = normalize(node_roles)

                E_g = np.asarray(estimate_basis(global_measurements, normalized_node_roles))
                E_n = np.asarray(estimate_basis(neighborhood_measurements, normalized_node_roles))
                E_l = np.asarray(estimate_basis(local_measurements, normalized_node_roles))

                node_sense_g = get_node_sense_matrix(E_g, E_g_ones)
                node_sense_n = get_node_sense_matrix(E_n, E_n_ones)
                node_sense_l = get_node_sense_matrix(E_l, E_l_ones)

                for i in xrange(30):
                    random_role_assignment = normalize(get_random_role_assignment(node_measurements.shape[0],
                                                                                  rank, 1000 + i))

                    E_ran_g = np.asarray(estimate_basis(global_measurements, random_role_assignment))
                    E_ran_n = np.asarray(estimate_basis(neighborhood_measurements, random_role_assignment))
                    E_ran_l = np.asarray(estimate_basis(local_measurements, random_role_assignment))

                    random_sense_g = get_node_sense_matrix(E_ran_g, E_g_ones)
                    random_sense_l = get_node_sense_matrix(E_ran_l, E_l_ones)
                    random_sense_n = get_node_sense_matrix(E_ran_n, E_n_ones)

                    for j, label in enumerate(global_measurement_labels):
                        # if idx == 1 and jdx >= 8:
                        #     break
                        label_measurement = node_sense_g[:, j]
                        random_label_measurement = random_sense_g[:, j]
                        aad = np.mean(np.abs(label_measurement - random_label_measurement))
                        aads_global_loop[label].append(aad)
                        gbl.append(aad)

                    for j, label in enumerate(neighborhood_measurement_labels):
                        label_measurement = node_sense_n[:, j]
                        random_label_measurement = random_sense_n[:, j]
                        aad = np.mean(np.abs(label_measurement - random_label_measurement))
                        aads_neighbours_loop[label].append(aad)
                        nl.append(aad)

                    for j, label in enumerate(local_measurement_labels):
                        label_measurement = node_sense_l[:, j]
                        random_label_measurement = random_sense_l[:, j]
                        aad = np.mean(np.abs(label_measurement - random_label_measurement))
                        aads_local_loop[label].append(aad)
                        ll.append(aad)

            aad_global[idx][jdx] = np.mean(gbl)
            print idx, jdx, np.mean(gbl)
            aad_neighbourhood[idx][jdx] = np.mean(nl)
            aad_local[idx][jdx] = np.mean(ll)
            between_aad[idx][jdx] = np.mean(aads_global_loop['Betweenness'])
            closeness_aad[idx][jdx] = np.mean(aads_global_loop['Closeness'])
            bcc_aad[idx][jdx] = np.mean(aads_global_loop['#BCC'])
            ego_zero_deg_aad[idx][jdx] = np.mean(aads_neighbours_loop['Ego_0_Deg'])
            ego_one_deg_aad[idx][jdx] = np.mean(aads_neighbours_loop['Ego_1_Deg'])
            ego_zero_wt_aad[idx][jdx] =  np.mean(aads_neighbours_loop['Ego_0_Wt'])
            ego_one_wt_aad[idx][jdx] =  np.mean(aads_neighbours_loop['Ego_1_Wt'])
            degree_aad[idx][jdx] = np.mean(aads_local_loop['Degree'])
            wt_degree_aad[idx][jdx] = np.mean(aads_local_loop['Wt. Degree'])
            clus_coeff_aad[idx][jdx] = np.mean(aads_local_loop['Clustering Coeff'])

    np.savetxt('aad_global.txt', aad_global)
    np.savetxt('aad_neighbourhood.txt', aad_neighbourhood)
    np.savetxt('aad_local.txt', aad_local)
    # np.savetxt('msd_between.txt', between_aad)
    # np.savetxt('msd_closeness.txt', closeness_aad)
    # np.savetxt('msd_bcc.txt', bcc_aad)
    # np.savetxt('msd_ego_0_deg.txt', ego_zero_deg_aad)
    # np.savetxt('msd_ego_1_deg.txt', ego_one_deg_aad)
    # np.savetxt('msd_ego_0_wt.txt', ego_zero_wt_aad)
    # np.savetxt('msd_ego_1_wt.txt', ego_one_wt_aad)
    # np.savetxt('msd_degree.txt', degree_aad)
    # np.savetxt('msd_wt_degree.txt', wt_degree_aad)
    # np.savetxt('msd_clus_coeff.txt', clus_coeff_aad)

    # from pandas import *
    # # import pylab
    # from matplotlib import pyplot as plt
    #
    # df = pandas.DataFrame(bcc_aad, columns=[str(i) for i in xrange(10, 21)])
    #
    # vals = np.around(df.values, 2)
    # normal = plt.normalize(vals.min() - 1, vals.max() + 1)
    #
    # fig = plt.figure(figsize=(15,10))
    # ax = fig.add_subplot(111, xticks=[], yticks=[])
    #
    # the_table=plt.table(cellText=vals, rowLabels=methods, colLabels=df.columns,
    #                     colWidths = [0.03]*vals.shape[1], loc='center',
    #                     cellColours=plt.cm.RdYlGn(normal(vals)))
    #
    # plt.show()
    # # pylab.savefig('colorscaletest.pdf', bbox_inches=0)