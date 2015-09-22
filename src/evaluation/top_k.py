__author__ = 'pratik'

import argparse
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import normalize as norm


def cosine_similarity(a, b):
    tn = np.inner(a, b)
    td = la.norm(a) * la.norm(b)
    if td != 0.0:
        return tn / td
    else:
        return 0.0


def get_measurements(measurements, node_id, labels):
    ranks = {}
    for label in labels.keys():
        nodes = measurements[:, 0]
        values = measurements[:, labels[label]]
        for node, value in zip(nodes, values):
            if node_id == int(node):
                ranks[label] = value
    return ranks


def load_name_mapping(_file):
    names = {}
    reverse_names = {}
    for line in open(_file):
        line = line.strip().split('\t')
        names[line[0]] = int(line[1])
        reverse_names[int(line[1])] = line[0].strip()
    return names, reverse_names


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(prog='Author Visuals')
    argument_parser.add_argument('-n', '--author_name', help='author name', required=True)
    argument_parser.add_argument('-nw', '--network', help='author name', required=True)
    argument_parser.add_argument('-m', '--method', help='role discovery method', required=True)

    args = argument_parser.parse_args()

    author_name = args.author_name
    network = args.network
    method = args.method

    method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

    names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/CIKM%s_Graph_mapping.txt' % (network))
    names_icdm, rev_names_icdm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/ICDM%s_Graph_mapping.txt' % (network))
    names_kdd, rev_names_kdd = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/KDD%s_Graph_mapping.txt' % (network))
    names_sdm, rev_names_sdm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/SDM%s_Graph_mapping.txt' % (network))
    names_sigmod, rev_names_sigmod = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/SIGMOD%s_Graph_mapping.txt' % (network))
    names_vldb, rev_names_vldb = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/VLDB%s_Graph_mapping.txt' % (network))

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

    measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
    pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

    cikm_node_mapping = {}
    for i, node_id in enumerate(id_cikm):
        cikm_node_mapping[node_id] = i

    # m_cikm = norm(m_cikm, norm='l2', axis=0)
    p, q = m_cikm.shape
    normalized_measurements_cikm = np.zeros((p, q))
    normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
    normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

    author_nr_vector = nr_cikm[cikm_node_mapping[names_cikm[author_name]], :]
    author_nr_idx = cikm_node_mapping[names_cikm[author_name]]

    cosine_values = []
    for idx, node_id in enumerate(id_cikm):
        if idx == author_nr_idx:
            continue
        sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
        cosine_values.append((idx, sim))

    sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

    author_ranks = get_measurements(normalized_measurements_cikm, names_cikm[author_name], labels)
    print author_name

    ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[0][0], labels)
    print '%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (rev_names_cikm[sorted_cosine[0][0]],
                                                            abs(author_ranks['Betweenness'] - ranks['Betweenness']),
                                                            abs(author_ranks['Closeness'] - ranks['Closeness']),
                                                            abs(author_ranks['#BCC'] - ranks['#BCC']),
                                                            abs(author_ranks['Degree'] - ranks['Degree']),
                                                            abs(author_ranks['Wt_Degree'] - ranks['Wt_Degree']),
                                                            abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']),
                                                            sorted_cosine[0][1])
    ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[1][0], labels)
    print '%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (rev_names_cikm[sorted_cosine[1][0]],
                                                            abs(author_ranks['Betweenness'] - ranks['Betweenness']),
                                                            abs(author_ranks['Closeness'] - ranks['Closeness']),
                                                            abs(author_ranks['#BCC'] - ranks['#BCC']),
                                                            abs(author_ranks['Degree'] - ranks['Degree']),
                                                            abs(author_ranks['Wt_Degree'] - ranks['Wt_Degree']),
                                                            abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']),
                                                            sorted_cosine[1][1])