__author__ = 'pratik'

import argparse
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import normalize as norm
import brewer2mpl
import matplotlib.pyplot as plt


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
    from matplotlib import rcParams
    rcParams['text.usetex'] = True

    argument_parser = argparse.ArgumentParser(prog='Top-k')
    # argument_parser.add_argument('-n', '--author_name', help='author name', required=True)
    argument_parser.add_argument('-nw', '--network', help='author name', required=True)
    # argument_parser.add_argument('-m', '--method', help='role discovery method', required=True)

    args = argument_parser.parse_args()

    # author_name = args.author_name
    author_names = ['Jiawei Han', 'Christos Faloutsos', 'Lise Getoor']
    network = args.network
    # method = args.method
    methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']
    legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                    'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

    method_line_style = ['-', '-', '--', '-.', '-.']
    colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
    method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

    method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

    bet_final_output = {}
    clus_final_output = {}
    ego_wt_final_output = {}

    fig = plt.figure()

    author_name = 'Lizhu Zhou'

    for method in methods:
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


        bet_final_output[method] = []
        clus_final_output[method] = []
        ego_wt_final_output[method] = []

        bet_vals = []
        clus_vals = []
        ego_wt_vals = []
        for i in xrange(0, 31):
            ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
            bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
            ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
            clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
            bet_final_output[method].append(np.mean(bet_vals))
            clus_final_output[method].append(np.mean(clus_vals))
            ego_wt_final_output[method].append(np.mean(ego_wt_vals))

    ax = fig.add_subplot(3, 3, 1)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(bet_final_output[method]), min_y)
        max_y = max(max(bet_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], bet_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('CIKM - %s, Betweenness' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    ax = fig.add_subplot(3, 3, 4)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(ego_wt_final_output[method]), min_y)
        max_y = max(max(ego_wt_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], ego_wt_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('CIKM - %s, Ego-1-Weight' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    ax = fig.add_subplot(3, 3, 7)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(clus_final_output[method]), min_y)
        max_y = max(max(clus_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], clus_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('CIKM - %s, Clustering Coeff.' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    # 9,3,2
    for method in methods:
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

        icdm_node_mapping = {}
        for i, node_id in enumerate(id_icdm):
            icdm_node_mapping[node_id] = i

        p, q = m_icdm.shape
        normalized_measurements_icdm = np.zeros((p, q))
        normalized_measurements_icdm[:, 0] = m_icdm[:, 0]
        normalized_measurements_icdm[:, 1:] = norm(m_icdm[:, 1:], norm='l2', axis=0)

        author_nr_vector = nr_icdm[icdm_node_mapping[names_icdm[author_name]], :]
        author_nr_idx = icdm_node_mapping[names_icdm[author_name]]

        cosine_values = []
        for idx, node_id in enumerate(id_icdm):
            if idx == author_nr_idx:
                continue
            sim = cosine_similarity(author_nr_vector, nr_icdm[idx, :])
            cosine_values.append((idx, sim))

        sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

        author_ranks = get_measurements(normalized_measurements_icdm, names_icdm[author_name], labels)


        bet_final_output[method] = []
        clus_final_output[method] = []
        ego_wt_final_output[method] = []

        bet_vals = []
        clus_vals = []
        ego_wt_vals = []
        for i in xrange(0, 31):
            ranks = get_measurements(normalized_measurements_icdm, sorted_cosine[i][0], labels)
            bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
            ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
            clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
            bet_final_output[method].append(np.mean(bet_vals))
            clus_final_output[method].append(np.mean(clus_vals))
            ego_wt_final_output[method].append(np.mean(ego_wt_vals))

    ax = fig.add_subplot(3, 3, 2)

    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(bet_final_output[method]), min_y)
        max_y = max(max(bet_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], bet_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')

    ax.set_title('ICDM - %s, Betweenness' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))
    ax.legend(loc='upper center', ncol=3, fontsize='small')

    ax = fig.add_subplot(3, 3, 5)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(ego_wt_final_output[method]), min_y)
        max_y = max(max(ego_wt_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], ego_wt_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('ICDM - %s, Ego-1-Weight' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    ax = fig.add_subplot(3, 3, 8)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(clus_final_output[method]), min_y)
        max_y = max(max(clus_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], clus_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('ICDM - %s, Clustering Coeff.' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))


    # 9,3,3
    for method in methods:
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

        sigmod_node_mapping = {}
        for i, node_id in enumerate(id_sigmod):
            sigmod_node_mapping[node_id] = i

        p, q = m_sigmod.shape
        normalized_measurements_sigmod = np.zeros((p, q))
        normalized_measurements_sigmod[:, 0] = m_sigmod[:, 0]
        normalized_measurements_sigmod[:, 1:] = norm(m_sigmod[:, 1:], norm='l2', axis=0)

        author_nr_vector = nr_sigmod[sigmod_node_mapping[names_sigmod[author_name]], :]
        author_nr_idx = sigmod_node_mapping[names_sigmod[author_name]]

        cosine_values = []
        for idx, node_id in enumerate(id_sigmod):
            if idx == author_nr_idx:
                continue
            sim = cosine_similarity(author_nr_vector, nr_sigmod[idx, :])
            cosine_values.append((idx, sim))

        sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

        author_ranks = get_measurements(normalized_measurements_sigmod, names_sigmod[author_name], labels)


        bet_final_output[method] = []
        clus_final_output[method] = []
        ego_wt_final_output[method] = []

        bet_vals = []
        clus_vals = []
        ego_wt_vals = []
        for i in xrange(0, 31):
            ranks = get_measurements(normalized_measurements_sigmod, sorted_cosine[i][0], labels)
            bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
            ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
            clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
            bet_final_output[method].append(np.mean(bet_vals))
            clus_final_output[method].append(np.mean(clus_vals))
            ego_wt_final_output[method].append(np.mean(ego_wt_vals))

    ax = fig.add_subplot(3, 3, 3)

    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(bet_final_output[method]), min_y)
        max_y = max(max(bet_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], bet_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')

    ax.set_title('SIGMOD - %s, Betweenness' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    ax = fig.add_subplot(3, 3, 6)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(ego_wt_final_output[method]), min_y)
        max_y = max(max(ego_wt_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], ego_wt_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('SIGMOD - %s, Ego-1-Weight' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    ax = fig.add_subplot(3, 3, 9)
    min_y = 0.0
    max_y = 0.0
    for j, method in enumerate(methods):
        min_y = min(min(clus_final_output[method]), min_y)
        max_y = max(max(clus_final_output[method]), max_y)
        ax.plot([i for i in xrange(0, 31)], clus_final_output[method], label=legend_names[method],
                linewidth=2.5, color=method_colors[j], linestyle='-')
    ax.set_title('SIGMOD - %s, Clustering Coeff.' % author_name)
    ax.set_xlabel('Top-k Set Size', fontsize='medium')
    ax.set_ylabel('Cumulative AAD', fontsize='medium')
    ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 2))

    plt.show()

    # Cumulative AAD, Top-k Node-Role Vectors Set Size
    # CIKM, KDD, SIGMOD
    # CIKM: BC, Ego_1_Wt, Clus_Coeff
    # KDD: BC, Ego_1_Wt, Clus_Coeff
    # SIGMOD: BC, Ego_1_Wt, Clus_Coeff

    # print author_name
    # for method in methods:
    #     print '%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (method,
    #                                                                 final_output[method][0],
    #                                                                 final_output[method][1],
    #                                                                 final_output[method][2],
    #                                                                 final_output[method][3],
    #                                                                 final_output[method][4],
    #                                                                 final_output[method][5],
    #                                                                 final_output[method][6],
    #                                                                 final_output[method][7])
        # final_output[method] = [rev_names_cikm[sorted_cosine[0][0]],
        #                         abs(author_ranks['Betweenness'] - ranks['Betweenness']),
        #                         abs(author_ranks['Closeness'] - ranks['Closeness']),
        #                         abs(author_ranks['#BCC'] - ranks['#BCC']),
        #                         abs(author_ranks['Degree'] - ranks['Degree']),
        #                         abs(author_ranks['Wt_Degree'] - ranks['Wt_Degree']),
        #                         abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']),
        #                         sorted_cosine[0][1]]
