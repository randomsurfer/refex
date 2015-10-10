import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import normalize as norm
import brewer2mpl
import matplotlib.pyplot as plt
import pickle


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


def reservoir_sampling(dataset, sample_size, seed):
    np.random.seed(seed)
    sample_data = []
    for index, data in enumerate(dataset):
        if index < sample_size:
            sample_data.append(data)
        else:
            r = np.random.randint(0, index)
            if r < sample_size:
                sample_data[r] = data
    return sample_data

from matplotlib import rcParams
rcParams['text.usetex'] = True

####
# CIKM
methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/CIKM%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/%s%s/out-CIKM%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/%s%s/out-CIKM%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('cikm_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('cikm_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('cikm_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

fig = plt.figure()
ax = fig.add_subplot(6, 3, 1)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('CIKM Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 2)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('CIKM Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 3)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('CIKM Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

##################
# ICDM

methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'ICDM SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/ICDM%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/%s%s/out-ICDM%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/ICDM/%s%s/out-ICDM%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('icdm_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('icdm_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('icdm_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

# fig = plt.figure()
ax = fig.add_subplot(6, 3, 4)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('ICDM Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 5)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('ICDM Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 6)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('ICDM Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

#### KDD
methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'KDD SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/KDD%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/%s%s/out-KDD%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/KDD/%s%s/out-KDD%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('kdd_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('kdd_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('kdd_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

# fig = plt.figure()
ax = fig.add_subplot(6, 3, 7)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('KDD Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 8)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('KDD Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 9)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('KDD Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))


####
# SDM
methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'SDM SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/SDM%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/%s%s/out-SDM%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SDM/%s%s/out-SDM%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('sdm_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('sdmm_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('sdmm_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

# fig = plt.figure()
ax = fig.add_subplot(6, 3, 10)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SDM Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 11)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SDM Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 12)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SDM Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

####
# SIGMOD

methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'SIGMOD SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/SIGMOD%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/%s%s/out-SIGMOD%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/SIGMOD/%s%s/out-SIGMOD%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('sigmod_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('sigmod_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('sigmod_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

# fig = plt.figure()
ax = fig.add_subplot(6, 3, 13)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SIGMOD Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 14)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SIGMOD Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 15)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('SIGMOD Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

###
# VLDB
methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}

bet_final_output = {}
clus_final_output = {}
ego_wt_final_output = {}

network = "_05_10"

for method in methods:
    bet_final_output[method] = {}
    clus_final_output[method] = {}
    ego_wt_final_output[method] = {}
    for i in xrange(0, 31):
        bet_final_output[method][i] = []
        clus_final_output[method][i] = []
        ego_wt_final_output[method][i] = []

bet_final = {}
clus_final = {}
ego_final = {}

for method in methods:
    bet_final[method] = {}
    clus_final[method] = {}
    ego_final[method] = {}

for seed in range(20):
    print 'VLDB SEED', seed
    for method in methods:
        print method
        names_cikm, rev_names_cikm = load_name_mapping('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/VLDB%s_Graph_mapping.txt' % (network))

        id_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/%s%s/out-VLDB%s-ids.txt' % (method_node_ids[method], network, network))

        m_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/properties%s/measurements.txt' % (network), delimiter=',')

        nr_cikm = np.loadtxt('/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/VLDB/%s%s/out-VLDB%s-nodeRoles.txt' % (method, network, network))

        all_measurement_labels = ['Betweenness', 'Closeness', '#BCC',
                                  'Ego_0_Deg', 'Ego_1_Deg', 'Ego_0_Wt', 'Ego_1_Wt',
                                  'Degree', 'Wt_Degree', 'Clus_Coeff']

        labels = dict((x, y + 1) for x, y in zip(all_measurement_labels, range(len(all_measurement_labels))))

        measurement_labels = ['Betweenness', 'Closeness', '#BCC', 'Degree', 'Wt_Degree', 'Clus_Coeff']
        pruned_labels = dict((x, y) for x, y in zip(all_measurement_labels, range(len(measurement_labels))))

        cikm_node_mapping = {}
        for i, id_file_id in enumerate(id_cikm):
            cikm_node_mapping[id_file_id] = i

        p, q = m_cikm.shape
        normalized_measurements_cikm = np.zeros((p, q))
        normalized_measurements_cikm[:, 0] = m_cikm[:, 0]
        normalized_measurements_cikm[:, 1:] = norm(m_cikm[:, 1:], norm='l2', axis=0)

        num_nodes, num_roles = nr_cikm.shape

        sampled_node_ids = reservoir_sampling(range(num_nodes), 50, seed=100+seed)
        print sampled_node_ids

        for j, node_id in enumerate(sampled_node_ids):
            print 'Processing Node: ', j
            author_nr_idx = cikm_node_mapping[node_id]  # treating author_nr_idx as id file id
            author_nr_vector = nr_cikm[author_nr_idx]
            author_ranks = get_measurements(normalized_measurements_cikm, node_id, labels)

            cosine_values = []
            for idx, id_file_id in enumerate(id_cikm):
                if idx == author_nr_idx:
                    continue
                sim = cosine_similarity(author_nr_vector, nr_cikm[idx, :])
                cosine_values.append((id_file_id, sim))

            sorted_cosine = sorted(cosine_values, key=lambda x: x[1], reverse=True)

            bet_vals = []
            clus_vals = []
            ego_wt_vals = []
            for i in xrange(0, 31):
                ranks = get_measurements(normalized_measurements_cikm, sorted_cosine[i][0], labels)
                bet_vals.append(abs(author_ranks['Betweenness'] - ranks['Betweenness']))
                ego_wt_vals.append(abs(author_ranks['Ego_1_Wt'] - ranks['Ego_1_Wt']))
                clus_vals.append(abs(author_ranks['Clus_Coeff'] - ranks['Clus_Coeff']))
                bet_final_output[method][i].append(np.mean(bet_vals))
                clus_final_output[method][i].append(np.mean(clus_vals))
                ego_wt_final_output[method][i].append(np.mean(ego_wt_vals))

        bet_final[method][seed] = []
        clus_final[method][seed] = []
        ego_final[method][seed] = []
        for i in xrange(0, 31):
            bet_final[method][seed].append(np.mean(bet_final_output[method][i]))
            clus_final[method][seed].append(np.mean(clus_final_output[method][i]))
            ego_final[method][seed].append(np.mean(ego_wt_final_output[method][i]))

bet = {}
clus = {}
ego = {}

for method in methods:
    bet[method] = []
    clus[method] = []
    ego[method] = []
    for seed in range(20):
        bet[method].append(bet_final[method][seed])
        clus[method].append(clus_final[method][seed])
        ego[method].append(ego_final[method][seed])

    bet[method] = np.asarray(bet[method])
    clus[method] = np.asarray(clus[method])
    ego[method] = np.asarray(ego[method])

    a, b = bet[method].shape
    for j in range(b):
        bet[method][0, j] = np.mean(bet[method][:, j])
        clus[method][0, j] = np.mean(clus[method][:, j])
        ego[method][0, j] = np.mean(ego[method][:, j])

    bet[method] = list(bet[method][0, :])
    clus[method] = list(clus[method][0, :])
    ego[method] = list(ego[method][0, :])

with open('vldb_bet.pickle', 'wb') as handle:
    pickle.dump(bet, handle)

with open('vldb_clus.pickle', 'wb') as handle:
    pickle.dump(clus, handle)

with open('vldb_ego.pickle', 'wb') as handle:
    pickle.dump(ego, handle)

method_line_style = ['-', '-', '--', '-.', '-.']
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
method_colors = [colors[0], colors[1], colors[2], 'black', colors[4]]

method_node_ids = {'riders_s': 'riders', 'riders_r': 'riders', 'riders': 'riders', 'rolx': 'rolx', 'sparse': 'rolx', 'diverse': 'rolx'}
legend_names = {'riders_r': r'RID$\varepsilon$Rs-R', 'riders': r'RID$\varepsilon$Rs',
                'rolx': r'RolX', 'sparse': r'GLRD-S', 'diverse': r'GLRD-D'}

# fig = plt.figure()
ax = fig.add_subplot(6, 3, 16)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(bet[method]), min_y)
    max_y = max(max(bet[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], bet[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('VLDB Betweenness')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

ax = fig.add_subplot(6, 3, 17)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(ego[method]), min_y)
    max_y = max(max(ego[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], ego[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('VLDB Ego-1-Wt')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))
# ax.legend(loc='lower right', ncol=3, fontsize='small')

ax = fig.add_subplot(6, 3, 18)

min_y = 0.0
max_y = 0.0
for j, method in enumerate(methods):
    min_y = min(min(clus[method]), min_y)
    max_y = max(max(clus[method]), max_y)
    ax.plot([i for i in xrange(0, 31)], clus[method], label=legend_names[method],
            linewidth=2.5, color=method_colors[j], linestyle='-')

ax.set_title('VLDB Clustering Coeff.')
ax.set_xlabel('Top-k Set Size', fontsize='medium')
ax.set_ylabel('Cumulative AAD', fontsize='medium')
ax.set_yticks(np.round(np.linspace(min_y, max_y, 4), 4))

# Shrink current axis's height by 10% on the bottom
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

plt.show()
plt.savefig('Topk_Sample.pdf')
