import networkx as nx
import numpy as np
import argparse
from collections import defaultdict
from scipy import stats


def forest_fire_sampling(graph, sampling_fraction, geometric_dist_param=0.7):
    sampled_graph = nx.Graph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_nodes = graph.nodes()
    np.random.shuffle(shuffled_graph_nodes)
    already_visited = dict()

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        burn_seed_node = shuffled_graph_nodes[0]
        shuffled_graph_nodes = shuffled_graph_nodes[1:]

        if burn_seed_node in already_visited:
            continue

        already_visited[burn_seed_node] = 1

        num_edges_to_burn = np.random.geometric(p=geometric_dist_param)
        neighbors_to_burn = graph.neighbors(burn_seed_node)[:num_edges_to_burn]
        np.random.shuffle(neighbors_to_burn)
        burn_queue = []

        for n in neighbors_to_burn:
            if burn_seed_node != n:
                sampled_graph.add_edge(burn_seed_node, n)
                burn_queue.append(n)

        while len(burn_queue) > 0:
            burn_seed_node = burn_queue[0]
            burn_queue = burn_queue[1:]

            if burn_seed_node in already_visited:
                continue

            already_visited[burn_seed_node] = 1

            num_edges_to_burn = np.random.geometric(p=geometric_dist_param)

            neighbors_to_burn = graph.neighbors(burn_seed_node)[:num_edges_to_burn]
            np.random.shuffle(neighbors_to_burn)

            for n in neighbors_to_burn:
                if burn_seed_node != n:
                    sampled_graph.add_edge(burn_seed_node, n)
                    burn_queue.append(n)

    return sampled_graph


def induced_edge_sampling(graph, sampling_fraction):
    sampled_graph = nx.Graph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_edges = graph.edges()
    np.random.shuffle(shuffled_graph_edges)

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        u, v = shuffled_graph_edges[0]
        shuffled_graph_edges = shuffled_graph_edges[1:]

        if u != v:
            sampled_graph.add_edge(u, v)

    for u, v in graph.edges():
        if sampled_graph.has_node(u) and sampled_graph.has_node(v) and (not sampled_graph.has_edge(u, v)):
            if u != v:
                sampled_graph.add_edge(u, v)

    return sampled_graph


def reservoir_sampling(dataset, sample_size):
    sample_data = []
    for index, data in enumerate(dataset):
        if index < sample_size:
            sample_data.append(data)
        else:
            r = np.random.randint(0, index)
            if r < sample_size:
                sample_data[r] = data
    return sample_data


def get_node_role_assignment(node_role_matrix_with_ids):
    n, r = node_role_matrix_with_ids.shape
    role_assignments = {}

    for i in xrange(n):
        node_id = node_role_matrix_with_ids[i][0]
        row = node_role_matrix_with_ids[i, 1:]
        reversed_sorted_indices = row.argsort()[-2:][::-1]

        primary_role = reversed_sorted_indices[0]
        secondary_role = reversed_sorted_indices[1]

        if node_role_matrix_with_ids[i][primary_role] <= 0.0:
            primary_role = -1
        if node_role_matrix_with_ids[i][secondary_role] <= 0.0:
            secondary_role = -1

        role_assignments[node_id] = (primary_role, secondary_role)
    return role_assignments


def get_role_blocks(node_role_assignments):
    primary_role_blocks = defaultdict(list)
    secondary_role_blocks = defaultdict(list)
    for node in node_role_assignments.keys():
        primary_role_blocks[node_role_assignments[node][0]].append(node)
        secondary_role_blocks[node_role_assignments[node][1]].append(node)
    return primary_role_blocks, secondary_role_blocks


def role_based_sampling(graph, node_role_file, node_id_file, sampling_fraction):
    sampled_graph = nx.Graph()
    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    node_roles = np.loadtxt(node_role_file)
    node_ids = np.loadtxt(node_id_file)

    node_roles[node_roles <= 0.0] = 0.0
    num_nodes, num_roles = node_roles.shape

    node_roles_with_id = np.zeros((num_nodes, num_roles+1))
    node_roles_with_id[:, 0] = node_ids
    node_roles_with_id[:, 1:] = node_roles

    roles_assignments = get_node_role_assignment(node_roles_with_id)
    primary_blocks, secondary_blocks = get_role_blocks(roles_assignments)

    sampled_nodes = dict()
    for block_number in primary_blocks.keys():
        block = primary_blocks[block_number]

        sample_size_for_block = int(np.ceil(float(len(block)) * max_sampled_nodes / float(num_nodes)))

        block_sample = reservoir_sampling(block, sample_size_for_block)
        for n in block_sample:
            sampled_nodes[n] = 1

    for u, v in graph.edges():
        if u != v and u in sampled_nodes and v in sampled_nodes and not sampled_graph.has_edge(u, v):
            sampled_graph.add_edge(u, v)

    return sampled_graph


def load_graph(file_name):
    graph = nx.Graph()
    for line in open(file_name):
        line = line.strip().split(',')
        u = int(line[0])
        v = int(line[1])
        graph.add_edge(u, v)
    return graph


def degree_cdf(graph):
    nodes = graph.nodes()
    num_nodes = float(graph.number_of_nodes())

    dist = defaultdict(int)

    for n in nodes:
        num_neighbors = len(graph.neighbors(n))
        if num_neighbors > 0:
            dist[num_neighbors] += 1

    cdf = []
    for k in sorted(dist.keys()):
        cdf.append(float(dist[k]) / num_nodes)

    return np.asarray(cdf)


def clus_coeff_cdf(graph):
    nodes = graph.nodes()
    clus_c = nx.clustering(graph)

    dist = defaultdict(int)
    num_node_ge_one = 0
    for n in nodes:
        num_neighbors = len(graph.neighbors(n))
        if num_neighbors > 1:
            cc = np.round(clus_c[n], 2)
            dist[cc] += 1
            num_node_ge_one += 1

    cdf = []
    num_node_ge_one = float(num_node_ge_one)
    for k in sorted(dist.keys()):
        cdf.append(float(dist[k]) / num_node_ge_one)

    return np.asarray(cdf)


def hop_cdf(graph):
    nodes = graph.number_of_nodes()
    paths = nx.shortest_path_length(graph)

    dist = defaultdict(int)

    for u in paths.keys():
        for v in paths[u].keys():
            dist[paths[u][v]] += 1

    cdf = []
    for k in sorted(dist.keys()):
        cdf.append(float(dist[k]) / (nodes * nodes))

    return np.asarray(cdf)


def kcores_cdf(graph):
    max_core_graph = nx.k_core(graph)
    print max_core_graph
    return
    # max_core_node = max_core_graph.nodes()[0]
    # max_k = max_core_graph.succe


def eigenvalues(graph):
    try:
        import numpy.linalg as linal
        eigenvalues = linal.eigvals
    except ImportError:
        raise ImportError("numpy can not be imported.")

    L = nx.normalized_laplacian_matrix(graph)
    eigen_values = eigenvalues(L.A)

    return sorted(eigen_values, reverse=True)[:25]


def normalized_L1(p, q):
    size = len(p)
    return sum([float(np.abs(p[i] - q[i])) / float(p[i]) for i in xrange(size)]) / float(size)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(prog='network sampler')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    # argument_parser.add_argument('-nr', '--node-role', help='node role matrix file', required=False)
    # argument_parser.add_argument('-id', '--node-ids', help='node ids file', required=False)

    args = argument_parser.parse_args()

    graph_file = args.graph
    # node_role_file = args.node_role
    # node_id_file = args.node_ids

    graph = load_graph(graph_file)

    degree_cdf_graph = degree_cdf(graph)
    cc_cdf_graph = clus_coeff_cdf(graph)
    ev_graph = eigenvalues(graph)
    print 'Eigen Values Computed'

    deg_mean = np.zeros((8, 9))
    cc_mean = np.zeros((8, 9))
    ev_mean = np.zeros((8, 9)) # FF, ESi, Corex, Corex_R, Corex_S, RolX, GLRD-S, GLRD-D

    print 'Forest Fire'
    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = forest_fire_sampling(graph=graph, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[0][j] = np.mean(deg)
        cc_mean[0][j] = np.mean(cc)
        ev_mean[0][j] = np.mean(ev)

    print 'Induced Edges'
    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = induced_edge_sampling(graph=graph, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[1][j] = np.mean(deg)
        cc_mean[1][j] = np.mean(cc)
        ev_mean[1][j] = np.mean(ev)

    print 'COREX'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[2][j] = np.mean(deg)
        cc_mean[2][j] = np.mean(cc)
        ev_mean[2][j] = np.mean(ev)


    print 'COREX-R'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex_r/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[3][j] = np.mean(deg)
        cc_mean[3][j] = np.mean(cc)
        ev_mean[3][j] = np.mean(ev)


    print 'COREX-S'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex_s/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/corex/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[4][j] = np.mean(deg)
        cc_mean[4][j] = np.mean(cc)
        ev_mean[4][j] = np.mean(ev)


    print 'RolX'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/rolx/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/rolx/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[5][j] = np.mean(deg)
        cc_mean[5][j] = np.mean(cc)
        ev_mean[5][j] = np.mean(ev)

    print 'GLRD-S'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/glrd_s/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/rolx/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[6][j] = np.mean(deg)
        cc_mean[6][j] = np.mean(cc)
        ev_mean[6][j] = np.mean(ev)


    print 'GLRD-D'
    node_role_file = '/Users/pratik/Research/datasets/AutoSys/experiments/glrd_d/out-autosys-nodeRoles.txt'
    node_id_file = '/Users/pratik/Research/datasets/AutoSys/experiments/rolx/out-autosys-ids.txt'

    for j, fraction in enumerate(xrange(1, 10)):
        fraction = float(fraction) / 10.0
        print 'Fraction:', fraction
        deg = []
        cc = []
        ev = []

        for iter_no in xrange(5):
            ff_sampled_graph = role_based_sampling(graph=graph, node_role_file=node_role_file,
                                                   node_id_file=node_id_file, sampling_fraction=fraction)

            degree_cdf_ff = degree_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(degree_cdf_ff, degree_cdf_graph)
            deg.append(D)

            cc_cdf_ff = clus_coeff_cdf(ff_sampled_graph)
            D, p = stats.ks_2samp(cc_cdf_ff, cc_cdf_graph)
            cc.append(D)

            ev_ff = eigenvalues(ff_sampled_graph)
            l1 = normalized_L1(ev_graph, ev_ff)
            ev.append(l1)

        deg_mean[7][j] = np.mean(deg)
        cc_mean[7][j] = np.mean(cc)
        ev_mean[7][j] = np.mean(ev)

    np.savetxt('KS-Degree.txt', deg_mean)
    np.savetxt('KS-CC.txt', cc_mean)
    np.savetxt('KS-EV.txt', ev_mean)