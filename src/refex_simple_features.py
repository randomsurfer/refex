import argparse
import networkx as nx
import numpy as np
from multiprocessing import Pool
from collections import defaultdict


graph = nx.Graph()
p = 0.5
s = 0
TOLERANCE = 0.0001
MAX_ITERATIONS = 5
refex_log_binned_buckets = []
vertex_egonets = {}
memo_recursive_fx_names = {}
iter_no = 0


def load_graph(file_name):
    for line in open(file_name):
        line = line.strip()
        line = line.split(',')
        source = int(line[0])
        dest = int(line[1])
        if source == 7657:
            print dest
        graph.add_edge(source, dest)


def get_egonet_members(vertex, level=0):
    lvl_zero_egonet = graph.neighbors(vertex)
    lvl_zero_egonet.append(vertex)
    if level == 1:
        lvl_one_egonet = []
        for node in lvl_zero_egonet:
            if node != vertex:
                lvl_one_egonet.extend(graph.neighbors(node))
        lvl_zero_egonet.extend(lvl_one_egonet)
    return list(set(lvl_zero_egonet))


def simple_base_egonet_primitive_features(vertex, level_id='0'):
    props = (vertex, {})
    props[1]['wn'+level_id] = 0.0
    props[1]['weu'+level_id] = 0.0
    props[1]['wet'+level_id] = 0.0
    props[1]['xedu'+level_id] = 0.0
    props[1]['xedt'+level_id] = 0.0

    egonet = get_egonet_members(vertex)

    for n1 in egonet:
        neighbours = graph.neighbors(n1)

        props[1]['wn'+level_id] += 1.0

        for n2 in neighbours:
            if n2 in egonet:
                props[1]['weu'+level_id] += 1.0
                props[1]['wet'+level_id] += len(graph.neighbors(n2))
            else:
                props[1]['xedu'+level_id] += 1.0
                props[1]['xedt'+level_id] += len(graph.neighbors(n2))
    return props


def init_log_binned_fx_buckets():
    no_of_vertices = graph.number_of_nodes()

    max_fx_value = np.ceil(np.log2(no_of_vertices) + TOLERANCE)  # fixing value of p = 0.5,
    log_binned_fx_keys = [value for value in xrange(0, int(max_fx_value))]

    fx_bucket_size = []
    starting_bucket_size = no_of_vertices

    for idx in np.arange(0.0, max_fx_value):
        starting_bucket_size *= p
        fx_bucket_size.append(int(np.ceil(starting_bucket_size)))

    total_slots_in_all_buckets = sum(fx_bucket_size)
    if total_slots_in_all_buckets > no_of_vertices:
        fx_bucket_size[0] -= (total_slots_in_all_buckets - no_of_vertices)

    log_binned_buckets_dict = dict(zip(log_binned_fx_keys, fx_bucket_size))

    for binned_value in sorted(log_binned_buckets_dict.keys()):
        for count in xrange(0, log_binned_buckets_dict[binned_value]):
            refex_log_binned_buckets.append(binned_value)

    if len(refex_log_binned_buckets) != no_of_vertices:
        raise Exception("Vertical binned bucket size not equal to the number of vertices!")


def get_sorted_feature_values(feature_values):
    sorted_fx_values = sorted(feature_values, key=lambda x: x[1])
    return sorted_fx_values, len(sorted_fx_values)


def vertical_bin(feature):
    vertical_binned_vertex = {}
    count_of_vertices_with_log_binned_fx_value_assigned = 0
    fx_value_of_last_vertex_assigned_to_bin = -1
    previous_binned_value = 0

    sorted_fx_values, sorted_fx_size = get_sorted_feature_values(feature)

    for vertex, value in sorted_fx_values:
        current_binned_value = refex_log_binned_buckets[count_of_vertices_with_log_binned_fx_value_assigned]

        # If there are ties, it may be necessary to include more than p|V| nodes
        if current_binned_value != previous_binned_value and value == fx_value_of_last_vertex_assigned_to_bin:
            vertical_binned_vertex[vertex] = previous_binned_value
        else:
            vertical_binned_vertex[vertex] = current_binned_value
            previous_binned_value = current_binned_value

        count_of_vertices_with_log_binned_fx_value_assigned += 1
        fx_value_of_last_vertex_assigned_to_bin = value

    return vertical_binned_vertex


def get_current_fx_names():
    return [attr for attr in sorted(graph.node[graph.nodes()[0]])]


def compute_log_binned_features(fx_list):
    graph_nodes = sorted(graph.nodes())
    for feature in fx_list:
        node_fx_values = []
        for n in graph_nodes:
            node_fx_values.append(tuple([n, graph.node[n][feature]]))

        vertical_binned_vertices = vertical_bin(node_fx_values)
        for vertex in vertical_binned_vertices.keys():
            binned_value = vertical_binned_vertices[vertex]
            graph.node[vertex][feature] = float(binned_value)


def simple_node_features(vertex):
    return simple_base_egonet_primitive_features(vertex, level_id='0')


def compute_recursive_egonet_features(vertex):
    sum_fx = '-s'
    mean_fx = '-m'
    vertex_lvl_0_egonet = vertex_egonets[vertex][0]
    vertex_lvl_0_egonet_size = float(len(vertex_lvl_0_egonet))

    fx_list = [fx_name for fx_name in sorted(graph.node[vertex].keys())
               if fx_name not in memo_recursive_fx_names]

    level_id = '0'
    for fx_name in fx_list:
        fx_value = 0.0
        for node in vertex_lvl_0_egonet:
            fx_value += graph.node[node][fx_name]

        s_fx_name = fx_name + '-' + str(iter_no) + sum_fx + level_id
        m_fx_name = fx_name + '-' + str(iter_no) + mean_fx + level_id

        graph.node[vertex][s_fx_name] = fx_value
        graph.node[vertex][m_fx_name] = float(fx_value) / vertex_lvl_0_egonet_size


def compute_recursive_features(fx_list):
    print 'Number of features: ', len(graph.node[0].keys())

    graph_nodes = graph.nodes()
    prev_fx_names = set(get_current_fx_names())

    pool = Pool(8)
    pool.map(compute_recursive_egonet_features, graph_nodes)

    new_fx_names = list(set(get_current_fx_names()) - prev_fx_names)

    # compute and replace the new feature values with their log binned values
    compute_log_binned_features(new_fx_names)


def save_feature_matrix(out_file_name):
    graph_nodes = sorted(graph.nodes())
    feature_names = list(sorted(graph.node[graph_nodes[0]].keys()))
    ff = open('feature-names-out.txt', 'w')
    for feature in feature_names:
        ff.write('%s,' % feature)
    ff.close()

    fo = open(out_file_name, 'w')
    for node in graph_nodes:
        fo.write('%s' % node)
        for feature in feature_names:
            fo.write(',%s' % graph.node[node][feature])
        fo.write('\n')


argument_parser = argparse.ArgumentParser(prog='run_recursive_feature_extraction')
argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
argument_parser.add_argument('-i', '--iterations', help='number of refex iterations', required=True)
argument_parser.add_argument('-s', '--max-diff', help='start value of feature similarity threshold (default=0)',
                             default=0, type=int)

args = argument_parser.parse_args()

graph_file = args.graph
max_diff = args.max_diff

MAX_ITERATIONS = int(args.iterations)

# load input graph
load_graph(graph_file)
print 'Graph Loaded'

print graph
# compute simple primitive features
graph_nodes = graph.nodes()

pool = Pool(8)
result = pool.map(simple_node_features, graph_nodes)

for n, fx in result:
    graph.node[n].update(fx)

fx_names = get_current_fx_names()

print 'Computed Simple Primitive Features'

init_log_binned_fx_buckets()

fx_names = get_current_fx_names()

compute_log_binned_features(fx_names)

print 'Computed Primitive Features:', len(fx_names)
import sys
sys.exit(0)



no_iterations = 0

while no_iterations <= MAX_ITERATIONS:
    # compute and prune recursive features for iteration #no_iterations
    current_fx = len(get_current_fx_names())


    abc = compute_recursive_egonet_features()
    new_fx = len(get_current_fx_names())

    print 'Prev Fx: %s, Current Fx: %s, Iter: %s' % (current_fx, new_fx, iter_no)
    iter_no += 1
    no_iterations += 1

save_feature_matrix("featureValues.csv")