__author__ = 'pratik'

import sys
import numpy as np
import numpy.linalg as la
import networkx as nx
from collections import defaultdict

try:
    graph_t1 = sys.argv[1]
    graph_t2 = sys.argv[2]
    partition_t1 = sys.argv[3]
    partition_t2 = sys.argv[4]
    epsilon = int(sys.argv[5])
    output_file = sys.argv[6]
except IndexError:
    print 'usage: python %s <graph-at-t1> <graph-at-t2> ' \
          '<partition-at-t1> <partition-at-t2> <epsilon> <output-file>' % sys.argv[0]
    sys.exit(1)


# LSH signature generation using random projection
def get_signature(user_vector, rand_proj):
    res = 0
    for p in rand_proj:
        res <<= 1
        val = np.dot(p, user_vector)
        if val >= 0:
            res |= 1
    return res


# get number of '1's in binary
# running time: O(# of '1's)
def nnz(num):
    if num == 0:
        return 0
    res = 1
    num &= (num-1)
    while num:
        res += 1
        num &= (num-1)
    return res

def load_graph(file_name):
    graph = nx.Graph()
    for line in open(file_name):
        line = line.strip()
        line = line.split(',')
        source = int(line[0])
        dest = int(line[1])
        wt = float(line[2])
        graph.add_edge(source, dest, weight=wt)
    return graph


def load_partition(graph, file_name, epsilon=-1):
    partition = defaultdict(list)
    for i, line in enumerate(open(file_name)):
        line = line.strip().split()
        if len(line) > 1:  # discard singletons
            for node in line:
                node = int(node)
                adj_list = graph.neighbors(node)
                if len(adj_list) <= epsilon:  # discard EEP based trivial partition
                    break
                else:
                    partition[i].append(node)
    return partition


def get_block_id_for_node(partition):
    node_to_block_id = {}
    for block_id in partition.keys():
        nodes = partition[block_id]
        for node in nodes:
            node_to_block_id[node] = block_id
    return node_to_block_id

g1 = load_graph(graph_t1)
g2 = load_graph(graph_t2)
p1 = load_partition(g1, partition_t1, epsilon=epsilon)
p2 = load_partition(g2, partition_t2, epsilon=epsilon)
print 'Graphs and Partitions Loaded!'

node_pairs = defaultdict(list)
total_pairs = 0
for key in p1.keys():
    block = p1[key]
    block_size = len(block)
    if block_size > 1:
        for i in xrange(0, block_size - 1):
            for j in xrange(i + 1, block_size):
                node_pairs[block[i]].append(block[j])
                total_pairs += 1

print 'Total Pairs: ', total_pairs

node_to_block_id_for_p1 = get_block_id_for_node(p1)
node_to_block_id_for_p2 = get_block_id_for_node(p2)

comparison_block_order_for_p1 = []
comparison_block_order_for_p2 = []

for node in sorted(node_to_block_id_for_p1.keys()):
    block_id_for_node_in_p1 = node_to_block_id_for_p1[node]
    block_id_for_node_in_p2 = node_to_block_id_for_p2[node]

    if (not block_id_for_node_in_p1 in comparison_block_order_for_p1) and (not block_id_for_node_in_p2 in comparison_block_order_for_p2):
        comparison_block_order_for_p1.append(block_id_for_node_in_p1)
        comparison_block_order_for_p2.append(block_id_for_node_in_p2)

for block_id in sorted(p1.keys()):
    if block_id not in comparison_block_order_for_p1:
        comparison_block_order_for_p1.append(block_id)

for block_id in sorted(p2.keys()):
    if block_id not in comparison_block_order_for_p2:
        comparison_block_order_for_p2.append(block_id)

cosine_similarities = []
c = 0
# d = 2**10  # number of bits per signature
for a in node_pairs.keys():
    a_t = []
    a_tdt = []

    adj_a_t = set(g1.neighbors(a))
    adj_a_tdt = set(g2.neighbors(a))

    for block_id in comparison_block_order_for_p1:
        block = set(p1[block_id])
        a_t.append(float(len(adj_a_t & block)))

    for block_id in comparison_block_order_for_p2:
        block = set(p2[block_id])
        a_tdt.append(float(len(adj_a_tdt & block)))

    for b in node_pairs[a]:
        b_t = []
        b_tdt = []

        adj_b_t = set(g1.neighbors(b))
        adj_b_tdt = set(g2.neighbors(b))

        for block_id in comparison_block_order_for_p1:
            block = set(p1[block_id])
            b_t.append(float(len(adj_b_t & block)))

        for block_id in comparison_block_order_for_p2:
            block = set(p2[block_id])
            b_tdt.append(float(len(adj_b_tdt & block)))

        diff_t = np.asarray(a_t) - np.asarray(b_t)
        diff_tdt = np.asarray(a_tdt) - np.asarray(b_tdt)

        diff_t_size = diff_t.shape[0]
        diff_tdt_size = diff_tdt.shape[0]

        if diff_tdt_size >= diff_t_size:
            for i in xrange(diff_t_size, diff_tdt_size):
                diff_t = np.append(diff_t, 0.0)
        else:
            for i in xrange(diff_tdt_size, diff_t_size):
                diff_tdt = np.append(diff_tdt, 0.0)

        # randv = np.random.randn(d, diff_t.shape[0])
        # r1 = get_signature(diff_t, randv)
        # r2 = get_signature(diff_tdt, randv)
        # xor = r1 ^ r2
        # hash_sim = d - nnz(xor) / float(d)
        # cosine_similarities.append(hash_sim)
        tn = np.inner(diff_t, diff_tdt)
        td = la.norm(diff_t) * la.norm(diff_tdt)
        if td != 0.0:
            cosine_similarity = tn / td
            cosine_similarities.append(cosine_similarity)
        c += 1
        if c % 1000 == 0:
            print 'Completed %s Pairs' % c

cosine_similarities = np.asarray(cosine_similarities)
np.save(output_file, cosine_similarities)