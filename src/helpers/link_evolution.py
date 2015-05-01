__author__ = 'pratik'

import sys
import math
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
        res = res << 1
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
    num = num & (num-1)
    while num:
        res += 1
        num = num & (num-1)
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


g1 = load_graph(graph_t1)
g2 = load_graph(graph_t2)
p1 = load_partition(g1, partition_t1, epsilon=epsilon)
p2 = load_partition(g2, partition_t2, epsilon=epsilon)
print 'Graphs and Partitions Loaded!'

node_pairs = []
for key in p1.keys():
    block = p1[key]
    block_size = len(block)
    if block_size > 1:
        for i in xrange(0, block_size - 1):
            for j in xrange(i + 1, block_size):
                node_pairs.append((block[i], block[j]))

print 'Total Pairs: ', len(node_pairs)
p1_keys = [(k, len(p1[k])) for k in p1.keys()]
sorted_p1_keys = [k for k, v in sorted(p1_keys, key=lambda x: x[1])]
p2_keys = [(k, len(p2[k])) for k in p2.keys()]
sorted_p2_keys = [k for k, v in sorted(p2_keys, key=lambda x: x[1])]

a_t = []
b_t = []
a_tdt = []
b_tdt = []
cosine_similarities = []
d = 2**6  # number of bits per signature

c = 0
for a, b in node_pairs:
    c += 1
    adj_a_t = set(g1.neighbors(a))
    adj_b_t = set(g1.neighbors(b))

    for key in sorted_p1_keys:
        block = set(p1[key])
        a_t.append(float(len(adj_a_t & block)))
        b_t.append(float(len(adj_b_t & block)))

    adj_a_tdt = set(g1.neighbors(a))
    adj_b_tdt = set(g1.neighbors(b))

    for key in sorted_p2_keys:
        block = set(p2[key])
        a_tdt.append(float(len(adj_a_tdt & block)))
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
    #
    # xor = r1 ^ r2
    # hash_sim = d - nnz(xor) / float(d)
    # cosine_similarities.append(hash_sim)

    tn = np.inner(diff_t, diff_tdt)
    td = la.norm(diff_t) * la.norm(diff_tdt)
    if td != 0.0:
        cosine_similarity = tn / td
        cosine_similarities.append(cosine_similarity)

    if c % 1000 == 0:
        print 'Completed %s Pairs' % c

cosine_similarities = np.asarray(cosine_similarities)
np.save(output_file, cosine_similarities)