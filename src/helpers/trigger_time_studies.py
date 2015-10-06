import os
import sys
import time
import networkx as nx
import numpy as np


def load_graph(file_name):
    graph = nx.Graph()
    for line in open(file_name):
        line = line.strip()
        line = line.split('\t')
        source = int(line[0])
        dest = int(line[1])
        graph.add_edge(source, dest)
    return graph


current_time = lambda: int(round(time.time() * 1000))


cores = '60'
gamma = '2-5'

for size in range(200000, 1000001, 200000):
    in_file = '%s.txt' % size
    out_file = 'time_%s_%s_%s.txt' % (cores, gamma, size)

    f_out = open(out_file, 'w')

    graph = load_graph(in_file)

    d = 0.0
    for n in graph.nodes():
        d += float(len(graph.neighbors(n)))
    V = graph.number_of_nodes()

    avg_degree = int(np.ceil(d / V))

    fo = open('graph.txt', 'w')

    adj_lists = graph.adjacency_list()

    for adj_list in adj_lists:
        adj = [str(a) for a in adj_list]
        final_str = ' '.join(adj)
        fo.write(final_str + '\n')
    fo.close()

    start = current_time()
    f_out.write('EEP_Start\t%s\n' % start)
    out_path = 'partition'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        os.system('rm -f partition/*')

    for epsilon in range(1, avg_degree+1):
        os.system('java -cp eep-1.0-SNAPSHOT-jar-with-dependencies.jar rider.eep.ParallelEEP graph.txt %s %s/%s.txt' %
                  (epsilon, out_path, epsilon))
    end = current_time()
    f_out.write('EEP_End\t%s\n' % end)

    out_path = 'features'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        os.system('rm -f features/*')

    start = current_time()
    f_out.write('RIDER_Start\t%s\n' % start)
    os.system('python riders_features.py -g %s -b 17 -rd partition -od features' % (in_file))

    end = current_time()
    f_out.write('RIDER_End\t%s\n' % end)

    fo = open('mdl.txt', 'w')
    fo.close()

    start = current_time()
    f_out.write('NMF_Start\t%s\n' % start)

    os.system('seq 60 -1 1 | parallel -j60 python parameterized_nmf.py -nf features/out-features.txt -r {}')

    end = current_time()
    f_out.write('NMF_End\t%s\n' % end)