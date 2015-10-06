import os
import sys
import time
import networkx as nx


def load_graph(file_name):
    graph = nx.Graph()
    for line in open(file_name):
        line = line.strip()
        line = line.split(',')
        source = int(line[0])
        dest = int(line[1])
        graph.add_edge(source, dest)
    return graph


current_time = lambda: int(round(time.time() * 1000))


try:
    in_file = sys.argv[1]
except IndexError:
    print 'Usage:: python %s <graph_file>' % sys.argv[0]
    sys.exit(1)

graph = load_graph(in_file)

d = 0.0
for n in graph.nodes():
    d += float(len(graph.neighbors(n)))
V = graph.number_of_nodes()
avg_degree = int(V / d)

fo = open('graph.txt', 'w')

adj_lists = graph.adjacency_list()

for adj_list in adj_lists:
    adj = [str(a) for a in adj_list]
    final_str = ' '.join(adj)
    fo.write(final_str + '\n')
fo.close()


# Calculate the degree of the graph in this code
# 1. Run EEP, epsilon = 1 to degree
# 2. Run riders_features
# 3. Run parameterized_nmf in parallel
# os.system()
start = current_time()
print 'EEP_Start\t%s' % start
for epsilon in range(1, avg_degree+1):
    out_path = 'partition'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    else:
        os.system('rm -rf partition/*')
    os.system('java -cp eep-1.0-SNAPSHOT-jar-with-dependencies.jar rider.eep.ParallelEEP graph.txt %s %s/%s.txt' %
              (epsilon, out_path, epsilon))
end = current_time()
print 'EEP_End\t%s' % end

out_path = 'features'
if not os.path.exists(out_path):
    os.makedirs(out_path)
else:
    os.system('rm -rf features/*')

start = current_time()
print 'RIDER_Features_Start\t%s' % start
os.system('python riders_features.py -g %s -b 17 -rd partition -od features' % (in_file))

end = current_time()
print 'RIDER_Features_End\t%s' % end

fo = open('mdl.txt', 'w')
fo.close()

start = current_time()
print 'NMF_Start\t%s' % start

os.system('seq 100 2 | parallel python parameterized_nmf.py -nf features/out-features.txt -r {}')

end = current_time()
print 'NMF_End\t%s' % end
