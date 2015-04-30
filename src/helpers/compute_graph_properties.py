__author__ = 'pratik'

from collections import defaultdict
import networkx as nx
import numpy as np
import sys
import pickle

try:
    input_file = sys.argv[1]
    out_dir = sys.argv[2]
except IndexError:
    print 'usage: python %s <graph_file> <output_dir>' % sys.argv[0]
    sys.exit(1)

graph = nx.Graph()

for line in open(input_file):
    line = line.strip().split(',')
    s = int(line[0])
    d = int(line[1])
    w = float(line[2])
    graph.add_edge(s, d, weight=w)

betweenness = nx.betweenness_centrality(graph, normalized=True)
closeness = nx.closeness_centrality(graph, normalized=True)
# eccentricity = nx.eccentricity(graph)
clustering_coeff = nx.clustering(graph)
degree = nx.degree_centrality(graph)

components = nx.biconnected_components(graph)
biconn = defaultdict(int)
for component in components:
    for node in component:
        biconn[node] += 1

weighted_degree = defaultdict(int)
nodes = graph.nodes()
for node in nodes:
    adj_list = graph.neighbors(node)
    for neighbor in adj_list:
        weighted_degree[node] += graph[node][neighbor]['weight']

measurement_matrix = []
for node in sorted(nodes):
    node_measurements = [node]
    node_measurements.append(betweenness[node])
    node_measurements.append(closeness[node])
    node_measurements.append(degree[node])
    node_measurements.append(clustering_coeff[node])
    node_measurements.append(biconn[node])
    node_measurements.append(weighted_degree[node])
    measurement_matrix.append(node_measurements)

# betweenness = pickle.load( open( "betweenness.p", "rb" ) )
pickle.dump(betweenness, open(out_dir + '/betweenness.p', 'wb'))
pickle.dump(closeness, open(out_dir + '/closeness.p', 'wb'))
pickle.dump(degree, open(out_dir + '/degree.p', 'wb'))
pickle.dump(clustering_coeff, open(out_dir + '/clusteringcoeff.p', 'wb'))
pickle.dump(biconn, open(out_dir + '/biconn.p', 'wb'))
pickle.dump(weighted_degree, open(out_dir + '/weighteddegree.p', 'wb'))
np.savetxt(out_dir + '/measurements.txt', np.asarray(measurement_matrix), delimiter=',')

print '*'*50
print 'Finished: ', out_dir
print '*'*50

