__author__ = 'pratik'

import sys
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict
import node_sense

try:
    node_role_file = sys.argv[1]
    node_measurement_file = sys.argv[2]
    node_id_file = sys.argv[3]
except IndexError:
    print 'usage: python %s <node-role-file> <node-measurements> <node-id-file>' % sys.argv[0]
    sys.exit(1)

node_id_seq = []
for line in open(node_id_file):
    line = line.strip()
    val = int(float(line))
    node_id_seq.append(val)

try:
    node_roles = np.loadtxt(node_role_file)
    node_measurements = np.loadtxt(node_measurement_file, delimiter=',')
except ValueError as ve:
    print ve.args

# Node-Measurement matrix has node_id in the first column
# and is sorted according to node ids too
# Node-Role matrix is aligned with reference to the out-ids file
# Hence, the alignment of the measurement matrix to the node ids in the node role matrix

aligned_node_measurements = []
for node_id in node_id_seq:
    aligned_node_measurements.append(node_measurements[node_id, 1:])

aligned_node_measurements = np.asarray(aligned_node_measurements)
normalized_measurements = normalize(aligned_node_measurements, norm='l2', axis=0)

E = node_sense.estimate_basis(normalized_measurements, normalize(node_roles))
G_ones = np.ones((aligned_node_measurements.shape[0], 1))
E_ones = node_sense.estimate_basis(normalized_measurements, G_ones)

measurement_labels = ['Betweenness', 'Closeness', 'Degree', 'Clustering Coeff', '#BCC', 'Wt. Degree']

E = np.asarray(E)
E_ones = np.asarray(E_ones)

node_sense_matrix = node_sense.get_node_sense_matrix(E, E_ones)

# for each property compute AAD |E(r, s) - E_ran(r, s)|
num_nodes, num_roles = node_roles.shape

aads = defaultdict(list)

for i in xrange(50):
    random_role_assignment = node_sense.get_random_role_assignment(num_nodes, num_roles, i+1000)

    E_ran = node_sense.estimate_basis(normalized_measurements, normalize(random_role_assignment))
    E_ran = np.asarray(E_ran)

    random_sense_matrix = node_sense.get_node_sense_matrix(E_ran, E_ones)

    for j, label in enumerate(measurement_labels):
        label_measurement = node_sense_matrix[:, j]
        random_label_measurement = random_sense_matrix[:, j]
        aad = np.mean(np.abs(label_measurement - random_label_measurement))
        aads[label].append(aad)

final_str = ''
for label in measurement_labels:
    val = '%.2f' % np.mean(aads[label])
    final_str += val + '\t'
# print '\t'.join(measurement_labels)
print final_str
