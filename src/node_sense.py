__author__ = 'pratik'

import sys
import numpy as np
import scipy.optimize as opt


'''
V ~ GF
G is the basis matrix
F is the coeff matrix
---
Node Sense
G.E ~ M
E.T G.T ~ M.T
(m x r) (r x n) ~ m x n
M - node measurement matrix node x |measurements|
G - node-role matrix n x r
E - role-measurement matrix r x |measurements| (role contribution to node measurements)

def estimate_W(V, H):
    W = np.zeros((V.shape[0], H.shape[0]))
    print V.shape, H.shape
    for j in xrange(0, W.shape[0]):
        res = opt.nnls(H.T, V[j, :])
        W[j, :] = res[0]
    return W


'''

def estimate_coeff(measurements, basis):
    coeff = np.zeros((measurements.shape[0], basis.shape[0]))
    for j in xrange(0, coeff.shape[0]):
        res = opt.nnls(basis.T, measurements[j, :])
        coeff[j, :] = res[0]
    return coeff


def estimate_basis(measurements, coeff):
    return estimate_coeff(measurements.T, coeff.T).T

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

E = estimate_basis(aligned_node_measurements, node_roles)
G_ones = np.ones((aligned_node_measurements.shape[0], 1))
E_ones = estimate_basis(aligned_node_measurements, G_ones)

measurement_labels = ['Betweenness', 'Closeness', 'Degree', 'CC', 'BCC', 'WtDeg']
print '\t'.join(measurement_labels)
E = np.asarray(E)
E_ones = np.asarray(E_ones)

for r in xrange(E.shape[0]):
    m = []
    for s in xrange(E.shape[1]):
        try:
            m.append(str(E[r][s] / E_ones[0][s] * 100.0))
        except IndexError:
            print r,s
    print '\t'.join(m)