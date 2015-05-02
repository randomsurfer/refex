__author__ = 'pratik'

import sys
import random
import numpy as np

try:
    num_nodes = int(sys.argv[1])
    num_roles = int(sys.argv[2])
    out_dir = sys.argv[3]
    conf_id = sys.argv[4]
except IndexError:
    print 'usage: python %s <number-nodes> <number-roles> <out-dir> <conf-id>' % sys.argv[0]
    # python assign_random_roles.py 2202 18 /Users/pratik/Research/datasets/DBLP/coauthorship/CIKM/random_roles/coridex_05_09/ CIKM_05_09
    sys.exit(1)

partition = np.zeros((num_nodes, num_roles + 1))
random.seed(1000)

for node in xrange(num_nodes):
    role = random.randint(1, num_roles)
    partition[node][0] = node
    partition[node][role] = 1.0

ids_file = 'out-' + conf_id + '-ids.txt'
np.savetxt(out_dir + '/' + ids_file, partition[:, 0])

node_role_file = 'out-' + conf_id + '-nodeRoles.txt'
np.savetxt(out_dir + '/' + node_role_file, partition[:, 1:])