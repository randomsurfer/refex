__author__ = 'pratik'

import sys
from collections import defaultdict

try:
    id_file = sys.argv[1]
    node_role_file = sys.argv[2]
    output_file = sys.argv[3]
except Exception:
    print 'usage:: python %s <node_id_sequence> <node_role_file> <output_file>' % sys.argv[0]
    sys.exit(1)

node_id_sequence = []
node_role = []

for line in open(id_file):
    line = line.strip().split()[0]
    node_id = int(float(line))
    node_id_sequence.append(node_id)

for line in open(node_role_file):
    line = line.strip()
    line = line.split()
    nr = []
    all_zeros = 0.0
    for idx, value in enumerate(line):
        value = float(value)
        nr.append([value, idx])
        all_zeros += value

    if all_zeros > 0.0:
        node_role.append(sorted(nr, key=lambda x: x[0], reverse=True)[0][1])
        # node_role_secondary = sorted(nr, key=lambda x: x[0], reverse=True)
        # if len(node_role_secondary) > 1:
        #     secondary_role_val = node_role_secondary[1][0]
        #     if secondary_role_val > 0.0:
        #         node_role.append(node_role_secondary[1][1])
        #     else:
        #         node_role.append(-1)
        # else:
        #     node_role.append(-1)
    else:
        node_role.append(-1)

partition = defaultdict(list)

for node_id, role_id in zip(node_id_sequence, node_role):
    if role_id > -1:
        partition[role_id].append(node_id)

fw = open(output_file, 'w')

for key in sorted(partition.keys()):
    for node in sorted(partition[key]):
        fw.write('%s ' % node)
    fw.write('\n')

if len(partition) < 2:
    print len(partition), node_role_file
fw.close()