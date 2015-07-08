import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


row_labels = {'between': 1, 'close': 2, 'bcc': 3, 'ed0': 4, 'ed1': 5, 'ew0': 6, 'ew1': 7, 'deg': 8,
              'wdeg': 9, 'clusc': 10}
conf_ids = ['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB']
methods = ['corex', 'corex_r', 'corex_s', 'riders', 'rolx', 'sparse', 'diverse']

file_suffix = '_10_13.txt'

labels = ['between', 'close', 'bcc', 'ed0', 'ed1', 'ew0', 'ew1', 'deg', 'wdeg', 'clusc']
print_labels = ['Betweenness', 'Closeness', 'BCC', 'Ego\_0\_Deg', 'Ego\_1\_Deg', 'Ego\_0\_Wt', 'Ego\_1\_Wt',
                'Degree', 'Wt\_Deg', 'Clus\_Coeff']

TOLERANCE = 0.0
results = {}

for conf in conf_ids:
    input_dir = '/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/experiments/node_coevolution/' + conf
    final = conf + '\t' + '\t'.join(print_labels)
    print final
    for method in methods:
        results[method] = {}
        for property in row_labels.keys():
            diffs_matrix = np.loadtxt(os.path.join(input_dir, method+file_suffix))
            diffs_property = diffs_matrix[row_labels[property] - 1]
            c = 0
            d = 0
            for val in diffs_property:
                if val == 0.0 + TOLERANCE:
                    c += 1
                if val > 0.0 + TOLERANCE:
                    d += 1
            a = c / float(len(diffs_property)) * 100.0
            results[method][property] = '%.2f' % a

    for method in methods:
        final = method + '\t'
        for property in labels:
            final += '%s\t' % results[method][property]
        print final