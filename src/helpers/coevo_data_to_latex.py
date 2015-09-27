__author__ = 'pratik'

import numpy as np
import os

row_labels = {'between': 1, 'close': 2, 'bcc': 3, 'ed0': 4, 'ed1': 5, 'ew0': 6, 'ew1': 7, 'deg': 8,
              'wdeg': 9, 'clusc': 10}
conf_ids = ['CIKM', 'ICDM', 'KDD', 'SDM', 'SIGMOD', 'VLDB']
methods = ['riders', 'riders_r', 'rolx', 'sparse', 'diverse']
latex_methods_name = {'riders': '\\riders', 'riders_r': '\\ridersr', 'rolx': '\\rolx', 'sparse': '\\glrds',
                      'diverse': '\\glrdd'}

file_suffix = '_10_13.txt'

labels = ['between', 'close', 'bcc', 'ed0', 'ed1', 'ew0', 'ew1', 'deg', 'wdeg', 'clusc']
print_labels = ['Betweenness', 'Closeness', 'BCC', 'Ego\_0\_Deg', 'Ego\_1\_Deg', 'Ego\_0\_Wt', 'Ego\_1\_Wt',
                'Degree', 'Wt\_Deg', 'Clus\_Coeff']

latex_labels = '\\textbf{%s} & \\multicolumn{1}{l|}{$\#Pairs_{t}$} & ' \
               '\\multicolumn{1}{l|}{$\#CoEvo_{t+\delta t}$} & ' \
               '\\multicolumn{1}{l|}{$Betweenness$} & \\multicolumn{1}{l|}{$Closeness$} & ' \
               '\\multicolumn{1}{l|}{$\#BCC$} & ' \
               '\\multicolumn{1}{l|}{$Ego\_1\_Deg$} & ' \
               '\\multicolumn{1}{l|}{$Ego\_1\_Wt$} & \\multicolumn{1}{l|}{$Degree$} & ' \
               '\\multicolumn{1}{l|}{$Wt. Degree$} & \\multicolumn{1}{l}{$Clus\_Coeff$} \\\\ \\midrule'

results = {}
for conf in conf_ids:
    input_dir = '/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/experiments_with_riders/node_coevolution/' + conf
    header = latex_labels % (conf)

    print '\\begin{tabular}{l|r|r|r|r|r|r|r|r|r|r} \\toprule'
    print header

    for method in methods:
        file_name = method + file_suffix
        inp_file = os.path.join(input_dir, file_name)
        data = np.loadtxt(inp_file)

        pairs = data[3][0]
        co_evolved = data[4][0]

        result_string = '%s & %s & %s & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) ' \
                        '& %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\\\ ' % (latex_methods_name[method],
                                                                             int(pairs), int(co_evolved*pairs/100.0),
                                                                             data[0][0], data[1][0],
                                                                             data[0][1], data[1][1],
                                                                             data[0][2], data[1][2],
                                                                             data[0][4], data[1][4],
                                                                             data[0][6], data[1][6],
                                                                             data[0][7], data[1][7],
                                                                             data[0][8], data[1][8],
                                                                             data[0][9], data[1][9])
        if method == 'riders_r':
            result_string += '\\midrule'
        elif method == 'diverse':
            result_string += '\\bottomrule'
        print result_string
    print '\\end{tabular}'
    print ''