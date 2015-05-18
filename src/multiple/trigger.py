__author__ = 'pratik'

import os
import sys

out_dir = '/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/experiments/msd_experiments/cikm/'
input_path = '/Users/pratik/Research/datasets/DBLP/coauthorship/New_Experiments/CIKM/'

nf = input_path + 'corex_05_10/out-CIKM_05_10-featureValues.csv'
prefix = 'corex'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/corex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
prefix = 'corex_r'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/corex_r.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
prefix = 'corex_s'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/glrd_sparse.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))

nf = input_path + 'riders_05_10/out-CIKM_05_10-featureValues.csv'
prefix = 'riders'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/corex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))

nf = input_path + 'rolx_05_10/out-CIKM_05_10-featureValues.csv'
prefix = 'rolx'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/corex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
prefix = 'sparse'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/glrd_sparse.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
prefix = 'diverse'
os.system('python /Users/pratik/Projects/personal/refex/src/multiple/glrd_diverse.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
