__author__ = 'pratik'

import os
import sys

out_dir = '/Users/pratik/Research/datasets/DBLP/coauthorship/experiments/msd_experiments/CIKM'
input_path = '/Users/pratik/Research/datasets/DBLP/coauthorship/CIKM/'

prefix = 'corex'
nf = input_path + 'coridex_05_10/out-CIKM_05_10-featureValues.csv'

os.system('python /Users/pratik/Projects/personal/refex/src/multiple/coridex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))

nf = input_path + 'rolx_05_10/out-CIKM_05_10-featureValues.csv'
prefix = 'rolx'

os.system('python /Users/pratik/Projects/personal/refex/src/multiple/coridex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))


nf = input_path + 'rolx_05_10/out-CIKM_05_10-featureValues.csv'
prefix = 'rolx'

os.system('python /Users/pratik/Projects/personal/refex/src/multiple/coridex.py -nf %s -o %s -od %s' % (nf, prefix, out_dir))
