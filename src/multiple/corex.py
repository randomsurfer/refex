__author__ = 'pratik'

import argparse
import nimfa
import numpy as np


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute corex')
    argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    argument_parser.add_argument('-o', '--output-prefix', help='glrd output prefix', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='glrd output dir', required=True)

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    out_prefix = args.output_prefix
    out_dir = args.output_dir

    refex_features = np.loadtxt(node_feature, delimiter=',')
    actual_fx_matrix = refex_features[:, 1:]
    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    for i in xrange(1, 6):
        for rank in xrange(20, 29 + 1):
            lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=500)
            lsnmf_fit = lsnmf()
            W = np.asarray(lsnmf_fit.basis())
            H = np.asarray(lsnmf_fit.coef())

            w_out = '%s-%s-%s-nodeRoles.txt' % (rank, i, out_prefix)
            h_out = '%s-%s-%s-roleFeatures.txt' % (rank, i, out_prefix)

            np.savetxt(out_dir + w_out, X=W)
            np.savetxt(out_dir + h_out, X=H)