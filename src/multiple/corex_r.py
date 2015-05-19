__author__ = 'pratik'

import nimfa
import numpy as np
import argparse

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute corex right sparse')
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

    sparsity_threshold = 2.0
    for i in xrange(1, 6):
        for rank in xrange(20, 29 + 1):
            snmf = nimfa.Snmf(actual_fx_matrix, seed="random_vcol", version='r', rank=rank, beta=2.0)
            snmf_fit = snmf()
            G = np.asarray(snmf_fit.basis())
            F = np.asarray(snmf_fit.coef())

            w_out = '%s-%s-%s-nodeRoles.txt' % (rank, i, out_prefix)
            h_out = '%s-%s-%s-roleFeatures.txt' % (rank, i, out_prefix)

            np.savetxt(out_dir + w_out, X=G)
            np.savetxt(out_dir + h_out, X=F)
