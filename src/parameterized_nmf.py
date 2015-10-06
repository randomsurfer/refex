import features
import mdl
import argparse
import numpy as np
import nimfa
from collections import defaultdict


if __name__ == "__main__":
    np.random.seed(1004)
    argument_parser = argparse.ArgumentParser(prog='param nmf')
    argument_parser.add_argument('-nf', '--node-feature', help='node feature file', required=True)
    argument_parser.add_argument('-r', '--rank', help='nmf rank', required=True)

    args = argument_parser.parse_args()

    nf_file = args.node_feature
    rank = int(args.rank)

    actual_fx_matrix = np.loadtxt(nf_file)
    n, f = actual_fx_matrix.shape

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0

    lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=20)
    lsnmf_fit = lsnmf()
    W = np.asarray(lsnmf_fit.basis())
    H = np.asarray(lsnmf_fit.coef())
    estimated_matrix = np.asarray(np.dot(W, H))

    code_length_W = mdlo.get_huffman_code_length(W)
    code_length_H = mdlo.get_huffman_code_length(H)

    model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
    loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

    description_length = model_cost - loglikelihood

    fo = open('mdl.txt', 'a')
    final_str = '%s\t%.2f\n' % (rank, description_length)
    fo.write(final_str)
    fo.close()