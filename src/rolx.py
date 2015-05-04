__author__ = 'pratik'

import features
import argparse
import nimfa
import numpy as np
import networkx as nx
from collections import defaultdict
import mdl

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute coridex')
    argument_parser.add_argument('-rx', '--refex-features', help='refex feature file', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)
    argument_parser.add_argument('-p', '--output-prefix', help='prefix', required=True)

    args = argument_parser.parse_args()

    out_dir = args.output_dir
    prefix = args.output_prefix
    refex_file = args.refex_features

    refex_features = np.loadtxt(refex_file, delimiter=',')

    np.savetxt(out_dir + '/out-' + prefix + '-ids.txt', X=refex_features[:, 0])

    actual_fx_matrix = refex_features[:, 1:]
    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    number_nodes = n
    number_bins = int(np.log2(number_nodes))
    max_roles = min([number_nodes, f])
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0


    for rank in xrange(1, max_roles + 1):
        lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=1000)
        lsnmf_fit = lsnmf()
        W = np.asarray(lsnmf_fit.basis())
        H = np.asarray(lsnmf_fit.coef())
        estimated_matrix = np.asarray(np.dot(W, H))

        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        # For total bit length:
        # model_cost = code_length_W + code_length_H  # For total bit length
        # For avg. symbol bit length:
        model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        description_length = model_cost - loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_W = np.copy(W)
            best_H = np.copy(H)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 10:
                break

        print 'Number of Roles: %s, Model Cost: %.2f, -loglikelihood: %.2f, Description Length: %.2f, MDL: %.2f (%s)' \
              % (rank, model_cost, loglikelihood, description_length, minimum_description_length, best_W.shape[1])

    print 'MDL has not changed for these many iters:', min_des_not_changed_counter
    print '\nMDL: %.2f, Roles: %s' % (minimum_description_length, best_W.shape[1])

    np.savetxt(out_dir + '/out-' + prefix + "-nodeRoles.txt", X=best_W)
    np.savetxt(out_dir + '/out-' + prefix + "-roleFeatures.txt", X=best_H)