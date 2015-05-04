import features
import mdl
import argparse
import numpy as np
import nimfa


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)
    argument_parser.add_argument('-p', '--output-prefix', help='prefix', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)
    out_dir = args.output_dir
    prefix = args.output_prefix

    fx = features.Features()

    binned_fx_matrix = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins, bin_features=True)
    actual_fx_matrix = fx.prune_matrix(binned_fx_matrix, 0.0)

    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', n

    fx_matrix_with_node_ids = np.zeros((n, f+1))
    fx_matrix_with_node_ids[:, 0] = np.array([float(node) for node in xrange(n)])
    fx_matrix_with_node_ids[:, 1:] = actual_fx_matrix
    np.savetxt(out_dir + '/out-' + prefix + '-featureValues.csv', X=fx_matrix_with_node_ids, delimiter=',')
    np.savetxt(out_dir + '/out-' + prefix + '-ids.txt', X=fx_matrix_with_node_ids[:, 0])

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
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
