import features
import mdl
import argparse
import numpy as np
import pymf


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('-o', '--output-prefix', help='output prefix for factors', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)
    out_prefix = args.output_prefix

    fx = features.Features()

    actual_fx_matrix = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins)
    m, n = actual_fx_matrix.shape
    print 'Number of Features: ', n

    number_nodes = fx.graph.number_of_nodes()
    number_bins = int(np.log2(number_nodes))
    max_roles = min([number_nodes, n])
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0

    for rank in xrange(1, max_roles + 1):
        nmf_mdl = pymf.NMF(actual_fx_matrix, num_bases=rank)
        nmf_mdl.factorize(niter=30)
        W = nmf_mdl.W
        H = nmf_mdl.H
        estimated_matrix = np.dot(W, H)

        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

        # For total bit length:
        # model_cost = code_length_W + code_length_H  # For total bit length
        # For avg. symbol bit length:
        model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])
        reconstruction_error = mdlo.get_reconstruction_error(actual_fx_matrix, estimated_matrix)

        description_length = model_cost + reconstruction_error

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_W = np.copy(W)
            best_H = np.copy(H)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 30:
                break

        print 'Number of Roles: %s, Model Cost: %.2f, Reconstruct Err: %.2f, Description Length: %.2f, MDL: %.2f (%s)' \
              % (rank, model_cost, reconstruction_error, description_length, minimum_description_length, best_W.shape[1])

    print 'MDL has not changed for these many iters:', min_des_not_changed_counter
    print '\nMDL: %.2f, Roles: %s' % (minimum_description_length, best_W.shape[1])
    np.savetxt(out_prefix+"-nodeRoles.txt", X=best_W, delimiter=',')
    np.savetxt(out_prefix+"-rolesFeatures.txt", X=best_H, delimiter=',')