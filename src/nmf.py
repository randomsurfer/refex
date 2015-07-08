import mdl
import argparse
import numpy as np
import nimfa


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-nf', '--node_feature', help='node-feature matrix', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)
    argument_parser.add_argument('-p', '--output-prefix', help='prefix', required=True)

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    out_dir = args.output_dir
    prefix = args.output_prefix

    actual_fx_matrix = np.loadtxt(node_feature, delimiter=',')[:, 1:]
    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    max_roles = 20
    best_W = None
    best_H = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0

    for rank in xrange(20, max_roles + 1):
        lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=100)
        lsnmf_fit = lsnmf()
        W = np.asarray(lsnmf_fit.basis())
        H = np.asarray(lsnmf_fit.coef())
        estimated_matrix = np.asarray(np.dot(W, H))

        code_length_W = mdlo.get_huffman_code_length(W)
        code_length_H = mdlo.get_huffman_code_length(H)

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
            if min_des_not_changed_counter == 4:
                break

        print 'Number of Roles: %s, Model Cost: %.2f, -loglikelihood: %.2f, Description Length: %.2f, MDL: %.2f (%s)' \
              % (rank, model_cost, loglikelihood, description_length, minimum_description_length, best_W.shape[1])

    print 'MDL has not changed for these many iters:', min_des_not_changed_counter
    print '\nMDL: %.2f, Roles: %s' % (minimum_description_length, best_W.shape[1])
    np.savetxt(out_dir + '/out-' + prefix + "-nodeRoles.txt", X=best_W)
    np.savetxt(out_dir + '/out-' + prefix + "-roleFeatures.txt", X=best_H)
