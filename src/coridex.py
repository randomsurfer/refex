__author__ = 'pratik'

import features
import argparse
import nimfa
import numpy as np
import networkx as nx
from collections import defaultdict
import mdl


def fx_column_comparator(col_1, col_2, max_diff):
    # input two columns -> i.e. two node features and the max_dist
    # returns True/False if the two features agree within the max_dist criteria
    diff = float(max_diff) - abs(col_1 - col_2) + 0.0001
    return (diff >= 0.0001).all()


def prune_matrix(actual_matrix, max_diff):
    fx_graph = nx.Graph()
    n = actual_matrix.shape[1]
    for i in xrange(0, n-1):
        for j in xrange(i+1, n):
            if fx_column_comparator(actual_matrix[:, i], actual_matrix[:, j], max_diff):
                fx_graph.add_edge(i, j)

    cols_to_remove = []
    connected_fx = nx.connected_components(fx_graph)
    for cc in connected_fx:
        for col in sorted(cc[1:]):
            cols_to_remove.append(col)

    return np.delete(actual_matrix, cols_to_remove, axis=1)


def get_refex_features_as_dict(refex_fx_file):
    refex_features = defaultdict(list)
    for line in open(refex_fx_file):
        line = line.strip().split(',')
        values = [float(val) for val in line]
        node_id = int(values[0])
        feature_row = values[1:]
        refex_features[node_id] = feature_row
    return refex_features


def merge_features(rider_fx, refex_fx, prune=False):
    final_fx_matrix = []
    for node in rider_fx.keys():
        feature_row = [node]
        feature_row.extend(rider_fx[node])
        feature_row.extend(refex_fx[node])
        final_fx_matrix.append(tuple(feature_row))
    final_fx_matrix = np.array(final_fx_matrix)
    if prune:
        pruned_matrix = prune_matrix(final_fx_matrix[:, 1:], 0.0)
        n, f = pruned_matrix.shape
        pruned_matrix_with_node_ids = np.zeros((n, f+1))
        pruned_matrix_with_node_ids[:, 0] = final_fx_matrix[:, 0]
        pruned_matrix_with_node_ids[:, 1:] = pruned_matrix
        return pruned_matrix_with_node_ids
    else:
        return final_fx_matrix


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute coridex')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('-rx', '--refex-features', help='refex feature file', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)
    argument_parser.add_argument('-p', '--output-prefix', help='prefix', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)
    out_dir = args.output_dir
    prefix = args.output_prefix
    refex_file = args.refex_features

    fx = features.Features()

    rider_features = fx.only_riders_as_dict(graph_file=graph_file, rider_dir=rider_dir, bins=bins, all_features=False)
    refex_features = get_refex_features_as_dict(refex_file)
    print 'Rider FX: %s, Refex FX: %s' % (len(rider_features[0]), len(refex_features[0]))

    merged_fx_matrix = merge_features(rider_features, refex_features, prune=True)

    np.savetxt(out_dir + '/out-' + prefix + '-featureValues.csv', X=merged_fx_matrix, delimiter=',')
    np.savetxt(out_dir + '/out-' + prefix + '-ids.txt', X=merged_fx_matrix[:, 0])

    actual_fx_matrix = merged_fx_matrix[:, 1:]
    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    number_nodes = fx.graph.number_of_nodes()
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