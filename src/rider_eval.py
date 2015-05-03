import features
import argparse
import numpy as np
import nimfa
import mdl
from sklearn import metrics
import pickle


def get_role_assignment(node_role_matrix):
    nodes, roles = node_role_matrix.shape
    primary_roles = []
    secondary_roles = []
    for node in xrange(nodes):
        node_labels = []
        for role in xrange(roles):
            node_labels.append((node_role_matrix[node][role], role))
        sorted_node_labels = sorted(node_labels, key=lambda x: x[0], reverse=True)

        if sorted_node_labels[0][0] > 0.0:
            primary_roles.append(sorted_node_labels[0][1])
        else:
            primary_roles.append(-1)
        if sorted_node_labels[1][0] > 0.0:
            secondary_roles.append(sorted_node_labels[1][1])
        else:
            secondary_roles.append(-1)
    return primary_roles, secondary_roles


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute rider matrix')
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

    full_fx_matrix = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins, prune=False)
    number_nodes = fx.graph.number_of_nodes()
    number_bins = int(np.log2(number_nodes))

    n, f = full_fx_matrix.shape

    mdlo = mdl.MDL(number_bins)
    code_length = mdlo.get_huffman_code_length(full_fx_matrix)
    model_cost = code_length * (n + f)

    primary_ari = []
    primary_ari_uniform = []
    secondary_ari = []
    secondary_ari_uniform = []
    model_costs = []
    model_costs_uniform = []

    for bins in xrange(1, 40):
        fx_b = features.Features()
        fx_bu = features.Features()

        binned_fx_matrix = fx_b.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins, prune=True, uniform=False)
        binned_fx_matrix_u = fx_bu.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins, prune=True, uniform=True)

        n_b, f_b = binned_fx_matrix.shape
        n_bu, f_bu = binned_fx_matrix_u.shape

        mdlo_b = mdl.MDL(number_bins)
        mdlo_bu = mdl.MDL(number_bins)

        code_length_b = mdlo_b.get_huffman_code_length(binned_fx_matrix)
        code_length_bu = mdlo_bu.get_huffman_code_length(binned_fx_matrix_u)

        model_cost_b = code_length_b * (n_b + f_b)
        model_cost_bu = code_length_bu * (n_bu + f_bu)

        model_costs.append((bins, model_cost_b))
        model_costs_uniform.append((bins, model_cost_bu))

        p_ari = []
        p_ari_u = []
        s_ari = []
        s_ari_u = []

        c = 0
        for rank in xrange(33, 43):
            for i in xrange(10):
                c += 1
                fctr = nimfa.mf(full_fx_matrix, rank=rank, method="lsnmf", max_iter=100)
                fctr_b = nimfa.mf(binned_fx_matrix, rank=rank, method="lsnmf", max_iter=100)
                fctr_bu = nimfa.mf(binned_fx_matrix_u, rank=rank, method="lsnmf", max_iter=100)

                fctr_res = nimfa.mf_run(fctr)
                W = np.asarray(fctr_res.basis())

                fctr_res_b = nimfa.mf_run(fctr_b)
                fctr_res_bu = nimfa.mf_run(fctr_bu)
                W_b = np.asarray(fctr_res_b.basis())
                W_bu = np.asarray(fctr_res_bu.basis())

                actual_primary, actual_secondary = get_role_assignment(W)
                estimated_primary, estimated_secondary = get_role_assignment(W_b)
                estimated_primary_u, estimated_secondary_u = get_role_assignment(W_bu)

                ari_1 = metrics.adjusted_rand_score(actual_primary, estimated_primary)
                ari_1_u = metrics.adjusted_rand_score(actual_primary, estimated_primary_u)
                ari_2 = metrics.adjusted_rand_score(actual_secondary, estimated_secondary)
                ari_2_u = metrics.adjusted_rand_score(actual_secondary, estimated_secondary_u)

                p_ari.append(ari_1)
                p_ari_u.append(ari_1_u)
                s_ari.append(ari_2)
                s_ari_u.append(ari_2_u)

        print "For Bins: ", bins
        primary_ari.append((bins, np.mean(p_ari)))
        primary_ari_uniform.append((bins, np.mean(p_ari_u)))
        secondary_ari.append((bins, np.mean(s_ari)))
        secondary_ari_uniform.append((bins, np.mean(s_ari_u)))

    pickle.dump(model_costs, open("model_cost.p", "wb"))
    pickle.dump(model_costs_uniform, open("model_cost_uniform.p", "wb"))
    pickle.dump(primary_ari, open("p_ari.p", "wb"))
    pickle.dump(primary_ari_uniform, open("p_ari_uniform.p", "wb"))
    pickle.dump(secondary_ari, open("s_ari.p", "wb"))
    pickle.dump(secondary_ari_uniform, open("s_ari_uniform.p", "wb"))