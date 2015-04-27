__author__ = 'pratik'

import features
import argparse
import numpy as np
from collections import defaultdict


def get_refex_features_as_dict(refex_fx_file):
    refex_features = defaultdict(list)
    for line in open(refex_fx_file):
        line = line.strip().split(',')
        values = [float(val) for val in line]
        node_id = int(values[0])
        feature_row = values[1:]
        refex_features[node_id] = feature_row
    return refex_features


def merge_features(rider_fx, refex_fx):
    final_fx_matrix = []
    for node in rider_fx.keys():
        feature_row = [node]
        feature_row.extend(rider_fx[node])
        feature_row.extend(refex_fx[node])
        final_fx_matrix.append(tuple(feature_row))
    return np.array(final_fx_matrix)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('-rx', '--refex-features', help='refex feature file', required=True)
    argument_parser.add_argument('-o', '--output-file', help='combined fx output file', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)
    out_file = args.output_file
    refex_file = args.refex_features

    fx = features.Features()

    rider_features = fx.only_riders_as_dict(graph_file=graph_file, rider_dir=rider_dir, bins=bins)
    refex_features = get_refex_features_as_dict(refex_file)
    print 'Rider FX: %s, Refex FX: %s' % (len(rider_features[0]), len(refex_features[0]))

    merged_fx_matrix = merge_features(rider_features, refex_features)
    np.savetxt(out_file, X=merged_fx_matrix, delimiter=',')