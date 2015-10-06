import features
import argparse
import numpy as np


if __name__ == "__main__":
    np.random.seed(1004)
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='final output dir', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)
    out_dir = args.output_dir

    fx = features.Features()

    rider_features = fx.only_riders_as_dict(graph_file=graph_file, rider_dir=rider_dir, bins=bins, bin_features=True)

    actual_fx_matrix = []
    for node in sorted(rider_features.keys()):
        actual_fx_matrix.append(rider_features[node])

    actual_fx_matrix = np.array(actual_fx_matrix)

    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    np.savetxt(out_dir + '/out-features.txt', X=actual_fx_matrix)