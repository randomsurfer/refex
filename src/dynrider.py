import features
import argparse
import os
import numpy as np
from scipy.optimize import nnls as opt

def estimate_W(V, H):
    W = np.zeros((V.shape[0], H.shape[0]))
    for j in xrange(0, V.shape[0]):
        res = opt.nnls(H, V[j, :].toarray()[0, :])
        W[j, :] = res[0]
    return W


def load_role_fx_matrix(rf_matrix_file):
    matrix = []
    for line in open(rf_matrix_file):
        line = line.strip().split(',')
        row = [float(value) for value in line]
        matrix.append(row)
    return np.asarray(matrix)


def get_base_features(base_fx_dir, bins=15):
    base_features = {}
    for file_name in sorted(os.listdir(base_fx_dir)):
        if file_name == ".DS_Store":
            continue
        block_sizes = []

        for line in open(os.path.join(base_fx_dir, file_name)):
            line = line.strip().split()
            block_sizes.append(len(line))
        log_bins = np.logspace(np.log10(min(block_sizes)+1), np.log10(max(block_sizes)+1), bins)
        base_features[file_name] = log_bins


def digitize(block_size, log_bins, file_name):
    # block_size_value IS NOT the log10(block_size)
    # returns the feature_name corresponding to this block size assigned bin
    start = log_bins[0]
    i = 0
    for curr_bin in log_bins[1:]:
        if start <= block_size < curr_bin:
            return file_name + '_' + str(i)
        start = curr_bin
        i += 1

    # if here => value is greater than last bin value in the original feature space
    # return None and do a none check in the function that calls this one.
    return "DISCARD"

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='dynamic riders')
    argument_parser.add_argument('-g', '--init-graph', help='initial graph', required=True)
    argument_parser.add_argument('-gd', '--graph-dir', help='graph dir', required=True)
    argument_parser.add_argument('-bfd', '--base-fx-dir', help='base features directory', required=True)
    argument_parser.add_argument('-rf', '--role-feature', help='input role feature matrix', required=True)
    # argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    # argument_parser.add_argument('-dn', '--dir-no', help='max timestamp directory number', required=True)
    # argument_parser.add_argument('-o', '--output-dir', help='output dir', required=True)

    args = argument_parser.parse_args()

    initial_graph = args.init_graph
    graph_dir = args.graph_dir
    base_fx_dir = args.base_fx_dir
    rf_matrix = args.role_feature
    rider_dir = args.rider_dir
    # bins = int(args.bins)
    # max_dir_no = int(args.dir_no)
    # out_dir = args.output_dir
    H = load_role_fx_matrix(rf_matrix)
    print H.shape
    fx = features.Features()
    actual_matrix = fx.only_riders(graph_file=initial_graph,rider_dir=rider_dir, bins=15)
    print actual_matrix.shape



    # fx = features.Features()


