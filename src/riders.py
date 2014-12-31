import huffman
import features
import argparse
import sys
import numpy as np
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from nonnegfac.nmf import NMF
from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT


def code_frequencies(matrix):
    frequencies = {}
    for (x, y), code in np.ndenumerate(matrix):
        if x == y:
            continue
        if code in frequencies:
            frequencies[code] += 1.0
        else:
            frequencies[code] = 1.0
    return frequencies


def get_huffman_code_length(sub_matrix):
    threshold = 1e-5
    whitened_mat = whiten(sub_matrix)
    code_book, distortion = kmeans(whitened_mat, number_bins, thresh=threshold)
    quantized_mat = vq(whitened_mat, code_book)
    frequencies = code_frequencies(quantized_mat)
    frequency_values = [frequencies[code] for code in frequencies.keys()]
    Z = sum(frequency_values)
    probabilities = [x / Z for x in frequency_values]
    huffman_codes = huffman.huffman(probabilities)
    bit_length = sum(value * len(code) for value, code in zip(frequency_values, huffman_codes))
    avg_length = huffman.symbol_code_expected_length(probabilities, huffman_codes)
    return avg_length


def get_reconstruction_cost(actual_matrix, estimated_matrix):
    reconstruction_error = 0.0
    for (i, j), value in np.ndenumerate(actual_matrix):
        if i == j:
            continue
        if estimated_matrix[i][j] > 0.0 and actual_matrix[i][j]/estimated_matrix[i][j] > 1.0:
            reconstruction_error += (actual_matrix[i][j] * np.log(actual_matrix[i][j] / estimated_matrix[i][j])
                                     - actual_matrix[i][j] + estimated_matrix[i][j])
    return reconstruction_error


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)

    fx = features.Features()

    actual_fx_matrix = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins)
    m, n = actual_fx_matrix.shape

    number_nodes = fx.graph.number_of_nodes()
    number_bins = int(np.log2(number_nodes))
    max_roles = min([number_nodes, n])
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0
    best_W = None
    best_H = None

    for bases in xrange(2, max_roles + 1):
        W, H, info = NMF().run(actual_fx_matrix, bases, max_iter=50)
        estimated_fx_matrix = W.dot(np.transpose(H))

        code_length_W = get_huffman_code_length(W)
        code_length_H = get_huffman_code_length(H)
        # model_cost = code_length_W + code_length_H
        model_cost = code_length_W * (W.shape[0] + W.shape[1]) + code_length_H * (H.shape[0] + H.shape[1])

        reconstruction_error = get_reconstruction_cost(actual_fx_matrix, estimated_fx_matrix)

        description_length = model_cost + reconstruction_error

        print 'Number of Roles: %s, Model Cost: %.2f, Reconstruction Error: %.2f, Description Length: %.2f' % \
              (bases, model_cost, reconstruction_error, description_length)

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_W = np.copy(W)
            best_H = np.copy(H)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 5:
                break

    print best_W.shape, best_H.shape