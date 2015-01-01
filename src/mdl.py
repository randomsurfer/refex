import huffman
import numpy as np
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq


class MDL:
    def __init__(self, bins):
        self.threshold = 1e-5
        self.code_bins = bins

    def code_frequencies(self, matrix):
        frequencies = {}
        for (x, y), code in np.ndenumerate(matrix):
            if x == y:
                continue
            if code in frequencies:
                frequencies[code] += 1.0
            else:
                frequencies[code] = 1.0
        return frequencies

    def get_huffman_code_length(self, sub_matrix, return_code='avg'):
        # return_code 'avg': returns the average code length per symbol in bit (default)
        # otherwise: returns the total symbol length in bit required to encode the data

        whitened_mat = whiten(sub_matrix)
        code_book, distortion = kmeans(whitened_mat, self.code_bins, thresh=self.threshold)
        quantized_mat = vq(whitened_mat, code_book)
        frequencies = self.code_frequencies(quantized_mat)
        frequency_values = [frequencies[code] for code in frequencies.keys()]
        Z = sum(frequency_values)
        probabilities = [x / Z for x in frequency_values]
        huffman_codes = huffman.huffman(probabilities)

        if return_code == 'avg':
            return huffman.symbol_code_expected_length(probabilities, huffman_codes)  # Avg. symbol bit len
        else:
            return sum(value * len(code) for value, code in zip(frequency_values, huffman_codes))  # total sym bits

    def get_reconstruction_cost(self, actual_matrix, estimated_matrix):
        # KLD based reconstruction error. For more details refer
        reconstruction_error = 0.0
        for (i, j), value in np.ndenumerate(actual_matrix):
            if i == j:
                continue
            if estimated_matrix[i][j] > 0.0 and actual_matrix[i][j] / estimated_matrix[i][j] > 1.0:
                reconstruction_error += (actual_matrix[i][j] * np.log(actual_matrix[i][j] / estimated_matrix[i][j])
                                         - actual_matrix[i][j] + estimated_matrix[i][j])
        return reconstruction_error