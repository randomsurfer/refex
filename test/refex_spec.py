import numpy as np
import unittest
import features
import refex


class RefexSpec(unittest.TestCase):
    def test_should_init_refex_log_binned_buckets(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph.txt")
        rfx = refex.Refex(fx.get_number_of_vertices())
        expected_log_binned_buckets = {0: 3, 1: 2, 2: 1}
        rfx.init_log_binned_fx_buckets()
        self.assertEquals(rfx.refex_log_binned_buckets, expected_log_binned_buckets)

    def test_recursive_fx_extraction(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph.txt")
        rfx = refex.Refex(fx.get_number_of_vertices())
        rfx.init_log_binned_fx_buckets()
        feature_dict = {"degree": [(0, 4), (1, 3), (2, 2), (3, 2), (4, 4), (5, 1)]}
        expected = {5: [0], 2: [0], 3: [0], 1: [1], 0: [1], 4: [1]}  # fx_value of 1 has 2 candidates,
        expected_vertex_fx = np.array([expected[v] for v in sorted(expected.keys())])
        expected_fx_column = expected_vertex_fx[:, 0]
        # but there is a tie for node fx value of vertex 0 and 4, hence they belong to fx_value 1 and not 2
        first_iteration_computed_vertex_features = rfx.compute_initial_features(feature_dict)
        self.assertEquals(first_iteration_computed_vertex_features[0], expected_vertex_fx[0])
        self.assertEquals(first_iteration_computed_vertex_features[1], expected_vertex_fx[1])
        self.assertEquals(first_iteration_computed_vertex_features[2], expected_vertex_fx[2])
        self.assertEquals(first_iteration_computed_vertex_features[3], expected_vertex_fx[3])
        self.assertEquals(first_iteration_computed_vertex_features[4], expected_vertex_fx[4])
        self.assertEquals(first_iteration_computed_vertex_features[5], expected_vertex_fx[5])
        self.assertEquals(list(first_iteration_computed_vertex_features[:, 0]), list(expected_fx_column))