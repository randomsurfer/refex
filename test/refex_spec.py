import unittest
import refex


class RefexSpec(unittest.TestCase):
    def test_should_load_graph(self):
        rfx = refex.Refex()
        rfx.load_graph("resources/sample_graph.txt")
        self.assertEquals(rfx.get_number_of_vertices(), 6)
        self.assertEquals(rfx.get_degree_of_vertex(0), 4)
        self.assertEquals(rfx.get_degree_of_vertex(1), 3)
        self.assertEquals(rfx.get_degree_of_vertex(2), 2)
        self.assertEquals(rfx.get_degree_of_vertex(3), 2)
        self.assertEquals(rfx.get_degree_of_vertex(4), 4)
        self.assertEquals(rfx.get_degree_of_vertex(5), 1)
        self.assertEquals(rfx.get_egonet_members(0), [1, 2, 3, 4])
        self.assertEquals(rfx.get_egonet_members(1), [0, 2, 4])
        self.assertEquals(rfx.get_egonet_members(5), [4])

    def test_should_init_refex_log_binned_buckets(self):
        rfx = refex.Refex()
        rfx.load_graph("resources/sample_graph.txt")
        expected_log_binned_buckets = {0: 3, 1: 2, 2: 1}
        rfx.init_log_binned_fx_buckets()
        self.assertEquals(rfx.refex_log_binned_buckets, expected_log_binned_buckets)

    def test_recursive_fx_extraction(self):
        rfx = refex.Refex()
        rfx.load_graph("resources/sample_graph.txt")
        rfx.init_log_binned_fx_buckets()
        features = {"degree": [(0, 4), (1, 3), (2, 2), (3, 2), (4, 4), (5, 1)]}
        expected_vertex_fx = {5: [0], 2: [0], 3: [0], 1: [1], 0: [1], 4: [1]}  # fx_value of 1 has 2 candidates,
        # but there is a tie for node fx value of vertex 0 and 4, hence they belong to fx_value 1 and not 2
        first_iteration_computed_vertex_features = rfx.compute_recursive_features(features)
        self.assertEquals(first_iteration_computed_vertex_features, expected_vertex_fx)
