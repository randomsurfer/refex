import unittest
import features
import numpy as np


class FeaturesSpec(unittest.TestCase):
    def test_should_load_graph(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph.txt")
        self.assertEquals(fx.graph.number_of_nodes(), 4)
        self.assertEquals(fx.graph.number_of_edges(), 10)
        self.assertEquals(fx.graph[1][2]['weight'], 1)
        self.assertEquals(fx.graph[3][4]['weight'], 2)

    def test_should_return_egonet(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph.txt")
        self.assertEquals(fx.get_egonet_members(1), [1, 2, 3, 4])
        self.assertEquals(fx.get_egonet_members(2), [1, 2, 3])
        self.assertEquals(fx.get_egonet_members(2, level=1), [1, 2, 3, 4])
        self.assertEquals(fx.get_egonet_members(4), [1, 3, 4])
        self.assertEquals(fx.get_egonet_members(4, level=1), [1, 2, 3, 4])

    def test_should_compute_primitives(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph.txt")
        fx.compute_primitive_features()
        self.assertEquals(fx.graph.node[1]['wn0'], 4)
        self.assertEquals(fx.graph.node[1]['wn1'], 4)
        self.assertEquals(fx.graph.node[2]['wn0'], 3)
        self.assertEquals(fx.graph.node[2]['wn1'], 4)

    def test_should_init_vertical_bins(self):
        fx = features.Features()
        fx.no_of_vertices = 4
        fx.init_log_binned_fx_buckets()
        self.assertEquals(len(fx.refex_log_binned_buckets), fx.no_of_vertices)
        self.assertEquals(fx.refex_log_binned_buckets, [0, 0, 1, 2])

        fx.no_of_vertices = 5
        fx.refex_log_binned_buckets = []
        fx.init_log_binned_fx_buckets()
        self.assertEquals(len(fx.refex_log_binned_buckets), fx.no_of_vertices)
        self.assertEquals(fx.refex_log_binned_buckets, [0, 0, 1, 1, 2])

        fx.no_of_vertices = 8
        fx.refex_log_binned_buckets = []
        fx.init_log_binned_fx_buckets()
        self.assertEquals(len(fx.refex_log_binned_buckets), fx.no_of_vertices)
        self.assertEquals(fx.refex_log_binned_buckets, [0, 0, 0, 0, 1, 1, 2, 3])

    def test_should_vertical_bin_correctly(self):
        fx = features.Features()
        fx.no_of_vertices = 6
        fx.init_log_binned_fx_buckets()
        actual = fx.vertical_bin([(0, 4), (1, 3), (2, 2), (3, 2), (4, 4), (5, 1)])
        expected = {5: 0, 2: 0, 3: 0, 1: 1, 0: 1, 4: 1}  # fx_value of 1 has 2 candidates,
        self.assertEquals(actual, expected)

    def test_should_compute_rider_primitives(self):
        # TODO: Need to add tests around the computed rider and local features
        fx = features.Features()
        fx.load_graph("resources/sample_graph_2.txt")
        fx.compute_primitive_features(rider_fx=False, rider_dir="resources/riders/")
        for vertex in fx.graph.nodes():
            self.assertEquals(len(fx.graph.node[vertex]), 56)

        fx = features.Features()
        fx.load_graph("resources/sample_graph_2.txt")
        fx.compute_primitive_features(rider_fx=True, rider_dir="resources/riders/")
        for vertex in fx.graph.nodes():
            self.assertEquals(len(fx.graph.node[vertex]), 536)

    def test_should_compare_cols_within_maxdist(self):
        fx = features.Features()

        column_1 = np.array([1.0, 2.0, 3.0])
        column_2 = np.array([1.0, 2.0, 3.0])

        max_dist = 0
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertTrue(actual)

        max_dist = 1
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertTrue(actual)

        column_1 = np.array([1.0, 2.0, 3.0])
        column_2 = np.array([1.5, 2.5, 3.5])

        max_dist = 0
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertFalse(actual)

        max_dist = 1
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertTrue(actual)

        max_dist = 0.5
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertTrue(actual)

        max_dist = 0.49
        actual = fx.fx_column_comparator(column_1, column_2, max_dist)
        self.assertFalse(actual)

    def test_should_prune_similar_features(self):
        fx = features.Features()

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 0)
        expected = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

        expected_dtype_name = ['a', 'b', 'c']

        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertTrue((actual == expected).all())

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(1.0, 4.0, 5.0), (1.0, 4.0, 5.0), (1.0, 4.0, 5.0)],
                                  dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 0)
        expected = np.array([(1.0, 2.0, 3.0, 4.0, 5.0), (1.0, 2.0, 3.0, 4.0, 5.0), (1.0, 2.0, 3.0, 4.0, 5.0)],
                            dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ('e', '<f8'), ('f', '<f8')])
        expected_dtype_name = ['a', 'b', 'c', 'e', 'f']
        expected_not_dtype_name = ['b', 'c', 'd', 'e', 'f']

        self.assertTrue((actual == expected).all())
        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertNotEqual(list(actual.dtype.names), expected_not_dtype_name)

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(1.0, 6.0, 5.0), (1.0, 6.0, 5.0), (1.0, 6.0, 5.0)],
                                     dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 1)
        expected = np.array([(1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ('e', '<f8')])

        expected_dtype_name = ['a', 'b', 'c', 'e']

        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertTrue((actual == expected).all())

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(1.0, 6.0, 5.0), (1.0, 6.0, 5.0), (1.0, 6.0, 5.0)],
                                     dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 1.5)
        expected = np.array([(1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ('e', '<f8')])

        expected_dtype_name = ['a', 'b', 'c', 'e']

        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertTrue((actual == expected).all())

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(1.0, 6.0, 5.0), (1.0, 6.0, 5.0), (1.0, 6.0, 5.0)],
                                     dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 2)
        expected = np.array([(1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0), (1.0, 2.0, 3.0, 6.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8'), ('e', '<f8')])

        expected_dtype_name = ['a', 'b', 'c', 'e']

        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertTrue((actual == expected).all())

        prev_fx_matrix = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])
        new_fx_matrix = np.array([(3.0, 6.0, 5.0), (3.0, 6.0, 5.0), (3.0, 6.0, 5.0)],
                                     dtype=[('d', '<f8'), ('e', '<f8'), ('f', '<f8')])

        actual = fx.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, 2)
        expected = np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)],
                                  dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

        expected_dtype_name = ['a', 'b', 'c']

        self.assertEquals(list(actual.dtype.names), expected_dtype_name)
        self.assertTrue((actual == expected).all())