import unittest
import features


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