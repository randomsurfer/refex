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

    def test_should_compute_rider_primitives(self):
        fx = features.Features()
        fx.load_graph("resources/sample_graph_2.txt")
        fx.compute_primitive_features(rider_fx=True, rider_dir="resources/riders/")
        for vertex in fx.graph.nodes():
            self.assertEquals(len(fx.graph.node[vertex]), 268)
        # TODO: Need to add tests around the computed rider features