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
        fx =features.Features()
        fx.load_graph("resources/sample_graph.txt")
        self.assertEquals(fx.get_egonet_members(1), [1, 2, 3, 4])
        self.assertEquals(fx.get_egonet_members(2), [1, 2, 3])
        self.assertEquals(fx.get_egonet_members(2, level=1), [1, 2, 3, 4])
        self.assertEquals(fx.get_egonet_members(4), [1, 3, 4])
        self.assertEquals(fx.get_egonet_members(4, level=1), [1, 2, 3, 4])