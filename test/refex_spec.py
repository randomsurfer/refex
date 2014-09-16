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
