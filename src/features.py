import networkx as nx


class Features:
    """
    The features class. Placeholder for the graph and node features.
    """
    def __init__(self):
        """
        initializes an empty graph dict
        :return:
        """
        self.graph = nx.DiGraph()

    def load_graph(self, file_name):
        """
        loads a networkx DiGraph from a file. Support directed weighted graph. Each comma separated line is source, destination and weight.
        :param file_name: <file_name>
        :return:
        """
        for line in open(file_name):
            line = line.strip()
            line = line.split(',')
            source = int(line[0])
            dest = int(line[1])
            wt = float(line[2])
            self.graph.add_edge(source, dest, weight=wt)

    def get_egonet_members(self, vertex, level=0):
        """
        node's egonet members, same as its adjacency list
        :param vertex: vertex_id
        :param level: level <level> egonet. Default 0
        :return: returns level <level> egonet of the vertex
        """
        lvl_zero_egonet = self.graph.successors(vertex)
        lvl_zero_egonet.append(vertex)
        if level == 1:
            lvl_one_egonet = []
            for node in lvl_zero_egonet:
                if node != vertex:
                    lvl_one_egonet.extend(self.graph.successors(node))
            lvl_zero_egonet.extend(lvl_one_egonet)
        return list(set(lvl_zero_egonet))
