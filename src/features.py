
class Features:
    def __init__(self):
        self.graph = {}

    def load_graph(self, file_name):
        source = 0
        for line in open(file_name):
            line = line.strip()
            line = line.split()
            for dest in line:
                dest = int(dest)
                if source not in self.graph:
                    self.graph[source] = [dest]
                else:
                    self.graph[source].append(dest)
                if dest not in self.graph:
                    self.graph[dest] = [source]
                else:
                    self.graph[dest].append(source)
            source += 1
        for key in self.graph.keys():
            self.graph[key] = list(set(self.graph[key]))

    def get_number_of_vertices(self):
        return len(self.graph)

    def get_degree_of_vertex(self, vertex):
        return len(self.graph[vertex])

    def get_egonet_degree(self, vertex):
        adjacency_list = self.graph[vertex]
        return sum([self.get_degree_of_vertex(v) for v in adjacency_list])

    def get_egonet_members(self, vertex):
        return self.graph[vertex]

    def get_count_of_edges_leaving_egonet(self, vertex):
        edges_leaving_egonet = 0
        egonet = self.get_egonet_members(vertex)
        for vertex in egonet:
            for neighbour in self.get_egonet_members(vertex):
                if neighbour not in egonet:
                    edges_leaving_egonet += 1
        return edges_leaving_egonet

    def get_count_of_within_egonet_edges(self, vertex):
        within_egonet_edges = 0
        egonet = self.get_egonet_members(vertex)
        for vertex in egonet:
            for neighbour in self.get_egonet_members(vertex):
                if neighbour in egonet:
                    within_egonet_edges += 1
        return within_egonet_edges / 2

