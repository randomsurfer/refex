import numpy as np
import networkx as nx


class Refex:
    def __init__(self):
        self.graph = {}
        self.p = 0.5  # fraction of nodes placed in log bin
        self.s = 0  # feature similarity threshold
        self.TOLERANCE = 0.0001
        self.MAX_ITERATIONS = 100
        self.refex_log_binned_buckets = {}

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

    def get_sorted_feature_values(self, feature_values):
        sorted_fx_values = sorted(feature_values, key=lambda x: x[1])
        return sorted_fx_values, len(sorted_fx_values)

    def vertical_bin(self, feature):
        vertical_binned_vertex = {}
        no_vertices = self.get_number_of_vertices()
        sorted_fx_values, sorted_fx_size = self.get_sorted_feature_values(feature)
        count_of_vertices_with_log_binned_fx_value = 0

        for log_binned_fx_value in sorted(self.refex_log_binned_buckets.keys()):
            if no_vertices == count_of_vertices_with_log_binned_fx_value:
                # case when all vertices have been already binned owing to ties/collision
                break

            no_vertices_in_current_log_binned_bucket = self.refex_log_binned_buckets[log_binned_fx_value]

            fx_value_of_last_vertex_to_be_taken = sorted_fx_values[count_of_vertices_with_log_binned_fx_value +
                                                                   no_vertices_in_current_log_binned_bucket - 1][1]
            # If there are ties, it may be necessary to include more than p|V| nodes
            for idx in xrange(count_of_vertices_with_log_binned_fx_value +
                              no_vertices_in_current_log_binned_bucket, sorted_fx_size):
                if sorted_fx_values[idx][1] == fx_value_of_last_vertex_to_be_taken:
                    no_vertices_in_current_log_binned_bucket += 1
                else:
                    break

            for idx in xrange(count_of_vertices_with_log_binned_fx_value,
                              count_of_vertices_with_log_binned_fx_value +
                              no_vertices_in_current_log_binned_bucket):
                vertex_no = sorted_fx_values[idx][0]
                vertical_binned_vertex[vertex_no] = log_binned_fx_value  # assign log binned value to vertex

            count_of_vertices_with_log_binned_fx_value += no_vertices_in_current_log_binned_bucket

        return vertical_binned_vertex

    def compute_initial_features(self, features):
        vertex_fx_vector = {}
        no_vertices = self.get_number_of_vertices()
        for k in xrange(0, no_vertices):
            vertex_fx_vector[k] = []

        for feature in sorted(features.keys()):
            feature_values = features[feature]  # assuming list of 2-tuples
            binned_feature_vertex = self.vertical_bin(feature_values)
            for v in binned_feature_vertex:
                vertex_fx_vector[v].append(binned_feature_vertex[v])

        return np.array([vertex_fx_vector[v] for v in sorted(vertex_fx_vector.keys())])

    def fx_column_comparator(self, col_1, col_2, max_diff):
        diff = max_diff - abs(col_1 - col_2)
        return (diff > self.TOLERANCE).all()

    def compare_and_prune_vertex_fx_vectors(self, prev_vertex_fx_vector, curr_vertex_fx_vector, max_diff):
        fx_graph = nx.Graph()
        cols_to_delete_in_curr = []
        cols_to_delete_in_prev = []
        col_p = prev_vertex_fx_vector.shape[1]
        col_c = curr_vertex_fx_vector.shape[1]

        # compare current with previous
        for i in xrange(0, col_p):
            for j in xrange(0, col_c):
                if self.fx_column_comparator(prev_vertex_fx_vector[:, i], curr_vertex_fx_vector[:, j], max_diff):
                    fx_graph.add_edge(i, col_p + j)

        # compare current with current
        for i in xrange(0, col_p - 1):
            for j in xrange(i + 1, col_p):
                if self.fx_column_comparator(curr_vertex_fx_vector[:, i], curr_vertex_fx_vector[:, j], max_diff):
                    fx_graph.add_edge(col_p + i, col_p + j)

        connected_fx = nx.connected_components(fx_graph)
        for cc in connected_fx:
            sorted_cc = sorted(cc)[1:]
            for v in sorted_cc:
                if v >= col_p:
                    cols_to_delete_in_curr.append(v - col_p)
                else:
                    cols_to_delete_in_prev.append(v)

        return np.append(np.delete(prev_vertex_fx_vector, cols_to_delete_in_prev, 1),
                         np.delete(curr_vertex_fx_vector, cols_to_delete_in_curr, 1), 0)

    def feature_dict_to_numpy_array(self, vertex_fx_dict):
        return np.array([vertex_fx_dict[v] for v in sorted(vertex_fx_dict.keys())])

    def compute_recursive_feature(self, initial_vertex_fx_vector, features):
        no_iterations = 0
        max_diff = self.s
        prev_vertex_fx_vectors = initial_vertex_fx_vector

        while no_iterations <= self.MAX_ITERATIONS:
            current_iteration_fx_vectors = {}
            for k in xrange(0, no_iterations):
                current_iteration_fx_vectors[k] = []

            for feature in sorted(features.keys()):
                feature_values = features[feature]
                binned_vertex = self.vertical_bin(feature_values)
                for v in binned_vertex:
                    current_iteration_fx_vectors[v].append(binned_vertex[v])

            current_iteration_fx_vectors = self.feature_dict_to_numpy_array(current_iteration_fx_vectors)

            updated_and_pruned_vertex_fx = self.compare_and_prune_vertex_fx_vectors(prev_vertex_fx_vectors,
                                                                                    current_iteration_fx_vectors,
                                                                                    max_diff)
            updated_and_pruned_fx_size = updated_and_pruned_vertex_fx.shape[1]
            prev_fx_size = prev_vertex_fx_vectors.shape[1]

            if prev_fx_size >= updated_and_pruned_fx_size:
                return prev_vertex_fx_vectors

            prev_vertex_fx_vectors = np.copy(updated_and_pruned_vertex_fx)
            max_diff += 1
            no_iterations += 1

        return prev_vertex_fx_vectors

    def init_log_binned_fx_buckets(self):
        no_vertices = self.get_number_of_vertices()
        max_fx_value = np.ceil(np.log2(no_vertices))  # fixing value of p = 0.5,
        # In our experiments, we found p = 0.5 to be a sensible choice:
        # with each bin containing the bottom half of the remaining nodes.
        log_binned_fx_keys = [value for value in xrange(0, int(max_fx_value))]

        fx_bucket_size = []
        starting_bucket_size = no_vertices

        for idx in np.arange(0.0, max_fx_value):
            starting_bucket_size *= self.p
            fx_bucket_size.append(int(np.ceil(starting_bucket_size)))

        total_slots_in_all_buckets = sum(fx_bucket_size)
        if total_slots_in_all_buckets > no_vertices:
            fx_bucket_size[0] -= (total_slots_in_all_buckets - no_vertices)

        self.refex_log_binned_buckets =  dict(zip(log_binned_fx_keys, fx_bucket_size))

