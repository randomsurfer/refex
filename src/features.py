import networkx as nx
import os
import numpy as np
from numpy.lib.recfunctions import merge_arrays


class Features:
    """
    The features class. Placeholder for the graph, egonet and node features.
    """
    def __init__(self):
        """
        initializes an empty networkx directed graph
        :return:
        """
        self.graph = nx.DiGraph()
        self.no_of_vertices = 0
        self.p = 0.5  # fraction of nodes placed in log bin
        self.s = 0  # feature similarity threshold
        self.TOLERANCE = 0.0001
        self.MAX_ITERATIONS = 100
        self.refex_log_binned_buckets = []
        self.vertex_egonets = {}

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
        node's egonet members, supports level 0 and level 1 egonet
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

    def compute_base_egonet_primitive_features(self, vertex, attrs, level_id='0'):
        """
            * Counts several egonet properties. In particular:
            *
            * wn - Within Node - # of nodes in egonet
            * weu - Within Edge Unique - # of unique edges with both ends in egonet
            * wet - Within Edge Total - total # of internal edges
            * xesu - eXternal Edge Source Unique - # of unique edges exiting egonet
            * xest - eXternal Edge Source Total - total # of edges exiting egonet
            * xedu - eXternal Edge Destination Unique - # of unique edges entering egonet
            * xedt - eXternal Edge Destination Total - total # of edges entering egonet
            *
            * and three counts per attribute,
            *

            for(String base : new String[]{"xe", "xes", "xed"}) {

            * wea-ATTRNAME  - Within Edge Attribute - sum of attribute for internal edges
            * xea-ATTRNAME  - sum of xeda and xesa
            * xesa-ATTRNAME - eXternal Edge Source Attribute - sum of attr for exiting edges
            * xeda-ATTRNAME - eXternal Edge Destination Attribute - sum of attr for entering edges
            *
        :return: side effecting code, adds the features in the networkx DiGraph dict
        """
        self.graph.node[vertex]['wn'+level_id] = 0.0
        self.graph.node[vertex]['weu'+level_id] = 0.0
        self.graph.node[vertex]['wet'+level_id] = 0.0
        self.graph.node[vertex]['wea-'+level_id] = 0.0
        self.graph.node[vertex]['xedu'+level_id] = 0.0
        self.graph.node[vertex]['xedt'+level_id] = 0.0
        self.graph.node[vertex]['xesu'+level_id] = 0.0
        self.graph.node[vertex]['xest'+level_id] = 0.0
        self.graph.node[vertex]['xeu'+level_id] = 0.0
        self.graph.node[vertex]['xet'+level_id] = 0.0
        for attr in attrs:
            self.graph.node[vertex]['wea-'+attr+level_id] = 0.0
            self.graph.node[vertex]['xesa-'+attr+level_id] = 0.0
            self.graph.node[vertex]['xeda-'+attr+level_id] = 0.0
            self.graph.node[vertex]['xea-'+attr+level_id] = 0.0

        if level_id == '0':
            egonet = self.vertex_egonets[vertex][0]
        else:
            egonet = self.vertex_egonets[vertex][1]

        for n1 in egonet:
            in_neighbours = self.graph.predecessors(n1)
            out_neighbours = self.graph.successors(n1)

            self.graph.node[vertex]['wn'+level_id] += 1.0

            for n2 in in_neighbours:
                if n2 in egonet:
                    self.graph.node[vertex]['weu'+level_id] += 1.0
                    self.graph.node[vertex]['wet'+level_id] += len(self.graph.predecessors(n2))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['wea-'+attr+level_id] += self.graph[n2][n1]['weight']
                else:
                    self.graph.node[vertex]['xedu'+level_id] += 1.0
                    self.graph.node[vertex]['xedt'+level_id] += len(self.graph.predecessors(n2))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['xeda-'+attr+level_id] += self.graph[n2][n1]['weight']

            for n2 in out_neighbours:
                if n2 not in egonet:
                    self.graph.node[vertex]['xesu'+level_id] += 1.0
                    self.graph.node[vertex]['xest'+level_id] += len(self.graph.successors(n2))
                    for attr in attrs:
                        if attr == 'wgt':
                            self.graph.node[vertex]['xesa-'+attr+level_id] += self.graph[n1][n2]['weight']
                else:
                    # weu, wet and wea have already been counted as in_neighbours in some egonet
                    # do nothing
                    continue

        self.graph.node[vertex]['xeu'+level_id] = self.graph.node[vertex]['xesu'+level_id] + self.graph.node[vertex]['xedu'+level_id]
        self.graph.node[vertex]['xet'+level_id] = self.graph.node[vertex]['xest'+level_id] + self.graph.node[vertex]['xedt'+level_id]

        for attr in attrs:
            self.graph.node[vertex]['xea-'+attr+level_id] = self.graph.node[vertex]['xesa-'+attr+level_id] + self.graph.node[vertex]['xeda-'+attr+level_id]

    def compute_rider_egonet_primitive_features(self, rider_dir, attrs=['wgt']):
        for file_name in os.listdir(rider_dir):
            for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
                line = line.strip().split()
                block = set([int(n) for n in line])
                fx_name_base = file_name + '_' + str(i)
                for vertex in sorted(self.vertex_egonets.keys()):
                    vertex_lvl_0_egonet = self.vertex_egonets[vertex][0]
                    vertex_lvl_1_egonet = self.vertex_egonets[vertex][1]

                    in_neighbours = self.graph.predecessors(vertex)
                    out_neighbours = self.graph.successors(vertex)

                    in_connections_to_block = set(in_neighbours) & block
                    in_connections_to_block_size = len(in_connections_to_block)
                    out_connections_to_block = set(out_neighbours) & block
                    out_connections_to_block_size = len(out_connections_to_block)

                    ## Local Rider Features
                    self.graph.node[vertex]['wd_'+fx_name_base] = float(in_connections_to_block_size)  # destination
                    self.graph.node[vertex]['ws_'+fx_name_base] = float(out_connections_to_block_size)  # source

                    for attr in attrs:
                        self.graph.node[vertex]['wda-'+attr+'_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['wsa-'+attr+'_'+fx_name_base] = 0.0

                    if in_connections_to_block_size > 0:
                        for attr in attrs:
                            for connection in in_connections_to_block:
                                if attr == 'wgt':
                                    self.graph.node[vertex]['wda-'+attr+'_'+fx_name_base] \
                                        += self.graph[connection][vertex]['weight']

                    if out_connections_to_block_size > 0:
                        for attr in attrs:
                            for connection in out_connections_to_block:
                                if attr == 'wgt':
                                    self.graph.node[vertex]['wsa-'+attr+'_'+fx_name_base] \
                                        += self.graph[vertex][connection]['weight']

                    ## Egonet Rider Features
                    self.graph.node[vertex]['wes0-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wes1-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wed0-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wed1-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xes0-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xes1-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xed0-'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xed1-'+fx_name_base] = 0.0
                    for attr in attrs:
                        self.graph.node[vertex]['wesa-'+attr+'0-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xesa-'+attr+'0-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['weda-'+attr+'0-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xeda-'+attr+'0-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['wesa-'+attr+'1-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xesa-'+attr+'1-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['weda-'+attr+'1-'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xeda-'+attr+'1-'+fx_name_base] = 0.0

                    # Level 0 Egonet
                    for n1 in vertex_lvl_0_egonet:
                        in_neighbours = self.graph.predecessors(n1)
                        out_neighbours = self.graph.successors(n1)

                        for n2 in in_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['wed0-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['weda-'+attr+'0-'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']
                            else:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['xed0-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xeda-'+attr+'0-'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']

                        for n2 in out_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['wes0-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['wesa-'+attr+'0-'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']
                            else:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['xes0-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xesa-'+attr+'0-'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']

                    # Level 1 Egonet
                    for n1 in vertex_lvl_1_egonet:
                        in_neighbours = self.graph.predecessors(n1)
                        out_neighbours = self.graph.successors(n1)

                        for n2 in in_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['wed1-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['weda-'+attr+'1-'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']
                            else:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['xed1-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xeda-'+attr+'1-'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']

                        for n2 in out_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['wes1-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['wesa-'+attr+'1-'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']
                            else:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['xes1-'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xesa-'+attr+'1-'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']

    def compute_primitive_features(self, rider_fx=False, rider_dir='INVALID_PATH'):
        # computes the primitive local features
        # computes the rider based features if rider_fx is True
        # updates in place the primitive feature values with their log binned values

        for n1 in self.graph.nodes():
            if n1 not in self.vertex_egonets:
                vertex_lvl_0_egonet = self.get_egonet_members(n1)
                vertex_lvl_1_egonet = self.get_egonet_members(n1, level=1)
                self.vertex_egonets[n1] = [vertex_lvl_0_egonet, vertex_lvl_1_egonet]

            self.compute_base_egonet_primitive_features(n1, ['wgt'], level_id='0')
            self.compute_base_egonet_primitive_features(n1, ['wgt'], level_id='1')

        if rider_fx:
            if not os.path.exists(rider_dir):
                raise Exception("RIDeR output directory is empty!")
            self.compute_rider_egonet_primitive_features(rider_dir, ['wgt'])

        self.no_of_vertices = self.graph.number_of_nodes()
        self.init_log_binned_fx_buckets()

        fx_names = [attr for attr in sorted(self.graph.node[self.graph.nodes()[0]])]
        self.compute_log_binned_features(fx_names)

    def compute_recursive_features(self, prev_fx_matrix, iter_no, max_dist):
        # takes the prev feature matrix and the iteration number and max_dist
        # returns the new feature matrix after pruning similar features based on the similarity max dist

        new_fx_names = []

        for vertex in self.graph.nodes():
            new_fx_names = self.compute_recursive_egonet_features(vertex, iter_no)

        # compute and replace the new feature values with their log binned values
        self.compute_log_binned_features(new_fx_names)
        # create the feature matrix of all the features in the current graph structure
        new_fx_matrix = self.create_feature_matrix(new_fx_names)

        # return the pruned fx matrix after adding and comparing the new recursive features for similarity
        return self.compare_and_prune_vertex_fx_vectors(prev_fx_matrix, new_fx_matrix, max_dist)

    def compute_recursive_egonet_features(self, vertex, iter_no):
        # computes the sum and mean features of all the features which exist in the current graph structure
        # updates the new features in place and
        # returns the string list of the new feature names

        sum_fx = '-s'
        mean_fx = '-m'
        vertex_lvl_0_egonet = self.vertex_egonets[vertex][0]
        vertex_lvl_0_egonet_size = float(len(vertex_lvl_0_egonet))
        vertex_lvl_1_egonet = self.vertex_egonets[vertex][1]
        vertex_lvl_1_egonet_size = float(len(vertex_lvl_1_egonet))

        fx_list = [fx_name for fx_name in sorted(self.graph.node[vertex].keys())]
        new_fx_list = []

        level_id = '0'
        for fx_name in fx_list:
            fx_value = 0.0
            for node in vertex_lvl_0_egonet:
                fx_value += self.graph.node[node][fx_name]

            self.graph.node[vertex][fx_name + str(iter_no) + sum_fx + level_id] = fx_value
            self.graph.node[vertex][fx_name + str(iter_no) + mean_fx + level_id] = fx_value / vertex_lvl_0_egonet_size

            new_fx_list.append(fx_name + str(iter_no) + sum_fx + level_id)
            new_fx_list.append(fx_name + str(iter_no) + mean_fx + level_id)

        level_id = '1'
        for fx_name in fx_list:
            fx_value = 0.0
            for node in vertex_lvl_1_egonet:
                fx_value += self.graph.node[node][fx_name]

            self.graph.node[vertex][fx_name + str(iter_no) + sum_fx + level_id] = fx_value
            self.graph.node[vertex][fx_name + str(iter_no) + mean_fx + level_id] = fx_value / vertex_lvl_1_egonet_size

            new_fx_list.append(fx_name + str(iter_no) + sum_fx + level_id)
            new_fx_list.append(fx_name + str(iter_no) + mean_fx + level_id)

        return new_fx_list

    def get_sorted_feature_values(self, feature_values):
        # takes list of tuple(vertex_id, feature value)
        # returns the sorted list using feature value as the comparison key and the length of the sorted list
        sorted_fx_values = sorted(feature_values, key=lambda x: x[1])
        return sorted_fx_values, len(sorted_fx_values)

    def init_log_binned_fx_buckets(self):
        # initializes the refex_log_binned_buckets with the vertical log bin values,
        # computed based on p and the number of vertices in the graph

        max_fx_value = np.ceil(np.log2(self.no_of_vertices) + self.TOLERANCE)  # fixing value of p = 0.5,
        # In our experiments, we found p = 0.5 to be a sensible choice:
        # with each bin containing the bottom half of the remaining nodes.
        log_binned_fx_keys = [value for value in xrange(0, int(max_fx_value))]

        fx_bucket_size = []
        starting_bucket_size = self.no_of_vertices

        for idx in np.arange(0.0, max_fx_value):
            starting_bucket_size *= self.p
            fx_bucket_size.append(int(np.ceil(starting_bucket_size)))

        total_slots_in_all_buckets = sum(fx_bucket_size)
        if total_slots_in_all_buckets > self.no_of_vertices:
            fx_bucket_size[0] -= (total_slots_in_all_buckets - self.no_of_vertices)

        log_binned_buckets_dict = dict(zip(log_binned_fx_keys, fx_bucket_size))

        for binned_value in sorted(log_binned_buckets_dict.keys()):
            for count in xrange(0, log_binned_buckets_dict[binned_value]):
                self.refex_log_binned_buckets.append(binned_value)

        if len(self.refex_log_binned_buckets) != self.no_of_vertices:
            raise Exception("Vertical binned bucket size not equal to the number of vertices!")

    def vertical_bin(self, feature):
        # input a list of tuple(vertex_id, feature value)
        # returns a dict with key -> vertex id, value -> vertical log binned value

        vertical_binned_vertex = {}
        count_of_vertices_with_log_binned_fx_value_assigned = 0
        fx_value_of_last_vertex_assigned_to_bin = -1
        previous_binned_value = 0

        sorted_fx_values, sorted_fx_size = self.get_sorted_feature_values(feature)

        for vertex, value in sorted_fx_values:
            current_binned_value = self.refex_log_binned_buckets[count_of_vertices_with_log_binned_fx_value_assigned]

            # If there are ties, it may be necessary to include more than p|V| nodes
            if current_binned_value != previous_binned_value and value == fx_value_of_last_vertex_assigned_to_bin:
                vertical_binned_vertex[vertex] = previous_binned_value
            else:
                vertical_binned_vertex[vertex] = current_binned_value
                previous_binned_value = current_binned_value

            count_of_vertices_with_log_binned_fx_value_assigned += 1
            fx_value_of_last_vertex_assigned_to_bin = value

        return vertical_binned_vertex

    def compute_log_binned_features(self, fx_list):
        # input string list of feature names, which have been assigned regular feature value (non-log value)
        # computes the vertical binned values for all features in the fx_list and replaces them
        # in place with their log binned values

        graph_nodes = sorted(self.graph.nodes())
        for feature in fx_list:
            node_fx_values = []
            for n in graph_nodes:
                node_fx_values.append(tuple([n, self.graph.node[n][feature]]))

            vertical_binned_vertices = self.vertical_bin(node_fx_values)
            for vertex in vertical_binned_vertices.keys():
                binned_value = vertical_binned_vertices[vertex]
                self.graph.node[vertex][feature] = binned_value

    def create_initial_feature_matrix(self):
        # Returns a numpy structured node-feature matrix for all the features assigned to nodes in graph
        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        dtype = []
        for fx_name in sorted(self.graph.node[graph_nodes[0]].keys()):
            fx_names.append(fx_name)
            dtype.append(tuple([fx_name, '>f4']))

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix, dtype=dtype)  # return a structured numpy array

    def create_feature_matrix(self, fx_list):
        # Returns a numpy structured node-feature matrix for the features in the list
        graph_nodes = sorted(self.graph.nodes())
        fx_names = []
        dtype = []
        for fx_name in sorted(fx_list):
            fx_names.append(fx_name)
            dtype.append(tuple([fx_name, '>f4']))

        fx_matrix = []
        for node in graph_nodes:
            feature_row = []
            for fx_name in fx_names:
                feature_row.append(self.graph.node[node][fx_name])
            fx_matrix.append(tuple(feature_row))
        return np.array(fx_matrix, dtype=dtype)  # return a structured numpy array

    def fx_column_comparator(self, col_1, col_2, max_diff):
        # input two columns -> i.e. two node features and the max_dist
        # returns True/False if the two features agree within the max_dist criteria

        diff = float(max_diff) - abs(col_1 - col_2) + self.TOLERANCE
        return (diff >= self.TOLERANCE).all()

    def compare_and_prune_vertex_fx_vectors(self, prev_feature_vector, new_feature_vector, max_diff):
        # input: prev iteration node-feature matrix and current iteration node-feature matrix (as structured
        # numpy array) and max_dist. Creates a feature graph based on the max_dist criteria.
        # We use a (potentially different) logic from the original refex idea.
        # We compare the prev iteration features with the new iteration features using
        # a max_dist criteria of 0. And those amongst the current/new iteration criteria based on the max_diff/max_dist
        # given by the current iteration value of the max_dist. We do this to avoid pruning of old feature due to the
        # incrementally increasing value of max_dist with every iteration of recursive feature generation.
        # This aspect wasn't very clear from the ReFeX paper!
        # Returns a numpy structured array of the final features

        fx_graph = nx.Graph()

        if prev_feature_vector is not None:
            col_prev = list(prev_feature_vector.dtype.names)
        else:
            col_prev = []

        col_new = list(new_feature_vector.dtype.names)

        # compare new features with previous features
        if len(col_prev) > 0:  # compare for something which is not a first iteration
            for col_i in col_prev:
                for col_j in col_new:
                    if self.fx_column_comparator(prev_feature_vector[col_i], new_feature_vector[col_j], 0.0):
                        fx_graph.add_edge(col_i, col_j)

        # compare new features with new
        for col_i in col_new:
            for col_j in col_new:
                if col_i < col_j:  # to avoid redundant comparisons
                    if self.fx_column_comparator(new_feature_vector[col_i], new_feature_vector[col_j], max_diff):
                        fx_graph.add_edge(col_i, col_j)

        connected_fx = nx.connected_components(fx_graph)
        for cc in connected_fx:
            sorted_cc = sorted(cc)[1:]
            for fx in sorted_cc:
                if fx in col_prev:
                    col_prev.remove(fx)
                else:
                    col_new.remove(fx)

        # TODO: Might need a refactoring here
        # prev_fx_vector is None => First iteration, prune correlated features
        if prev_feature_vector is None:
            return new_feature_vector[col_new]

        # features in both prev and new same for max_dist, hence we send back prev
        if len(col_prev) == 0:
            return prev_feature_vector

        if len(col_new) == 0:
            return prev_feature_vector

        final_prev_vector = prev_feature_vector[col_prev]
        final_new_vector = new_feature_vector[col_new]

        # return the merged fx matrix
        return merge_arrays((final_prev_vector, final_new_vector), flatten=True)
