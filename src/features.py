import networkx as nx
import os

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

    def compute_base_egonet_primitive_features(self, vertex, egonet, attrs, level_id='0'):
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

    def compute_rider_egonet_primitive_features(self, rider_dir, vertex_egonets, attrs=['wgt']):
        for file_name in os.listdir(rider_dir):
            for i, line in enumerate(open(os.path.join(rider_dir, file_name))):
                line = line.strip().split()
                block = set([int(n) for n in line])
                fx_name_base = file_name + '_' + str(i)
                for vertex in sorted(vertex_egonets.keys()):
                    vertex_lvl_0_egonet = vertex_egonets[vertex][0]
                    vertex_lvl_1_egonet = vertex_egonets[vertex][1]

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
                    self.graph.node[vertex]['wes0_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wes1_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wed0_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['wed1_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xes0_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xes1_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xed0_'+fx_name_base] = 0.0
                    self.graph.node[vertex]['xed1_'+fx_name_base] = 0.0
                    for attr in attrs:
                        self.graph.node[vertex]['wesa-'+attr+'0_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xesa-'+attr+'0_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['weda-'+attr+'0_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xeda-'+attr+'0_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['wesa-'+attr+'1_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xesa-'+attr+'1_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['weda-'+attr+'1_'+fx_name_base] = 0.0
                        self.graph.node[vertex]['xeda-'+attr+'1_'+fx_name_base] = 0.0

                    # Level 0 Egonet
                    for n1 in vertex_lvl_0_egonet:
                        in_neighbours = self.graph.predecessors(n1)
                        out_neighbours = self.graph.successors(n1)

                        for n2 in in_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['wed0_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['weda-'+attr+'0_'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']
                            else:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['xed0_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xeda-'+attr+'0_'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']

                        for n2 in out_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['wes0_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['wesa-'+attr+'0_'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']
                            else:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['xes0_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xesa-'+attr+'0_'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']

                    # Level 1 Egonet
                    for n1 in vertex_lvl_1_egonet:
                        in_neighbours = self.graph.predecessors(n1)
                        out_neighbours = self.graph.successors(n1)

                        for n2 in in_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['wed1_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['weda-'+attr+'1_'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']
                            else:
                                common_connections = set(self.graph.predecessors(n2)) & block
                                self.graph.node[vertex]['xed1_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xeda-'+attr+'1_'+fx_name_base] \
                                                    += self.graph[connection][n2]['weight']

                        for n2 in out_neighbours:
                            if n2 in vertex_lvl_0_egonet:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['wes1_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['wesa-'+attr+'1_'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']
                            else:
                                common_connections = set(self.graph.successors(n2)) & block
                                self.graph.node[vertex]['xes1_'+fx_name_base] += len(common_connections)
                                if len(common_connections) > 0:
                                    for attr in attrs:
                                        if attr == 'wgt':
                                            for connection in common_connections:
                                                self.graph.node[vertex]['xesa-'+attr+'1_'+fx_name_base] \
                                                    += self.graph[n2][connection]['weight']

    def compute_primitive_features(self, rider_fx=False, rider_dir='INVALID_PATH'):
        node_egonets = {}
        for n1 in self.graph.nodes():
            vertex_lvl_0_egonet = self.get_egonet_members(n1)
            vertex_lvl_1_egonet = self.get_egonet_members(n1, level=1)
            if rider_fx:
                node_egonets[n1] = [vertex_lvl_0_egonet, vertex_lvl_1_egonet]

            self.compute_base_egonet_primitive_features(n1, vertex_lvl_0_egonet, ['wgt'], level_id='0')
            self.compute_base_egonet_primitive_features(n1, vertex_lvl_1_egonet, ['wgt'], level_id='1')

        if rider_fx:
            if not os.path.exists(rider_dir):
                raise Exception("RIDeR output directory is empty!")
            self.compute_rider_egonet_primitive_features(rider_dir, node_egonets, ['wgt'])

    def compute_iterative_features(self):
        pass