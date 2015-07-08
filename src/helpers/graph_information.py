import networkx as nx
import argparse


def load_graph(file_name):
    graph = nx.Graph()
    for line in open(file_name):
        line = line.strip()
        line = line.split(',')
        source = int(line[0])
        dest = int(line[1])
        graph.add_edge(source, dest)
    return graph


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='graph information')
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-n', '--name', help='network name', required=True)

    args = argument_parser.parse_args()
    graph_file = args.graph
    name = args.name

    graph = load_graph(graph_file)

    V = graph.number_of_nodes()
    E = graph.number_of_edges()
    CC = nx.number_connected_components(graph)
    CCSG = nx.connected_component_subgraphs(graph)

    d = 0.0
    for n in graph.nodes():
        d += float(len(graph.neighbors(n)))

    subgraphs = []
    for subgraph in CCSG:
        cc_size = subgraph.number_of_nodes()
        diameter = nx.diameter(subgraph)
        subgraphs.append((cc_size, diameter))

    sorted_subgraphs = sorted(subgraphs, key=lambda x: x[0], reverse=True)[0]
    LCC = sorted_subgraphs[0]
    dia = sorted_subgraphs[1]

    print '\\textbf{%s} & %s & %s & %.2f & %s & %s & %s \\\\ \\midrule' % (name, V, E, float(d) / V, CC, LCC, dia)
