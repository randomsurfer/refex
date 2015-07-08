import networkx as nx
import numpy as np


def forest_fire_sampling(graph, sampling_fraction, geometric_dist_param=0.7):
    sampled_graph = nx.DiGraph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)
    shuffled_graph_nodes = np.random.shuffle([n for n in graph.nodes()])
    already_visited = dict()

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        burn_seed_node = shuffled_graph_nodes[0]
        shuffled_graph_nodes = shuffled_graph_nodes[1:]

        if burn_seed_node in already_visited:
            continue

        already_visited[burn_seed_node] = 1

        num_edges_to_burn = np.random.geometric(p=geometric_dist_param)

        neighbors_to_burn = np.random.shuffle(graph.successors(burn_seed_node)[:num_edges_to_burn])
        burn_queue = []

        for n in neighbors_to_burn:
            sampled_graph.add_edge(burn_seed_node, n)
            burn_queue.append(n)

        while len(burn_queue) > 0:
            burn_seed_node = burn_queue[0]
            burn_queue = burn_queue[1:]

            if burn_seed_node in already_visited:
                continue

            already_visited[burn_seed_node] = 1

            num_edges_to_burn = np.random.geometric(p=geometric_dist_param)

            neighbors_to_burn = np.random.shuffle(graph.successors(burn_seed_node)[:num_edges_to_burn])

            for n in neighbors_to_burn:
                sampled_graph.add_edge(burn_seed_node, n)
                burn_queue.append(n)

    return sampled_graph


def induced_edge_sampling(graph, sampling_fraction):
    sampled_graph = nx.DiGraph()

    max_sampled_nodes = int(graph.number_of_nodes() * sampling_fraction)

    shuffled_graph_edges = np.random.shuffle(graph.edges())

    while sampled_graph.number_of_nodes() <= max_sampled_nodes:
        u, v = shuffled_graph_edges[0]
        shuffled_graph_edges = shuffled_graph_edges[1:]

        sampled_graph.add_edge(u, v)

    for u, v in graph.edges():
        if sampled_graph.has_node(u) and sampled_graph.has_node(v) and (not sampled_graph.has_edge(u, v)):
            sampled_graph.add_edge(u, v)

    return sampled_graph

if __name__ == '__main__':
    pass
