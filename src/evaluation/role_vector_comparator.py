__author__ = 'pratik'

import argparse
import numpy as np
import numpy.linalg as la
import networkx as nx


def cosine_similarity(a, b):
    tn = np.inner(a, b)
    td = la.norm(a) * la.norm(b)
    if td != 0.0:
        return tn / td
    else:
        return 0.0

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(prog='Role-Feature Vector Comparator')
    argument_parser.add_argument('-rf', '--role-feature', help='role feature matrix', required=True)
    argument_parser.add_argument('-nr', '--node-role', help='node role matrix', required=True)

    args = argument_parser.parse_args()

    rf_file = args.role_feature
    nr_file = args.node_role
    role_feature = np.loadtxt(rf_file)
    node_role = np.loadtxt(nr_file)
    num_roles, num_features = role_feature.shape
    num_nodes, x = node_role.shape

    g = nx.Graph()
    before = num_roles

    for i in xrange(num_roles-1):
        for j in xrange(i+1, num_roles):
            a = role_feature[i, :]
            b = role_feature[j, :]
            similarity = cosine_similarity(a, b)
            if similarity >= 0.85:
                g.add_edge(i, j)

    connected_components = nx.connected_components(g)
    nx.connected_component_subgraphs

    for cc in connected_components:
        roles_to_delete = [n for n in cc]
        if len(roles_to_delete) > 1:
            representative = roles_to_delete[0]
            roles_to_delete = roles_to_delete[1:]
            print representative, roles_to_delete
            role_feature = np.delete(role_feature, roles_to_delete, axis=0)
            node_role = np.delete(node_role, roles_to_delete, axis=1)

    after = role_feature.shape[0]
    print '*'*20
    print 'Redundant Roles: %s, Final: %s' % ((before-after), after)
    print '*'*20
    if before != after:
        np.savetxt(nr_file, X=node_role)
        np.savetxt(rf_file, X=role_feature)