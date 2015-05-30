import argparse
import numpy as np
from collections import defaultdict


def read_roles_ssn(ground_truth_file):
    roles = defaultdict(list)
    for line in open(ground_truth_file):
        line = line.strip().split('\t')
        node_id = int(line[0])
        rols = line[1].split(',')
        for rol in rols:
            rol = rol.strip()
            roles[node_id].append(rol)
    return roles


def read_roles_imdb(ground_truth_file):
    roles = defaultdict(list)
    ground_truth = np.loadtxt(ground_truth_file)
    # node_id is the 0th Column, subsequent cols are the weights in that role
    for node_id in ground_truth[:, 0]:
        for idx, value in enumerate(ground_truth[node_id, 1:]):
            if value > 0.0:
                roles[node_id].append(idx)
        roles[node_id] = set(roles[node_id])
    return roles


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='run ground truth evaluation')
    argument_parser.add_argument('-nr', '--node-role', help='node-role matrix file', required=True)
    argument_parser.add_argument('-i', '--ids', help='node ids file', required=True)
    argument_parser.add_argument('-g', '--ground-truth', help='ground-truth data file', required=True)
    argument_parser.add_argument('-L', '--num-gt-roles', help='number ground-truth roles/labels', required=True)

    args = argument_parser.parse_args()

    nr_file = args.node_role
    ids_file = args.ids
    gt_file = args.ground_truth
    L = float(args.num_gt_roles)

    node_role = np.loadtxt(nr_file)
    ids = np.loadtxt(ids_file)
    ground_node_roles = read_roles_imdb(gt_file)

    node_role[node_role <= 0.0] = 0.0
    id_seq = [i for i in ids]

    role_means = {}
    for i in xrange(node_role.shape[1]):
        role_means[i] = node_role[:, i].mean()

    predicted_roles = defaultdict(list)

    for i in xrange(len(id_seq)):
        node_id = id_seq[i]
        role_value = []
        for role in role_means.keys():
            if node_role[i, role] >= role_means[role]:
                role_value.append((role, node_role[i, role]))
        sorted_roles = sorted(role_value, key=lambda x: x[1], reverse=True)
        if len(sorted_roles) > 0:
            role, value = role_value[0]
            predicted_roles[role].append(node_id)

    predicted_node_roles = defaultdict(list)

    for role in predicted_roles.keys():
        role_list = []
        if len(predicted_roles[role]) > 0:
            for node_id in predicted_roles[role]:
                role_list.extend(ground_node_roles[node_id])
            role_list = set(role_list)

            for node_id in predicted_roles[role]:
                predicted_node_roles[node_id] = role_list

    # compute the multi-label metrics
    hamming_loss = 0.0
    precision = 0.0
    recall = 0.0
    D = 0.0

    for node in predicted_node_roles.keys():
        if node in ground_node_roles and len(ground_node_roles[node]) > 0:
            Y = ground_node_roles[node]
            Z = predicted_node_roles[node]

            hamming_loss += len(Y ^ Z) * 1.0 / L
            precision += len(Y & Z) * 1.0 / len(Z)
            recall += len(Y & Z) * 1.0 / len(Y)

            D += 1.0

    hamming_loss /= D
    precision /= D
    recall /= D

    print 'HammingLoss\t%.2f\tPrecision\t%.2f\tRecall\t%.2f' % (hamming_loss * 100.0, precision * 100.0,
                                                                recall * 100.0)

