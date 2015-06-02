__author__ = 'pratik'

import numpy as np
import argparse
from collections import defaultdict
from sklearn.preprocessing import normalize


def get_node_role_assignment(node_role_matrix_with_ids):
    n, r = node_role_matrix_with_ids.shape
    role_assignments = {}

    for i in xrange(n):
        node_id = node_role_matrix_with_ids[i][0]
        row = node_role_matrix_with_ids[i, 1:]
        reversed_sorted_indices = row.argsort()[-2:][::-1]

        primary_role = reversed_sorted_indices[0]
        secondary_role = reversed_sorted_indices[1]

        if node_role_matrix_with_ids[i][primary_role] <= 0.0:
            primary_role = -1
        if node_role_matrix_with_ids[i][secondary_role] <= 0.0:
            secondary_role = -1

        role_assignments[node_id] = (primary_role, secondary_role)
    return role_assignments


def get_role_blocks(node_role_assignments):
    primary_role_blocks = defaultdict(list)
    secondary_role_blocks = defaultdict(list)
    for node in node_role_assignments.keys():
        primary_role_blocks[node_role_assignments[node][0]].append(node)
        secondary_role_blocks[node_role_assignments[node][1]].append(node)
    return primary_role_blocks, secondary_role_blocks


def get_diff(at, atdt, bt, btdt):
    diff_1 = at - bt
    diff_2 = atdt - btdt
    return np.abs(diff_1 - diff_2)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='node co-evolution')
    argument_parser.add_argument('-m1', '--measurements-one', help='measurements for graph G1', required=True)
    argument_parser.add_argument('-m2', '--measurements-two', help='measurements for graph G2', required=True)
    argument_parser.add_argument('-nr1', '--node-role-one', help='node role file G1', required=True)
    argument_parser.add_argument('-id1', '--node-id-file', help='node id file G1', required=True)
    argument_parser.add_argument('-of', '--output-file', help='output file', required=True)

    args = argument_parser.parse_args()

    m1_file = args.measurements_one
    m2_file = args.measurements_two

    nr1_file = args.node_role_one
    id1_file = args.node_id_file
    out_file = args.output_file

    node_roles = np.loadtxt(nr1_file)
    node_roles[node_roles <= 0.0] = 0.0

    node_ids = np.loadtxt(id1_file)

    n, m = node_roles.shape
    node_roles_with_id = np.zeros((n, m+1))

    node_roles_with_id[:, 0] = node_ids
    node_roles_with_id[:, 1:] = node_roles

    measurements_one = np.loadtxt(m1_file, delimiter=',')
    measurements_two = np.loadtxt(m2_file, delimiter=',')

    node_ids_one = measurements_one[:, 0]
    node_ids_two = measurements_two[:, 0]

    norm_measurments_one = normalize(measurements_one[:, 1:], norm='l2', axis=0)
    norm_measurments_two = normalize(measurements_two[:, 1:], norm='l2', axis=0)

    n, m = measurements_one.shape
    normed_full_measure_matrix_one = np.zeros((n, m))
    normed_full_measure_matrix_one[:, 0] = node_ids_one
    normed_full_measure_matrix_one[:, 1:] = norm_measurments_one

    n, m = measurements_two.shape
    normed_full_measure_matrix_two = np.zeros((n, m))
    normed_full_measure_matrix_two[:, 0] = node_ids_two
    normed_full_measure_matrix_two[:, 1:] = norm_measurments_two

    roles_assignments = get_node_role_assignment(node_roles_with_id)

    primary_blocks, secondary_blocks = get_role_blocks(roles_assignments)
    # secondary_blocks, primary_blocks = get_role_blocks(roles_assignments)

    primary_node_pairs = defaultdict(list)

    total_pairs = 0
    for key in primary_blocks.keys():
        if key == -1:
            continue
        block = primary_blocks[key]
        block_size = len(block)
        if block_size > 1:
            for i in xrange(0, block_size - 1):
                for j in xrange(i + 1, block_size):
                    primary_node_pairs[block[i]].append(block[j])
                    total_pairs += 1

    print '*'*50
    print 'Total Pairs: ', total_pairs

    labels = {'between': 1, 'close': 2, 'bcc': 3, 'ed0': 4, 'ed1': 5,
              'ew0': 6, 'ew1': 7, 'deg': 8, 'wdeg': 9, 'clusc': 10}

    c = 0

    bet_diff = []
    closeness_diff = []
    bcc_diff = []
    ew0_diff = []
    ed0_diff = []
    ew1_diff = []
    ed1_diff = []
    clusc_diff = []
    degree_diff = []
    wdegree_diff = []

    for a in primary_node_pairs.keys():
        a_t = []
        a_tdt = []

        between_at = normed_full_measure_matrix_one[a][labels['between']]
        between_atdt = normed_full_measure_matrix_two[a][labels['between']]

        bcc_at = normed_full_measure_matrix_one[a][labels['bcc']]
        bcc_atdt = normed_full_measure_matrix_two[a][labels['bcc']]

        ew0_at = normed_full_measure_matrix_one[a][labels['ew0']]
        ew0_atdt = normed_full_measure_matrix_two[a][labels['ew0']]

        edeg0_at = normed_full_measure_matrix_one[a][labels['ed0']]
        edeg0_atdt = normed_full_measure_matrix_two[a][labels['ed0']]

        ew1_at = normed_full_measure_matrix_one[a][labels['ew1']]
        ew1_atdt = normed_full_measure_matrix_two[a][labels['ew1']]

        edeg1_at = normed_full_measure_matrix_one[a][labels['ed1']]
        edeg1_atdt = normed_full_measure_matrix_two[a][labels['ed1']]

        clus_at = normed_full_measure_matrix_one[a][labels['clusc']]
        clus_atdt = normed_full_measure_matrix_two[a][labels['clusc']]

        close_at = normed_full_measure_matrix_one[a][labels['close']]
        close_atdt = normed_full_measure_matrix_two[a][labels['close']]

        deg_at = normed_full_measure_matrix_one[a][labels['deg']]
        deg_atdt = normed_full_measure_matrix_two[a][labels['deg']]

        wdeg_at = normed_full_measure_matrix_one[a][labels['wdeg']]
        wdeg_atdt = normed_full_measure_matrix_two[a][labels['wdeg']]

        for b in primary_node_pairs[a]:
            b_t = []
            b_tdt = []

            between_bt = normed_full_measure_matrix_one[b][labels['between']]
            between_btdt = normed_full_measure_matrix_two[b][labels['between']]

            bcc_bt = normed_full_measure_matrix_one[b][labels['bcc']]
            bcc_btdt = normed_full_measure_matrix_two[b][labels['bcc']]

            close_bt = normed_full_measure_matrix_one[b][labels['close']]
            close_btdt = normed_full_measure_matrix_two[b][labels['close']]

            ew0_bt = normed_full_measure_matrix_one[b][labels['ew0']]
            ew0_btdt = normed_full_measure_matrix_two[b][labels['ew0']]

            edeg0_bt = normed_full_measure_matrix_one[b][labels['ed0']]
            edeg0_btdt = normed_full_measure_matrix_two[b][labels['ed0']]

            ew1_bt = normed_full_measure_matrix_one[b][labels['ew1']]
            ew1_btdt = normed_full_measure_matrix_two[b][labels['ew1']]

            edeg1_bt = normed_full_measure_matrix_one[b][labels['ed1']]
            edeg1_btdt = normed_full_measure_matrix_two[b][labels['ed1']]

            clus_bt =  normed_full_measure_matrix_one[b][labels['clusc']]
            clus_btdt = normed_full_measure_matrix_two[b][labels['clusc']]

            deg_bt = normed_full_measure_matrix_one[b][labels['deg']]
            deg_btdt = normed_full_measure_matrix_two[b][labels['deg']]

            wdeg_bt = normed_full_measure_matrix_one[b][labels['wdeg']]
            wdeg_btdt = normed_full_measure_matrix_two[b][labels['wdeg']]

            bet_diff.append(get_diff(between_at, between_atdt, between_bt, between_btdt))
            closeness_diff.append(get_diff(close_at, close_atdt, close_bt, close_btdt))
            bcc_diff.append(get_diff(bcc_at, bcc_atdt, bcc_bt, bcc_btdt))

            ew0_diff.append(get_diff(ew0_at, ew0_atdt, ew0_bt, ew0_btdt))
            ew1_diff.append(get_diff(ew1_at, ew1_atdt, ew1_bt, ew1_btdt))
            ed0_diff.append(get_diff(edeg0_at, edeg0_atdt, edeg0_bt, edeg0_btdt))
            ed1_diff.append(get_diff(edeg1_at, edeg1_atdt, edeg1_bt, edeg1_btdt))

            clusc_diff.append(get_diff(clus_at, clus_atdt, clus_bt, clus_btdt))
            degree_diff.append(get_diff(deg_at, deg_atdt, deg_bt, deg_btdt))
            wdegree_diff.append(get_diff(wdeg_at, wdeg_atdt, wdeg_bt, wdeg_btdt))

    final_diff = [bet_diff, closeness_diff, bcc_diff, ed0_diff, ed1_diff, ew0_diff,
                  ew1_diff, degree_diff, wdegree_diff, clusc_diff]
    np.savetxt(out_file, np.asarray(final_diff))
