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


def compute_zero_binned_percentage(final_diffs):
    count = 0
    for val in final_diffs:
        if val == 0.0:
            count += 1
    return count / float(len(final_diffs)) * 100.0


def is_role_same(role_assignments, node_one, node_two):
    return role_assignments[node_one][0] == role_assignments[node_two][0]


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='node co-evolution')
    argument_parser.add_argument('-m1', '--measurements-one', help='measurements for graph G1', required=True)
    argument_parser.add_argument('-m2', '--measurements-two', help='measurements for graph G2', required=True)
    argument_parser.add_argument('-nr1', '--node-role-one', help='node role file G1', required=True)
    argument_parser.add_argument('-nr2', '--node-role-two', help='node role file G2', required=True)
    argument_parser.add_argument('-id1', '--node-id-file-one', help='node id file G1', required=True)
    argument_parser.add_argument('-id2', '--node-id-file-two', help='node id file G2', required=True)
    argument_parser.add_argument('-of', '--output-file', help='output file', required=True)

    args = argument_parser.parse_args()

    m1_file = args.measurements_one
    m2_file = args.measurements_two

    nr1_file = args.node_role_one
    nr2_file = args.node_role_two
    id1_file = args.node_id_file_one
    id2_file = args.node_id_file_two
    out_file = args.output_file

    node_roles = np.loadtxt(nr1_file)
    node_roles_two = np.loadtxt(nr2_file)

    node_roles[node_roles <= 0.0] = 0.0
    node_roles_two[node_roles_two <= 0.0] = 0.0

    node_ids = np.loadtxt(id1_file)
    node_ids2 = np.loadtxt(id2_file)

    try:
        n, m = node_roles.shape
    except ValueError:
        n = node_roles.shape[0]
        nnr = np.zeros((n, 2))
        nnr[:, 0] = node_roles
        node_roles = nnr
        n, m = node_roles.shape

    node_roles_with_id = np.zeros((n, m+1))
    node_roles_with_id[:, 0] = node_ids
    node_roles_with_id[:, 1:] = node_roles

    try:
        n, m = node_roles_two.shape
    except ValueError:
        n = node_roles_two.shape[0]
        nnr = np.zeros((n, 2))
        nnr[:, 0] = node_roles_two
        node_roles_two = nnr
        n, m = node_roles.shape

    node_roles_two_with_id = np.zeros((n, m+1))
    node_roles_two_with_id[:, 0] = node_ids2
    node_roles_two_with_id[:, 1:] = node_roles_two

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

    roles_assignments_one = get_node_role_assignment(node_roles_with_id)
    roles_assignments_two = get_node_role_assignment(node_roles_two_with_id)

    primary_blocks, secondary_blocks = get_role_blocks(roles_assignments_one)
    # secondary_blocks, primary_blocks = get_role_blocks(roles_assignments)

    role_counts_total = sum([len(primary_blocks[i]) for i in primary_blocks.keys() if i != -1])
    role_probabilities = [float(len(primary_blocks[i])) / role_counts_total for i in primary_blocks.keys() if i != -1]

    if -1 in primary_blocks:
        role_index_map = dict([(x-1, y) for x, y in enumerate(sorted(primary_blocks.keys())) if y != -1])
    else:
        role_index_map = dict([(x, y) for x, y in enumerate(sorted(primary_blocks.keys()))])

    random_role_allotment_counts = defaultdict(int)
    random_roles_assignment = defaultdict(list)

    sampled_nodes = np.arange(node_roles.shape[0])[:role_counts_total]
    np.random.shuffle(sampled_nodes)

    for node in sampled_nodes:
        while True:
            outcome = np.nonzero(np.random.multinomial(1, role_probabilities, size=1)[0, :] == 1)[0][0]
            outcome = role_index_map[outcome]
            if random_role_allotment_counts[outcome] < len(primary_blocks[outcome]):
                random_role_allotment_counts[outcome] += 1
                random_roles_assignment[outcome].append(node)
                break

    node_pairs = defaultdict(list)
    total_pairs = 0
    for key in random_roles_assignment.keys():
        if key == -1:
            continue
        block = random_roles_assignment[key]
        block_size = len(block)
        if block_size > 1:
            for i in xrange(0, block_size - 1):
                for j in xrange(i + 1, block_size):
                    node_pairs[block[i]].append(block[j])
                    total_pairs += 1
    print '*'*50
    print 'Total Random Pairs: ', total_pairs

    labels = {'between': 1, 'close': 2, 'bcc': 3, 'ed0': 4, 'ed1': 5,
              'ew0': 6, 'ew1': 7, 'deg': 8, 'wdeg': 9, 'clusc': 10}

    #### FOR RANDOM Allocations
    bet = []
    close = []
    bcc = []
    ew0 =[]
    ed0 = []
    ew1 = []
    ed1 = []
    clusc = []
    deg = []
    wt_deg = []

    for iter in xrange(20):
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

        for a in node_pairs.keys():
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

            for b in node_pairs[a]:
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

        bet.append(compute_zero_binned_percentage(bet_diff))
        close.append(compute_zero_binned_percentage(closeness_diff))
        bcc.append(compute_zero_binned_percentage(bcc_diff))
        ew0.append(compute_zero_binned_percentage(ew0_diff))
        ed0.append(compute_zero_binned_percentage(ed0_diff))
        ew1.append(compute_zero_binned_percentage(ew1_diff))
        ed1.append(compute_zero_binned_percentage(ed1_diff))
        clusc.append(compute_zero_binned_percentage(clusc_diff))
        deg.append(compute_zero_binned_percentage(degree_diff))
        wt_deg.append(compute_zero_binned_percentage(wdegree_diff))

    final_diffs_random = [np.mean(bet), np.mean(close), np.mean(bcc), np.mean(ed0), np.mean(ed1), np.mean(ew0),
                          np.mean(ew1), np.mean(deg), np.mean(wt_deg), np.mean(clusc)]

    final_std_random = [np.std(bet), np.std(close), np.std(bcc), np.std(ed0), np.std(ed1), np.std(ew0),
                        np.std(ew1), np.std(deg), np.std(wt_deg), np.std(clusc)]

    ##### For Network Co-Evolution
    primary_node_pairs = defaultdict(list)

    total_pairs = 0
    evolved_to_same_block = 0
    for key in primary_blocks.keys():
        if key == -1:
            continue
        block = primary_blocks[key]
        block_size = len(block)
        if block_size > 1:
            for i in xrange(0, block_size - 1):
                for j in xrange(i + 1, block_size):
                    u = block[i]
                    v = block[j]
                    primary_node_pairs[u].append(v)
                    total_pairs += 1
                    if is_role_same(roles_assignments_two, u, v):
                        evolved_to_same_block += 1

    print 'Total Actual Pairs: ', total_pairs
    print 'Evolved as Same Pairs: ', evolved_to_same_block

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

    final_diffs = [compute_zero_binned_percentage(bet_diff), compute_zero_binned_percentage(closeness_diff),
                   compute_zero_binned_percentage(bcc_diff), compute_zero_binned_percentage(ed0_diff),
                   compute_zero_binned_percentage(ed1_diff), compute_zero_binned_percentage(ew0_diff),
                   compute_zero_binned_percentage(ew1_diff), compute_zero_binned_percentage(degree_diff),
                   compute_zero_binned_percentage(wdegree_diff), compute_zero_binned_percentage(clusc_diff)]

    np.savetxt(out_file, np.asarray([final_diffs, final_diffs_random, final_std_random,
                                     [total_pairs, 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [float(evolved_to_same_block) / total_pairs * 100.0,
                                      0., 0., 0., 0., 0., 0., 0., 0., 0.]]))