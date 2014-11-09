import features
import argparse

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='run_recursive_feature_extraction')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-i', '--iterations', help='number of refex iterations', required=True)
    argument_parser.add_argument('-s', '--max-diff', help='start value of feature similarity threshold (default=0)',
                                 default=0, type=int)
    argument_parser.add_argument('-r', '--rider', help='enable rider features (default)', dest='rider',
                                 action='store_true')
    argument_parser.add_argument('-dr', '--disable-rider', help='disable rider features', dest='rider',
                                 action='store_false')
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory (specify, if rider is enabled)')
    argument_parser.set_defaults(rider=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    max_diff = args.max_diff

    if args.rider:
        rider_dir = args.rider_dir
        if rider_dir is None:
            raise Exception('Please specify the RIDeR directory!!')
    else:
        rider_dir = 'INVALID_RIDER_PATH'

    fx = features.Features()

    fx.MAX_ITERATIONS = int(args.iterations)

    # load input graph
    fx.load_graph(graph_file)

    # compute primitive/rider features
    fx.compute_primitive_features(rider_fx=args.rider, rider_dir=rider_dir)

    # compute initial feature matrix
    primitive_feature_matrix = fx.create_initial_feature_matrix()

    # prune any redundant primitive/rider features
    prev_pruned_fx_matrix = fx.compare_and_prune_vertex_fx_vectors(prev_feature_vector=None,
                                                                   new_feature_vector=primitive_feature_matrix,
                                                                   max_dist=max_diff)

    prev_pruned_fx_size = len(list(prev_pruned_fx_matrix.dtype.names))

    fx.update_feature_matrix_to_graph(prev_pruned_fx_matrix)

    current_pruned_fx_size = 0
    print 'Initial number of features: %s' % prev_pruned_fx_size

    no_iterations = 0

    while no_iterations <= fx.MAX_ITERATIONS:
        # compute and prune recursive features for iteration #no_iterations
        current_iteration_pruned_fx_matrix = fx.compute_recursive_features(prev_fx_matrix=prev_pruned_fx_matrix,
                                                                           iter_no=no_iterations, max_dist=max_diff)

        if current_iteration_pruned_fx_matrix is None:
            print 'No new features added, all pruned. Exiting!'
            break

        current_pruned_fx_size = len(list(current_iteration_pruned_fx_matrix.dtype.names))

        print 'Iteration: %s, Number of Features: %s' % (no_iterations, current_pruned_fx_size)

        if current_pruned_fx_size == prev_pruned_fx_size:
            print 'No new features added, Exiting!'
            break

        # update the latest feature matrix to the graph
        fx.update_feature_matrix_to_graph(current_iteration_pruned_fx_matrix)

        # update the previous iteration feature matrix with the latest one
        prev_pruned_fx_matrix = current_iteration_pruned_fx_matrix
        prev_pruned_fx_size = current_pruned_fx_size

        # increment feature similarity threshold by 1
        max_diff += 1

        no_iterations += 1

    fx.save_feature_matrix("featureValues.csv")