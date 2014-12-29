import features
import argparse
from nonnegfac.nmf import NMF
from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute riders matrix')
    argument_parser.add_argument('-g', '--graph', help='input graph file', required=True)
    argument_parser.add_argument('-b', '--bins', help='input graph file', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)

    args = argument_parser.parse_args()

    graph_file = args.graph
    rider_dir = args.rider_dir
    bins = int(args.bins)

    fx = features.Features()

    fx_matrix = fx.only_riders(graph_file=graph_file, rider_dir=rider_dir, bins=bins)

    W, H, info = NMF().run(fx_matrix, 10)
    W, H, info = NMF_ANLS_BLOCKPIVOT().run(fx_matrix, 10, max_iter=50)

    fx.save_feature_matrix("intermediate.txt")
    print 'Written Intermediate Fx to file'