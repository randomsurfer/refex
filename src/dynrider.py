import features
import argparse
import numpy as np
import scipy.optimize as opt
import mdl
import nimfa

def estimate_W(V, H):
    W = np.zeros((V.shape[0], H.shape[0]))
    print V.shape, H.shape
    for j in xrange(0, W.shape[0]):
        res = opt.nnls(H.T, V[j, :])
        W[j, :] = res[0]
    return W


def load_role_fx_matrix(rf_matrix_file):
    matrix = []
    for line in open(rf_matrix_file):
        line = line.strip().split(',')
        row = [float(value) for value in line]
        matrix.append(row)
    return np.asarray(matrix)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='dynamic riders')
    argument_parser.add_argument('-g', '--graph', help='current graph', required=True)
    argument_parser.add_argument('-bfd', '--base-rider-dir', help='base rider directory', required=True)
    argument_parser.add_argument('-rf', '--role-feature', help='input role feature matrix', required=True)
    # argument_parser.add_argument('-b', '--bins', help='bins for rider features', required=True)
    argument_parser.add_argument('-rd', '--rider-dir', help='rider directory', required=True)
    # argument_parser.add_argument('-dn', '--dir-no', help='max timestamp directory number', required=True)
    argument_parser.add_argument('-o', '--output-dir', help='output dir', required=True)

    args = argument_parser.parse_args()

    curr_graph = args.graph
    base_fx_dir = args.base_rider_dir
    rf_matrix = args.role_feature
    rider_dir = args.rider_dir
    out_dir = args.output_dir
    H = load_role_fx_matrix(rf_matrix)

    losses = []
    original = []

    mdlo = mdl.MDL(15)

    for i in xrange(2, 13):
        fx = features.Features()
        actual_matrix = fx.dyn_rider(curr_graph, 151, base_fx_dir, rider_dir+'/'+str(i)+'/', bins=15)
        # W = estimate_W(actual_matrix, H)
        # losses.append(mdlo.get_reconstruction_error(actual_matrix, W.dot(H)))
        out_prefix = out_dir + '/' + str(i)

        fctr = nimfa.mf(actual_matrix, rank=8, method="lsnmf", max_iter=100)
        fctr_res = nimfa.mf_run(fctr)
        Wa = np.asarray(fctr_res.basis())
        np.savetxt(out_prefix+"-nodeRoles.txt", X=Wa, delimiter=',')
        # Ha = np.asarray(fctr_res.coef())
        # estimated_matrix = np.asarray(np.dot(Wa, Ha))
        # original.append(mdlo.get_reconstruction_error(actual_matrix, estimated_matrix))
    # print losses
    # print original