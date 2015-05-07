__author__ = 'pratik'

import nimfa
import numpy as np
import argparse

'''
V ~ GF is estimated using NMF for a factorization rank r.

where,
V = n * f, node-feature matrix
G = n * r, node-role matrix
F = r * f, role-feature matrix

For GLRD,

1. Iterate for  k = 1 to r (and for each role),
   Repeat steps 2 to 9. 'r' is the factorization rank/(# of roles) and is input to the program.
2. Calculate V ~ GF, using NMF
3. R = V - G(.)(!k) * F(!k)(.)  # residual
4. Calculate G(.)(!k) by solving for x as:
5. x_star = argmin_x ||R - xF(k)(.)||_2
6. G(.)(!k) = argmin_x ||x_star - x||_2 s.t. g_i(x) <= epsilon_i: for all i
7. Update G
8. Similarly, compute F(k)(.)
9. Update F

where,
G(.)(i) - denotes the i^th column vector of G.
F(i)(.) - denotes the i^th row vector of F.
'''


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(prog='compute glrd')
    argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    argument_parser.add_argument('-o', '--output-prefix', help='glrd output prefix', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='glrd output dir', required=True)

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    out_prefix = args.output_prefix
    out_dir = args.output_dir

    refex_features = np.loadtxt(node_feature, delimiter=',')
    actual_fx_matrix = refex_features[:, 1:]

    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    sparsity_threshold = 2.0
    for i in xrange(1, 6):
        for rank in xrange(10, 20 + 1):
            snmf = nimfa.Snmf(actual_fx_matrix, seed="random_vcol", version='r', rank=rank, beta=2.0)
            snmf_fit = snmf()
            G = np.asarray(snmf_fit.basis())
            F = np.asarray(snmf_fit.coef())

            w_out = '%s-%s-%s-nodeRoles.txt' % (rank, i, out_prefix)
            h_out = '%s-%s-%s-roleFeatures.txt' % (rank, i, out_prefix)

            np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-nodeRoles.txt", X=G)
            np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-roleFeatures.txt", X=F)
