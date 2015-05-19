__author__ = 'pratik'

import nimfa
import numpy as np
import cvxpy as cvx
from numpy import linalg
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


def glrd_sparse(V, G, F, r, err_V, err_F):
    # sparsity threshold is num_nodes / num_roles
    for k in xrange(r):
        G_copy = np.copy(G)  # create local copies for excluding the k^th col and row of G and F resp.
        F_copy = np.copy(F)
        G_copy[:, k] = 0.0
        F_copy[k, :] = 0.0

        R = V - np.dot(G_copy, F_copy)  # compute residual

        # Solve for optimal G(.)(k) with sparsity constraints
        F_k = F[k, :]
        x_star_G = linalg.lstsq(R.T, F_k.T)[0].T
        x_G = cvx.Variable(x_star_G.shape[0])
        objective_G = cvx.Minimize(cvx.norm2(x_star_G - x_G))
        constraints_G = [x_G >= 0]
        constraints_G += [cvx.norm1(x_G) <= err_V]
        prob_G = cvx.Problem(objective_G, constraints_G)
        result = prob_G.solve(solver='SCS')
        if not np.isinf(result):
            G_k_min = np.asarray(x_G.value)
            G[:, k] = G_k_min[:, 0]
        else:
            print result

        # Solve for optimal F(k)(.) with sparsity constraints
        G_k = G[:, k]
        x_star_F = linalg.lstsq(R, G_k)[0]
        x_F = cvx.Variable(x_star_F.shape[0])
        objective_F = cvx.Minimize(cvx.norm2(x_star_F - x_F))
        constraints_F = [x_F >= 0]
        constraints_F += [cvx.sum_entries(x_F) <= err_F]
        prob_F = cvx.Problem(objective_F, constraints_F)
        result = prob_F.solve(solver='SCS')
        if not np.isinf(result):
            F_k_min = np.asarray(x_F.value)
            F[k, :] = F_k_min[0, :]
        else:
            print result

    return G, F


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

    sparsity_threshold = 1.0
    for i in xrange(1, 6):
        for rank in xrange(20, 29 + 1):
            lsnmf = nimfa.Lsnmf(actual_fx_matrix, rank=rank, max_iter=200)
            lsnmf_fit = lsnmf()
            G = np.asarray(lsnmf_fit.basis())
            F = np.asarray(lsnmf_fit.coef())

            G, F = glrd_sparse(V=actual_fx_matrix, G=G, F=F, r=rank, err_V=sparsity_threshold, err_F=sparsity_threshold)
            G[G <= 0.0] = 0.0
            F[F <= 0.0] = 0.0

            w_out = '%s-%s-%s-nodeRoles.txt' % (rank, i, out_prefix)
            h_out = '%s-%s-%s-roleFeatures.txt' % (rank, i, out_prefix)
            np.savetxt(out_dir + w_out, X=G)
            np.savetxt(out_dir + h_out, X=F)
