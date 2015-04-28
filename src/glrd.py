__author__ = 'pratik'

import nimfa
import numpy as np
import cvxpy as cvx
from numpy import linalg
import argparse
import mdl
import sys

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

# numpy 2d array slicing:
# test = numpy.array([[1, 2], [3, 4], [5, 6]])
# test[:,:] => full array
# test[0,:] => 1st row
# test[:,0] => 1st col
# test[:,1] => 2nd col


def glrd_sparse(V, G, F, r, err_V, err_F):
    # sparsity threshold is num_nodes / num_roles
    for k in xrange(r):
        G_copy = np.copy(G)  # create local copies for excluding the k^th col and row of G and F resp.
        F_copy = np.copy(F)
        G_copy[:, k] = 0.0
        F_copy[k, :] = 0.0

        R = np.abs(V - np.dot(G_copy, F_copy))  # compute residual

        # Solve for optimal G(.)(k) with sparsity constraints
        F_k = F[k, :]
        x_star_G = linalg.lstsq(R.T, F_k.T)[0].T
        x_G = cvx.Variable(x_star_G.shape[0])
        objective_G = cvx.Minimize(cvx.norm2(x_star_G - x_G))
        constraints_G = [cvx.norm1(x_G) <= err_V, x_G >= 0]
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
        constraints_F = [cvx.sum_entries(x_F) <= err_F, x_F >= 0]
        prob_F = cvx.Problem(objective_F, constraints_F)
        result = prob_F.solve(solver='SCS')
        if not np.isinf(result):
            F_k_min = np.asarray(x_F.value)
            F[k, :] = F_k_min[0, :]
        else:
            print result

    return G, F


def glrd_diverse(V, G, F, r, err_V, err_F):
    # diversity threshold is 0.5
    for k in xrange(r):
        G_copy = np.copy(G)  # create local copies for excluding the k^th col and row of G and F resp.
        F_copy = np.copy(F)
        G_copy[:, k] = 0.0
        F_copy[k, :] = 0.0

        R = V - np.dot(G_copy, F_copy)  # compute residual

        # Solve for optimal G(.)(k) with diversity constraints
        F_k = F[k, :]
        x_star_G = linalg.lstsq(R.T, F_k.T)[0].T
        x_G = cvx.Variable(x_star_G.shape[0])

        objective_G = cvx.Minimize(cvx.norm2(x_star_G - x_G))

        constraints_G = [x_G >= 0]
        for j in xrange(r):
            if j != k:
                constraints_G += [x_G.T * G[:, j] <= err_V]

        prob_G = cvx.Problem(objective_G, constraints_G)
        result = prob_G.solve(solver='SCS')
        if not np.isinf(result):
            G_k_min = np.asarray(x_G.value)
            G[:, k] = G_k_min[:, 0]
        else:
            print result

        # Solve for optimal F(k)(.) with diversity constraints
        G_k = G[:, k]
        x_star_F = linalg.lstsq(R, G_k)[0]
        x_F = cvx.Variable(x_star_F.shape[0])
        objective_F = cvx.Minimize(cvx.norm2(x_star_F - x_F))

        constraints_F = [x_F >= 0]
        for j in xrange(r):
            if j != k:
                constraints_F += [x_F.T * F[j, :] <= err_F]

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

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    out_prefix = args.output_prefix

    with open(node_feature) as nf:
        n_cols = len(nf.readline().strip().split(','))

    actual_fx_matrix = np.loadtxt(node_feature, delimiter=',', usecols=range(1, n_cols))
    m, n = actual_fx_matrix.shape
    print 'Number of Features: ', n
    print 'Number of Nodes: ', m

    number_bins = int(np.log2(m))
    max_roles = min([m, n])
    best_G = None
    best_F = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0
    threshold_sparse = 0.5
    for rank in xrange(1, max_roles + 1):
        # threshold_sparse = float(max_roles) / rank
        fctr = nimfa.mf(actual_fx_matrix, rank=rank, method="lsnmf", max_iter=100)
        fctr_res = nimfa.mf_run(fctr)
        G = np.asarray(fctr_res.basis())
        F = np.asarray(fctr_res.coef())

        G, F = glrd_diverse(V=actual_fx_matrix, G=G, F=F, r=rank, err_V=threshold_sparse, err_F=threshold_sparse)
        code_length_G = mdlo.get_huffman_code_length(G)
        code_length_F = mdlo.get_huffman_code_length(F)

        # For total bit length:
        # model_cost = code_length_W + code_length_H  # For total bit length
        # For avg. symbol bit length:
        model_cost = code_length_G * (G.shape[0] + G.shape[1]) + code_length_F * (F.shape[0] + F.shape[1])
        estimated_matrix = np.asarray(np.dot(G, F))
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)

        description_length = model_cost - loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_G = np.copy(G)
            best_F = np.copy(F)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 10:
                break

        print 'Number of Roles: %s, Model Cost: %.2f, -loglikelihood: %.2f, Description Length: %.2f, MDL: %.2f (%s)' \
              % (rank, model_cost, loglikelihood, description_length, minimum_description_length, best_G.shape[1])

    print 'MDL has not changed for these many iters:', min_des_not_changed_counter
    print '\nMDL: %.2f, Roles: %s' % (minimum_description_length, best_G.shape[1])
    np.savetxt('out-' + out_prefix + "-nodeRoles.txt", X=best_G, delimiter=',')
    np.savetxt('out-' + out_prefix + "-rolesFeatures.txt", X=best_F, delimiter=',')
