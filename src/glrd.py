__author__ = 'pratik'

import nimfa
import numpy as np
import cvxpy as cvx
from numpy import linalg

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

# Sample data.
m = 30
n = 20
r = 4

np.random.seed(1)
V = np.random.random_integers(1, 10, (m, n))

fctr = nimfa.mf(V, rank=r, method="lsnmf", max_iter=100)
fctr_res = nimfa.mf_run(fctr)
G = np.asarray(fctr_res.basis())
F = np.asarray(fctr_res.coef())

# numpy 2d array slicing:
# test = numpy.array([[1, 2], [3, 4], [5, 6]])
# test[:,:] => full array
# test[0,:] => 1st row
# test[:,0] => 1st col
# test[:,1] => 2nd col

for k in xrange(r):
    G_copy = np.copy(G)  # create local copies for excluding the k^th col and row of G and F resp.
    F_copy = np.copy(F)
    G_copy[:, k] = 0.0
    F_copy[k, :] = 0.0

    R = V - np.dot(G_copy, F_copy)  # compute residual

    # Solve for optimal G(.)(k)
    F_k = F[k, :]
    x_star = linalg.lstsq(R.T, F_k.T)[0].T
    x = cvx.Variable(x_star.shape[0])
    objective = cvx.Minimize(cvx.norm2(x_star - x))
    constraints = [cvx.norm1(x) <= 0.5, x >= 0.0]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    G_k_min = np.asarray(x.value)
    G[:, k] = G_k_min[:, 0]

    # Solve for optimal F(k)(.)
    G_k = G[:, k]
    x_star = linalg.lstsq(R, G_k.T)[0].T
    x = cvx.Variable(x_star.shape[0])
    objective = cvx.Minimize(cvx.norm2(x_star - x))
    constraints = [cvx.norm1(x) <= 0.5, x >= 0.0]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    F_k_min = np.asarray(x.value)
    F[k, :] = F_k_min[0, :]
