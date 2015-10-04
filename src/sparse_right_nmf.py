import nimfa
import numpy as np
import cvxpy as cvx
from numpy import linalg as LA
import time

current_milli_time = lambda: int(round(time.time() * 1000))


def sparse_right_nmf(V, rank, max_iters=50, eta=1., beta=1e-4):
    np.random.seed(10)
    m, n = V.shape

    W_init = np.random.rand(m, rank)
    W = W_init

    for iter_num in range(1, max_iters+1):
        # for odd itereration, W is constant, optimize H
        if iter_num % 2 == 1:
            H = cvx.Variable(rank, n)
            constraint = [H >= 0]
            e = np.ones((1, rank))
            z = np.zeros((1, n))
            obj_W = np.append(W, np.sqrt(beta) * e, axis=0)
            obj_V = np.append(V, z, axis=0)
            objective = cvx.Minimize(cvx.norm(obj_W * H - obj_V, 'fro'))
        else:
            W = cvx.Variable(m, rank)
            constraint = [W >= 0]
            I_k = np.identity(rank)
            z = np.zeros((rank, m))
            obj_H_T = np.append(H.T, np.sqrt(eta) * I_k, axis=0)
            obj_V_T = np.append(V.T, z, axis=0)
            objective = cvx.Minimize(cvx.norm(obj_H_T * W.T - obj_V_T, 'fro'))

        problem = cvx.Problem(objective, constraint)
        problem.solve(solver='SCS')

        if problem.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")

        if iter_num % 2 == 1:
            H = H.value
        else:
            W = W.value

    return W, H


def nmf(A, k, max_iters=30):
    m, n = A.shape
    # Initialize Y randomly.
    Y_init = np.random.rand(m, k)
    # Ensure same initial random Y, rather than generate new one
    # when executing this cell.
    Y = Y_init

    # Perform alternating minimization.
    MAX_ITERS = max_iters
    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1+MAX_ITERS):
        print iter_num
        # At the beginning of an iteration, X and Y are NumPy
        # array types, NOT CVXPY variables.

        # For odd iterations, treat Y constant, optimize over X.
        if iter_num % 2 == 1:
            X = cvx.Variable(k, n)
            constraint = [X >= 0]
        # For even iterations, treat X constant, optimize over Y.
        else:
            Y = cvx.Variable(m, k)
            constraint = [Y >= 0]

        # Solve the problem.
        obj = cvx.Minimize(cvx.norm(A - Y*X, 'fro'))
        prob = cvx.Problem(obj, constraint)
        prob.solve(solver=cvx.SCS)

        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")

        print 'Iteration {}, residual norm {}'.format(iter_num, prob.value)
        residual[iter_num-1] = prob.value

        # Convert variable to NumPy array constant for next iteration.
        if iter_num % 2 == 1:
            X = X.value
        else:
            Y = Y.value

    return Y, X

V = np.random.rand(1000, 40)

start = current_milli_time()
w, h = sparse_right_nmf(V, rank=15, max_iters=10, beta=2.0)
end = current_milli_time()
# w, h =nmf(V, k=4, max_iters=30)
a = np.abs(V - np.dot(w,h))
c = LA.norm(a, 'fro')
print c, (end-start)

start = current_milli_time()
snmf = nimfa.Snmf(V, seed="random_vcol", version='r', rank=15, beta=2.0, max_iter=10)
snmf_fit = snmf()

G = np.asarray(snmf_fit.basis())
F = np.asarray(snmf_fit.coef())
end = current_milli_time()

a = np.abs(V- np.dot(G,F))
c = LA.norm(a, 'fro')
print c, (end-start)
