import numpy as np
from numpy.linalg import cholesky


def cholesky_solve(K, y):
    """
    Applies Cholesky decomposition to obtain the 'alpha coeficients'.
    K: kernel.
    y: known parameters.
    """
    # The initial mathematical problem is to solve Ka=y.

    # First, add a small lambda value.
    K[np.diag_indices_from(K)] += 1e-8

    # Get the Cholesky decomposition of the kernel.
    L = cholesky(K)
    size = len(L)

    # Solve Lx=y for x.
    x = np.zeros(size)
    x[0] = y[0] / L[0, 0]
    for i in range(1, size):
        temp_sum = 0.0
        for j in range(i):
            temp_sum += L[i, j] * x[j]
        x[i] = (y[i] - temp_sum) / L[i, i]

    # Now, solve LTa=x for a.
    L2 = L.T
    a = np.zeros(size)
    a_ms = size - 1
    a[a_ms] = x[a_ms] / L2[a_ms, a_ms]
    # Because of the form of L2 (upper triangular matriz), an inversion of
    # range() needs to be done.
    for i in range(0, a_ms)[::-1]:
        temp_sum = 0.0
        for j in range(i, size)[::-1]:
            temp_sum += L2[i, j] * a[j]
        a[i] = (x[i] - temp_sum) / L2[i, i]

    return a
