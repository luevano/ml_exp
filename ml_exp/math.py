"""MIT License

Copyright (c) 2019 David Luevano Alvarado

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np


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
    L = np.linalg.cholesky(K)
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
