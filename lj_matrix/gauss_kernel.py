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
import math
import numpy as np
from lj_matrix.frob_norm import frob_norm


def gauss_kernel(X_1, X_2, sigma):
    """
    Calculates the Gaussian Kernel.
    X_1: first representations.
    X_2: second representations.
    sigma: kernel width.
    """
    x1_l = len(X_1)
    x1_range = range(x1_l)
    x2_l = len(X_2)
    x2_range = range(x2_l)

    inv_sigma = -0.5 / (sigma*sigma)

    K = np.zeros((x1_l, x2_l))
    for i in x1_range:
        for j in x2_range:
            f_norm = frob_norm(X_1[i] - X_2[j])
            # print(f_norm)
            K[i, j] = math.exp(inv_sigma * f_norm)

    return K
