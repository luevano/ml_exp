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


def gaussian_kernel(X1,
                    X2,
                    sigma):
    """
    Calculates the Gaussian Kernel.
    X1: first representations.
    X2: second representations.
    sigma: kernel width.
    """
    inv_sigma = -0.5 / (sigma*sigma)

    K = np.zeros((X1.shape[0], X2.shape[0]), dtype=float)
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            f_norm = np.linalg.norm(x1 - x2)
            # print(f_norm)
            K[i, j] = math.exp(inv_sigma * f_norm)

    return K
