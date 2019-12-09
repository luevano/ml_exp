import math
import numpy as np
from frob_norm import frob_norm


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
