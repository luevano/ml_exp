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
from scipy.stats import wasserstein_distance
try:
    import tensorflow as tf
    TF_AV = True
except ImportError:
    print('Tensorflow couldn\'t be imported. Maybe it is not installed.')
    TF_AV = False


def gaussian_kernel(X1,
                    X2,
                    sigma,
                    use_tf=True):
    """
    Calculates the Gaussian Kernel.
    X1: first representations.
    X2: second representations.
    sigma: kernel width.
    use_tf: if tensorflow should be used.
    """
    # If tf is to be used but couldn't be imported, don't try to use it.
    if use_tf and not TF_AV:
        use_tf = False

    X1_size = X1.shape[0]
    X2_size = X2.shape[0]
    i_sigma = -0.5 / (sigma**2)

    if use_tf:
        if tf.config.experimental.list_physical_devices('GPU'):
            with tf.device('GPU:0'):
                X1 = tf.convert_to_tensor(X1)
                X2 = tf.convert_to_tensor(X2)
                X2r = tf.rank(X2)

                def cond(i, _):
                    return tf.less(i, X1_size)

                def body(i, K):
                    if X2r == 3:
                        norm = tf.norm(X2 - X1[i], axis=(1, 2))
                    else:
                        norm = tf.norm(X2 - X1[i], axis=-1)

                    return (i + 1,
                            K.write(i, tf.exp(i_sigma * tf.square(norm))))

                K = tf.TensorArray(dtype=tf.float64,
                                   size=X1_size)
                i_state = (0, K)
                n, K = tf.while_loop(cond, body, i_state)
                K = K.stack()
        else:
            raise TypeError('No GPU found, could not create Tensor objects.')
    else:
        K = np.zeros((X1_size, X2_size), dtype=np.float64)
        for i in range(X1_size):
            if X2.ndim == 3:
                norm = np.linalg.norm(X2 - X1[i], axis=(1, 2))
            else:
                norm = np.linalg.norm(X2 - X1[i], axis=-1)
            K[i, :] = np.exp(i_sigma * np.square(norm))

    return K


def laplacian_kernel(X1,
                     X2,
                     sigma,
                     use_tf=True):
    """
    Calculates the Laplacian Kernel.
    X1: first representations.
    X2: second representations.
    sigma: kernel width.
    use_tf: if tensorflow should be used.
    """
    # If tf is to be used but couldn't be imported, don't try to use it.
    if use_tf and not TF_AV:
        use_tf = False

    X1_size = X1.shape[0]
    X2_size = X2.shape[0]
    i_sigma = -0.5 / sigma

    if use_tf:
        if tf.config.experimental.list_physical_devices('GPU'):
            with tf.device('GPU:0'):
                X1 = tf.convert_to_tensor(X1)
                X2 = tf.convert_to_tensor(X2)
                X2r = tf.rank(X2)

                def cond(i, _):
                    return tf.less(i, X1_size)

                def body(i, K):
                    if X2r == 3:
                        norm = tf.norm(X2 - X1[i], axis=(1, 2))
                    else:
                        norm = tf.norm(X2 - X1[i], axis=-1)

                    return (i + 1,
                            K.write(i, tf.exp(i_sigma * norm)))

                K = tf.TensorArray(dtype=tf.float64,
                                   size=X1_size)
                i_state = (0, K)
                n, K = tf.while_loop(cond, body, i_state)
                K = K.stack()
        else:
            raise TypeError('No GPU found, could not create Tensor objects.')
    else:
        K = np.zeros((X1_size, X2_size), dtype=np.float64)
        for i in range(X1_size):
            if X2.ndim == 3:
                norm = np.linalg.norm(X2 - X1[i], axis=(1, 2))
            else:
                norm = np.linalg.norm(X2 - X1[i], axis=-1)
            K[i, :] = np.exp(i_sigma * norm)

    return K


def wasserstein_kernel(X1,
                       X2,
                       alpha):
    """
    Calculates the Wasserstein Kernel.
    X1: first representations.
    X2: second representations.
    alpha: wasserstein kernel parameter.
    NOTE: this doesn't work with tensorflow.
    """

    if X2.ndim == 3:
        raise TypeError('Representations must be 1D.')

    X1_size = X1.shape[0]
    X2_size = X2.shape[0]

    K = np.zeros((X1_size, X2_size), dtype=np.float64)
    for i in range(X1_size):
        norm = np.array([X2[j] - X1[i] for j in range(X2_size)], dtype=np.float64)
        K[i, :] = np.exp(- alpha * norm)

    return K
