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


def coulomb_matrix(coords,
                   nc,
                   size=23,
                   as_eig=True,
                   bhor_ru=False):
    """
    Creates the Coulomb Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    size: compound size.
    as_eig: if the representation should be as the eigenvalues.
    bhor_ru: if radius units should be in bohr's radius units.
    """
    if bhor_ru:
        cr = 0.52917721067
    else:
        cr = 1

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge\
                         size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)\
              instead of (size).')
        size = n

    nr = range(size)
    cm = np.zeros((size, size), dtype=float)

    # Actual calculation of the coulomb matrix.
    for i in nr:
        if i < n:
            x_i = coords[i, 0]
            y_i = coords[i, 1]
            z_i = coords[i, 2]
            Z_i = nc[i]
        else:
            break

        for j in nr:
            if j < n:
                x_j = coords[j, 0]
                y_j = coords[j, 1]
                z_j = coords[j, 2]
                Z_j = nc[j]

                x = (x_i-x_j)**2
                y = (y_i-y_j)**2
                z = (z_i-z_j)**2

                if i == j:
                    cm[i, j] = (0.5*Z_i**2.4)
                else:
                    cm[i, j] = (cr*Z_i*Z_j/math.sqrt(x + y + z))
            else:
                break

    # Now the value will be returned.
    if as_eig:
        cm_sorted = np.sort(np.linalg.eig(cm)[0])[::-1]
        # Thanks to SO for the following lines of code.
        # https://stackoverflow.com/a/43011036

        # Keep zeros at the end.
        mask = cm_sorted != 0.
        f_mask = mask.sum(0, keepdims=1) >\
            np.arange(cm_sorted.shape[0]-1, -1, -1)

        f_mask = f_mask[::-1]
        cm_sorted[f_mask] = cm_sorted[mask]
        cm_sorted[~f_mask] = 0.

        return cm_sorted
    else:
        return cm
