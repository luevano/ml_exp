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
                   bohr_radius_units=False):
    """
    Creates the Coulomb Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    size: compound size.
    as_eig: if data should be returned as matrix or array of eigenvalues.
    bohr_radius_units: if units should be in bohr's radius units.
    """
    if bohr_radius_units:
        conversion_rate = 0.52917721067
    else:
        conversion_rate = 1

    mol_n = len(coords)
    mol_nr = range(mol_n)

    if not mol_n == len(nc):
        print(''.join(['Error. Molecule matrix dimension is different ',
                       'than the nuclear charge array dimension.']))
    else:
        if size < mol_n:
            print(''.join(['Error. Molecule matrix dimension (mol_n) is ',
                           'greater than size. Using mol_n.']))
            size = None

        if size:
            cm = np.zeros((size, size))
            ml_r = range(size)

            # Actual calculation of the coulomb matrix.
            for i in ml_r:
                if i < mol_n:
                    x_i = coords[i, 0]
                    y_i = coords[i, 1]
                    z_i = coords[i, 2]
                    Z_i = nc[i]
                else:
                    break

                for j in ml_r:
                    if j < mol_n:
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
                            cm[i, j] = (conversion_rate*Z_i*Z_j/math.sqrt(x
                                                                          + y
                                                                          + z))
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

        else:
            cm_temp = []
            # Actual calculation of the coulomb matrix.
            for i in mol_nr:
                x_i = coords[i, 0]
                y_i = coords[i, 1]
                z_i = coords[i, 2]
                Z_i = nc[i]

                cm_row = []
                for j in mol_nr:
                    x_j = coords[j, 0]
                    y_j = coords[j, 1]
                    z_j = coords[j, 2]
                    Z_j = nc[j]

                    x = (x_i-x_j)**2
                    y = (y_i-y_j)**2
                    z = (z_i-z_j)**2

                    if i == j:
                        cm_row.append(0.5*Z_i**2.4)
                    else:
                        cm_row.append(conversion_rate*Z_i*Z_j/math.sqrt(x
                                                                        + y
                                                                        + z))

                cm_temp.append(np.array(cm_row))

            cm = np.array(cm_temp)
            # Now the value will be returned.
            if as_eig:
                return np.sort(np.linalg.eig(cm)[0])[::-1]
            else:
                return cm
