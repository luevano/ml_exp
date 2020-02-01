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
import time
import math
import numpy as np
from numpy.linalg import eig
from ml_exp.misc import printc


def lj_matrix(mol_data,
              nc_data,
              diag_value=None,
              sigma=1.0,
              epsilon=1.0,
              max_len=25,
              as_eig=True,
              bohr_radius_units=False):
    """
    Creates the Lennard-Jones Matrix from the molecule data given.
    mol_data: molecule data, matrix of atom coordinates.
    nc_data: nuclear charge data, array of atom data.
    diag_value: if special diagonal value is to be used.
    sigma: sigma value.
    epsilon: epsilon value.
    max_len: maximum amount of atoms in molecule.
    as_eig: if data should be returned as matrix or array of eigenvalues.
    bohr_radius_units: if units should be in bohr's radius units.
    """
    if bohr_radius_units:
        conversion_rate = 0.52917721067
    else:
        conversion_rate = 1

    mol_n = len(mol_data)
    mol_nr = range(mol_n)

    if not mol_n == len(nc_data):
        print(''.join(['Error. Molecule matrix dimension is different ',
                       'than the nuclear charge array dimension.']))
    else:
        if max_len < mol_n:
            print(''.join(['Error. Molecule matrix dimension (mol_n) is ',
                           'greater than max_len. Using mol_n.']))
            max_len = None

        if max_len:
            lj = np.zeros((max_len, max_len))
            ml_r = range(max_len)

            # Actual calculation of the coulomb matrix.
            for i in ml_r:
                if i < mol_n:
                    x_i = mol_data[i, 0]
                    y_i = mol_data[i, 1]
                    z_i = mol_data[i, 2]
                    Z_i = nc_data[i]
                else:
                    break

                for j in ml_r:
                    if j < mol_n:
                        x_j = mol_data[j, 0]
                        y_j = mol_data[j, 1]
                        z_j = mol_data[j, 2]

                        x = (x_i-x_j)**2
                        y = (y_i-y_j)**2
                        z = (z_i-z_j)**2

                        if i == j:
                            if diag_value is None:
                                lj[i, j] = (0.5*Z_i**2.4)
                            else:
                                lj[i, j] = diag_value
                        else:
                            # Calculations are done after i==j is checked
                            # so no division by zero is done.

                            # A little play with r exponents
                            # so no square root is calculated.
                            # Conversion factor is included in r^2.

                            # 1/r^2
                            r_2 = sigma**2/(conversion_rate**2*(x + y + z))

                            r_6 = math.pow(r_2, 3)
                            r_12 = math.pow(r_6, 2)
                            lj[i, j] = (4*epsilon*(r_12 - r_6))
                    else:
                        break

            # Now the value will be returned.
            if as_eig:
                lj_sorted = np.sort(eig(lj)[0])[::-1]
                # Thanks to SO for the following lines of code.
                # https://stackoverflow.com/a/43011036

                # Keep zeros at the end.
                mask = lj_sorted != 0.
                f_mask = mask.sum(0, keepdims=1) >\
                    np.arange(lj_sorted.shape[0]-1, -1, -1)

                f_mask = f_mask[::-1]
                lj_sorted[f_mask] = lj_sorted[mask]
                lj_sorted[~f_mask] = 0.

                return lj_sorted

            else:
                return lj

        else:
            lj_temp = []
            # Actual calculation of the coulomb matrix.
            for i in mol_nr:
                x_i = mol_data[i, 0]
                y_i = mol_data[i, 1]
                z_i = mol_data[i, 2]
                Z_i = nc_data[i]

                lj_row = []
                for j in mol_nr:
                    x_j = mol_data[j, 0]
                    y_j = mol_data[j, 1]
                    z_j = mol_data[j, 2]

                    x = (x_i-x_j)**2
                    y = (y_i-y_j)**2
                    z = (z_i-z_j)**2

                    if i == j:
                        if not diag_value:
                            lj_row.append(0.5*Z_i**2.4)
                        else:
                            lj_row.append(diag_value)
                    else:
                        # Calculations are done after i==j is checked
                        # so no division by zero is done.

                        # A little play with r exponents
                        # so no square root is calculated.
                        # Conversion factor is included in r^2.

                        # 1/r^2
                        r_2 = sigma**2/(conversion_rate**2*(x + y + z))

                        r_6 = math.pow(r_2, 3)
                        r_12 = math.pow(r_6, 2)
                        lj_row.append(4*epsilon*(r_12 - r_6))

                lj_temp.append(np.array(lj_row))

            lj = np.array(lj_temp)
            # Now the value will be returned.
            if as_eig:
                return np.sort(eig(lj)[0])[::-1]
            else:
                return lj


def lj_matrix_multiple(mol_data,
                       nc_data,
                       pipe=None,
                       diag_value=None,
                       sigma=1.0,
                       epsilon=1.0,
                       max_len=25,
                       as_eig=True,
                       bohr_radius_units=False):
    """
    Calculates the Lennard-Jones Matrix of multiple molecules.
    mol_data: molecule data, matrix of atom coordinates.
    nc_data: nuclear charge data, array of atom data.
    pipe: for multiprocessing purposes. Sends the data calculated
        through a pipe.
    diag_value: if special diagonal value is to be used.
    sigma: sigma value.
    epsilon: epsilon value.
    max_len: maximum amount of atoms in molecule.
    as_eig: if data should be returned as matrix or array of eigenvalues.
    bohr_radius_units: if units should be in bohr's radius units.
    """
    printc('L-J Matrices calculation started.', 'CYAN')
    tic = time.perf_counter()

    ljm_data = np.array([lj_matrix(mol,
                                   nc,
                                   diag_value,
                                   sigma,
                                   epsilon,
                                   max_len,
                                   as_eig,
                                   bohr_radius_units)
                        for mol, nc in zip(mol_data, nc_data)])

    toc = time.perf_counter()
    printc('\tL-JM calculation took {:.4f} seconds.'.format(toc-tic), 'GREEN')

    if pipe:
        pipe.send(ljm_data)

    return ljm_data