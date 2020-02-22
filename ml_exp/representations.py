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
                   bohr_ru=False):
    """
    Creates the Coulomb Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    size: compound size.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    """
    if bohr_ru:
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


def lennard_jones_matrix(coords,
                         nc,
                         diag_value=None,
                         sigma=1.0,
                         epsilon=1.0,
                         size=23,
                         as_eig=True,
                         bohr_ru=False):
    """
    Creates the Lennard-Jones Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    diag_value: if special diagonal value is to be used.
    sigma: sigma value.
    epsilon: epsilon value.
    size: compound size.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    """
    if bohr_ru:
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
    lj = np.zeros((size, size), dtype=float)

    # Actual calculation of the lennard-jones matrix.
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
                # Never really used because when needed it is equal to Z_i.
                # Z_j = nc[j]

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
                    r_2 = sigma**2/(cr**2*(x + y + z))

                    r_6 = math.pow(r_2, 3)
                    r_12 = math.pow(r_6, 2)
                    lj[i, j] = (4*epsilon*(r_12 - r_6))
            else:
                break

    # Now the value will be returned.
    if as_eig:
        lj_sorted = np.sort(np.linalg.eig(lj)[0])[::-1]
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


def first_neighbor_matrix(coords,
                          nc,
                          atoms,
                          use_forces=False,
                          bohr_ru=False):
    """
    Creates the First Neighbor Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    atoms: list of atoms.
    use_forces: if the use of forces instead of k_cx should be used.
    bohr_ru: if radius units should be in bohr's radius units.
    NOTE: Bond distance of carbon to other elements
        are (for atoms present in the qm7 dataset):
            C: 1.20 - 1.53 A
            H: 1.06 - 1.12 A
            O: 1.43 - 2.15 A
            N: 1.47 - 2.10 A
            S: 1.81 - 2.55 A
    """
    if bohr_ru:
        cr = 0.52917721067
    else:
        cr = 1

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge\
                         size. Arrays are not of the right shape.')

    # Possible bonds.
    cc_bond = sorted(['C', 'C'])
    ch_bond = sorted(['C', 'H'])
    co_bond = sorted(['C', 'O'])
    cn_bond = sorted(['C', 'N'])
    cs_bond = sorted(['C', 'S'])

    if use_forces:
        fnm = np.empty((n, n), dtype=float)
    else:
        fnm = np.empty((n, n), dtype=int)

    bonds = []
    forces = []
    for i, xyz_i in enumerate(coords):
        for j, xyz_j in enumerate(coords):

            # Ignore the diagonal.
            if i != j:
                bond = sorted([atoms[i], atoms[j]])
                rv = xyz_i - xyz_j
                r = np.linalg.norm(rv)/cr

                # Check for each type of bond.
                if (cc_bond == bond) and (r >= 1.20 and r <= 1.53):
                    fnm[i, j] = 1.0
                    if j > i:
                        bonds.append((i, j))
                        if use_forces:
                            forces.append(rv*nc[i]*nc[j]/r**3)
                elif (ch_bond == bond) and (r >= 1.06 and r <= 1.12):
                    fnm[i, j] = 1.0
                    if j > i:
                        bonds.append((i, j))
                        if use_forces:
                            forces.append(rv*nc[i]*nc[j]/r**3)
                elif (co_bond == bond) and (r >= 1.43 and r <= 2.15):
                    fnm[i, j] = 0.8
                    if j > i:
                        bonds.append((i, j))
                        if use_forces:
                            forces.append(rv*nc[i]*nc[j]/r**3)
                elif (cn_bond == bond) and (r >= 1.47 and r <= 2.10):
                    fnm[i, j] = 1.0
                    if j > i:
                        bonds.append((i, j))
                        if use_forces:
                            forces.append(rv*nc[i]*nc[j]/r**3)
                elif (cs_bond == bond) and (r >= 1.81 and r <= 2.55):
                    fnm[i, j] = 0.7
                    if j > i:
                        bonds.append((i, j))
                        if use_forces:
                            forces.append(rv*nc[i]*nc[j]/r**3)

    return fnm, bonds, forces


def adjacency_matrix(fnm,
                     bonds,
                     forces,
                     size=23):
    """
    Calculates the adjacency matrix given the bond list.
    fnm: first neighbour matrix.
    bonds: list of bonds (tuple of indexes).
    forces: list of forces.
    size: compund size.
    """
    n = len(bonds)

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)\
              instead of (size).')
        size = n

    if forces:
        am = np.empty((size, size), dtype=float)
    else:
        am = np.empty((size, size), dtype=int)

    for i, bond_i in enumerate(bonds):
        for j, bond_j in enumerate(bonds):
            # Ignore the diagonal.
            if i != j:
                if (bond_i[0] in bond_j) or (bond_i[1] in bond_j):
                    if forces:
                        am[i, j] = np.dot(forces[i], forces[j])
                    else:
                        am[i, j] = fnm[bond_i[0], bond_i[1]]

    return am
