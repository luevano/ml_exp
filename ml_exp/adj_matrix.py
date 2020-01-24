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
from numpy import array, zeros
from numpy.linalg import norm


def fneig_matrix(atoms,
                 xyz):
    """
    Creates the first neighbor matrix of the given molecule data.
    atoms: list of atoms.
    xyz: matrix of atomic coords.
    NOTE: Bond distance of carbon to other elements
        are (for atoms present in the qm7 dataset):
            H: 1.06 - 1.12 A
            O: 1.43 - 2.15 A
            N: 1.47 - 2.10 A
            S: 1.81 - 2.55 A
    """
    # Possible bonds.
    ch_bond = sorted(['C', 'H'])
    co_bond = sorted(['C', 'O'])
    cn_bond = sorted(['C', 'N'])
    cs_bond = sorted(['C', 'S'])

    # Number of atoms, empty matrix and bond list.
    n = len(atoms)
    fnm = array(zeros((n, n)))
    bonds = []
    for i, xyz_i in enumerate(xyz):
        for j, xyz_j in enumerate(xyz):
            # Ignore the diagonal.
            if i != j:
                bond = sorted([atoms[i], atoms[j]])
                r = norm(xyz_i - xyz_j)
                # Check for each type of bond.
                if (ch_bond == bond) and (r >= 1.06 and r <= 1.12):
                    fnm[i, j] = 1
                    if j > i:
                        bonds.append((i, j))
                elif (co_bond == bond) and (r >= 1.43 and r <= 2.15):
                    fnm[i, j] = 1
                    if j > i:
                        bonds.append((i, j))
                elif (cn_bond == bond) and (r >= 1.47 and r <= 2.10):
                    fnm[i, j] = 1
                    if j > i:
                        bonds.append((i, j))
                elif (cs_bond == bond) and (r >= 1.81 and r <= 2.55):
                    fnm[i, j] = 1
                    if j > i:
                        bonds.append((i, j))
    return fnm, bonds


def adj_matrix(bonds):
    """
    Calculates the adjacency matrix given the bond list.
    bonds: list of bonds (tuple of indexes).
    """
    n = len(bonds)
    am = array(zeros((n, n)))
    for i, bond_i in enumerate(bonds):
        for j, bond_j in enumerate(bonds):
            # Ignore the diagonal.
            if i != j:
                if (bond_i[0] in bond_j) or (bond_i[1] in bond_j):
                    am[i, j] = 1

    return am
