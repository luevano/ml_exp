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
from numpy import array, zeros, dot
from numpy.linalg import norm
from ml_exp.misc import printc


def fneig_matrix(atoms,
                 xyz,
                 nc=None,
                 use_forces=False):
    """
    Creates the first neighbor matrix of the given molecule data.
    atoms: list of atoms.
    xyz: matrix of atomic coords.
    nc: nuclear charge info.
    use_forces: if the use of forces instead of k_cx should be used.
    NOTE: Bond distance of carbon to other elements
        are (for atoms present in the qm7 dataset):
            C: 1.20 - 1.53 A
            H: 1.06 - 1.12 A
            O: 1.43 - 2.15 A
            N: 1.47 - 2.10 A
            S: 1.81 - 2.55 A
    """
    if use_forces and nc is None:
        printc(f'Error. use_forces={use_forces} but nc={nc}', 'RED')
        return None
    # Possible bonds.
    cc_bond = sorted(['C', 'C'])
    ch_bond = sorted(['C', 'H'])
    co_bond = sorted(['C', 'O'])
    cn_bond = sorted(['C', 'N'])
    cs_bond = sorted(['C', 'S'])

    # Number of atoms, empty matrix and bond list.
    n = len(atoms)
    fnm = array(zeros((n, n)), dtype=float)
    bonds = []
    forces = []
    for i, xyz_i in enumerate(xyz):
        for j, xyz_j in enumerate(xyz):
            # Ignore the diagonal.
            if i != j:
                bond = sorted([atoms[i], atoms[j]])
                rv = xyz_i - xyz_j
                r = norm(rv)
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
    if use_forces:
        return fnm, bonds, forces
    return fnm, bonds


def adj_matrix(fneig_matrix,
               bonds,
               forces=None,
               max_len=25):
    """
    Calculates the adjacency matrix given the bond list.
    bonds: list of bonds (tuple of indexes).
    max_len: maximum amount of atoms in molecule.
    forces: list of forces.
    """
    n = len(bonds)

    if max_len < n:
        print(''.join(['Error. Molecule matrix dimension (mol_n) is ',
                       'greater than max_len. Using mol_n.']))
        max_len = n

    am = array(zeros((max_len, max_len)))
    for i, bond_i in enumerate(bonds):
        for j, bond_j in enumerate(bonds):
            # Ignore the diagonal.
            if i != j:
                if (bond_i[0] in bond_j) or (bond_i[1] in bond_j):
                    if forces:
                        am[i, j] = dot(forces[i], forces[j])
                    else:
                        am[i, j] = fneig_matrix[bond_i[0], bond_i[1]]
    return am
