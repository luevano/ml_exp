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
from collections import Counter
from ml_exp.data import POSSIBLE_BONDS


def coulomb_matrix(coords,
                   nc,
                   size=23,
                   sort=False,
                   flatten=True,
                   as_eig=True,
                   bohr_ru=False):
    """
    Creates the Coulomb Matrix from the molecule data given.
    coords: compound coordinates.
    nc: nuclear charge data.
    size: compound size.
    sort: if the representation should be sorted row-norm-wise.
    flatten: if the representation should be 1D.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    """
    if bohr_ru:
        cr = 0.52917721067
    else:
        cr = 1.0

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge \
size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    cm = np.zeros((n, n), dtype=np.float64)

    # Actual calculation of the coulomb matrix.
    for i in range(n):
        cm[i, i] = 0.5*nc[i]**2.4

    # Calculates the values row-wise for faster timings.
    # Don't calculate the last element (it's only the diagonal element).
    for i in range(n - 1):
        rv = coords[i + 1:] - coords[i]
        r = np.linalg.norm(rv, axis=1)/cr
        val = nc[i]*nc[i +1:]/r
        cm[i, i + 1:] = val
        cm[i + 1:, i] = val

    # Now the value will be returned.
    if as_eig:
        cm_eigs = np.sort(np.linalg.eig(cm)[0])[::-1]

        return np.pad(cm_eigs, (0, size - n), 'constant')
    else:
        if sort:
            si = np.argsort(np.linalg.norm(cm, axis=-1))[::-1]
            cm = cm[si]

        if flatten:
            return np.pad(cm, ((0, size - n), (0, size - n)),
                          'constant').flatten()
        else:
            return np.pad(cm, ((0, size - n), (0, size - n)), 'constant')


def lennard_jones_matrix(coords,
                         nc,
                         diag_value=None,
                         sigma=1.0,
                         epsilon=1.0,
                         size=23,
                         sort=False,
                         flatten=True,
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
    sort: if the representation should be sorted row-norm-wise.
    flatten: if the representation should be 1D.
    as_eig: if the representation should be as the eigenvalues.
    bohr_ru: if radius units should be in bohr's radius units.
    """
    if bohr_ru:
        cr = 0.52917721067
    else:
        cr = 1.0

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge \
size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    lj = np.zeros((n, n), dtype=np.float64)

    # Actual calculation of the lennard-jones matrix.
    for i in range(n):
        if diag_value is None:
            lj[i, i] = 0.5*nc[i]**2.4
        else:
            lj[i, i] = diag_value

    # Calculates the values row-wise for faster timings.
    # Don't calculate the last element (it's only the diagonal element).
    for i in range(n - 1):
        rv = coords[i + 1:] - coords[i]
        r = (sigma*cr)/np.linalg.norm(rv, axis=1)

        # 1/r^n
        r_6 = r**6
        r_12 = r**12
        val = (4*epsilon*(r_12 - r_6))
        lj[i, i + 1:] = val
        lj[i + 1:, i] = val

    # Now the value will be returned.
    if as_eig:
        lj_eigs = np.sort(np.linalg.eig(lj)[0])[::-1]

        return np.pad(lj_eigs, (0, size - n), 'constant')
    else:
        if sort:
            si = np.argsort(np.linalg.norm(lj, axis=-1))[::-1]
            lj = lj[si]

        if flatten:
            return np.pad(lj, ((0, size - n), (0, size - n)),
                          'constant').flatten()
        else:
            return np.pad(lj, ((0, size - n), (0, size - n)), 'constant')


def get_helping_data(coords,
                     atoms,
                     nc,
                     size=23,
                     bohr_ru=False):
    """
    Creates helping data such as the First Neighbor Matrix for the compound.
    coords: compound coordinates.
    atoms: list of atoms.
    nc: nuclear charge data.
    size: compund size.
    bohr_ru: if radius units should be in bohr's radius units.
    """
    if bohr_ru:
        cr = 0.52917721067
    else:
        cr = 1.0

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge \
size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    fnm = np.zeros((n, n), dtype=bool)
    bonds = []
    bonds_i = []
    bonds_k = []
    bonds_f = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            bond = ''.join(sorted([atoms[i], atoms[j]]))
            if bond in POSSIBLE_BONDS.keys():
                r_min = POSSIBLE_BONDS[bond][0]
                r_max = POSSIBLE_BONDS[bond][1]
                rv = coords[i] - coords[j]
                r = np.linalg.norm(rv)/cr
                if r >= r_min and r <= r_max:
                    fnm[i, j] = True
                    fnm[j, i] = True
                    bonds.append(bond)
                    bonds_i.append((i, j))
                    bonds_k.append(POSSIBLE_BONDS[bond][2])
                    bonds_f.append(rv*nc[i]*nc[j]/r**3)

    fnm = np.pad(fnm, ((0, size - n), (0, size - n)), 'constant')

    return fnm, bonds, bonds_i, bonds_k, bonds_f


def adjacency_matrix(bonds_i,
                     bonds_k,
                     bonds_f,
                     use_forces=False,
                     size=23,
                     sort=False,
                     flatten=True):
    """
    Calculates the adjacency matrix given the bond list.
    bonds: list of bond names.
    bonds_i: list of bond indexes (tuple of indexes).
    bonds_k: list of k_cx values.
    bonds_f: list of force values.
    use_forces: if the use of forces instead of k_cx should be used.
    size: compund size.
    sort: if the representation should be sorted row-norm-wise.
    flatten: if the representation should be 1D.
    """
    if bonds_i is None:
        raise ValueError('The helping data hasn\'t been initialized for\
the current compound.')

    n = len(bonds_i)
    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)\
              instead of (size).')
        size = n

    am = np.zeros((n, n), dtype=np.float64)

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (bonds_i[i][0] in bonds_i[j]) or (bonds_i[i][1] in bonds_i[j]):
                if use_forces:
                    am[i, j] = np.dot(bonds_f[i], bonds_f[j])
                    am[j, i] = am[i, j]
                else:
                    am[i, j] = bonds_k[i]
                    am[j, i] = am[i, j]

    if sort:
        si = np.argsort(np.linalg.norm(am, axis=-1))
        am = am[si]

    if flatten:
        return np.pad(am, ((0, size - n), (0, size - n)), 'constant').flatten()
    else:
        return np.pad(am, ((0, size - n), (0, size - n)), 'constant')


def epsilon_index(am,
                  bonds_i,
                  size=23):
    """
    Calculates the Epsilon index of G, presented by Estrada.
    am: adjacency matrix.
    bonds_i: list of bond indexes (tuple of indexes).
    size: compund size.
    """
    if am is None:
        raise ValueError('The adjacency matrix hasn\'t been initialized for\
the current compound.')

    n = len(bonds_i)
    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)\
              instead of (size).')
        size = n

    deltas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        deltas[i] = np.sum(am[i, :])

    ei = 0.0
    for i in range(n - 1):
        for j in range(i +1, n):
            if (bonds_i[i][0] in bonds_i[j]) or (bonds_i[i][0] in bonds_i[j]):
                val = deltas[i]*deltas[j]
                ei += 1.0/val**0.5

    return ei


def bag_of_bonds(cm,
                 atoms,
                 sort=False,
                 acount={'C':7, 'H':16, 'N':3, 'O':3, 'S':1}):
    """
    Creates the Bag of Bonds using the Coulomb Matrix.
    cm: coulomb matrix.
    atoms: list of atoms.
    sort: if the representation should be sorted bag-wise.
    acount: atom count for the compound, defaults to qm7 sizes.
    NOTE: 'cm' shouldn't be sorted by row-norm since 'atoms' isn't (sorted).
    """
    if cm is None:
        raise ValueError('Coulomb Matrix hasn\'t been initialized for the \
current compound.')

    if cm.ndim == 1 and cm.shape[0] < 30:
        raise ValueError('CM was generated as the vector of eigenvalues. \
Use non-eigenvalue representation.')

    # Base bags.
    ackeys = list(acount.keys())
    bags = dict()
    for i, atom_i in enumerate(ackeys):
        for j, atom_j in enumerate(ackeys[i:]):
            # Add current bond to bags.
            if j == 0:
                bags[atom_i] = [acount[atom_i], []]
                if acount[atom_i] > 1:
                    bsize = np.int32((acount[atom_i]**2 - acount[atom_i])/2)
                    bags[''.join(sorted([atom_i, atom_j]))] = [bsize, []]
            else:
                bags[''.join(sorted([atom_i, atom_j]))] = [acount[atom_i] *
                                                           acount[atom_j], []]

    bond_size = 0
    for b in bags.keys():
        bond_size += bags[b][0]

    # Adding actual values to the bags.
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            # Operate on the upper triangular matrix and get the current bond.
            if j >= i:
                if j == i:
                    bag = atom_i
                else:
                    bag = ''.join(sorted([atom_i, atom_j]))

                bags[bag][1].append(cm[i, j])

    # Change to a numpy array and add padding.
    for bag in bags.keys():
        if sort:
            b = np.sort(np.array(bags[bag][1]))[::-1]
        else:
            b = np.array(bags[bag][1])
        b = np.pad(b, (0, bags[bag][0] - b.shape[0]), 'constant')
        bags[bag][1] = b

    return np.concatenate([bags[bag][1] for bag in bags.keys()])
