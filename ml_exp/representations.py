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


def check_bond(bags,
               bond):
    """
    Checks if a bond is in a bag.
    bags: list of bags, containing a bag per entry, which in turn
        contains a list of bond-values.
    bond: bond to check.
    """
    if bags == []:
        return False, None

    for i, bag in enumerate(bags):
        if bag[0] == bond:
            return True, i

    return False, None


def bag_of_bonds(cm,
                 atoms,
                 size=23):
    """
    Creates the Bag of Bonds using the Coulomb Matrix.
    cm: coulomb matrix.
    atoms: list of atoms.
    size: compound size.
    """
    if cm is None:
        raise ValueError('Coulomb Matrix hasn\'t been initialized for the \
current compound.')

    if cm.ndim == 1:
        raise ValueError('Coulomb Matrix (CM) dimension is 1. Maybe it was \
generated as the vector of eigenvalues, try (re-)generating the CM.')

    n = len(atoms)

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    # Bond max length, calculated using only the upper triangular matrix.
    bond_size = np.int32((size * size - size)/2 + size)

    # List where each bag data is stored.
    bags = []
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            # Work only in the upper triangle of the coulomb matrix.
            if j >= i:
                # Get the string of the current bond.
                if i == j:
                    current_bond = atom_i
                else:
                    current_bond = ''.join(sorted([atom_i, atom_j]))

                # Check if that bond is already in a bag.
                checker = check_bond(bags, current_bond)
                # Either create a new bag or add values to an existing one.
                if not checker[0]:
                    bags.append([current_bond, cm[i, j]])
                else:
                    bags[checker[1]].append(cm[i, j])

    # Create the actual bond list ordered.
    atom_counter = Counter(atoms)
    atom_list = sorted(list(set(atoms)))
    bonds = []
    for i, a_i in enumerate(atom_list):
        if atom_counter[a_i] > 1:
            for a_j in atom_list[i:]:
                bonds.append(''.join(sorted([a_i, a_j])))
    bonds = atom_list + bonds

    # Create the final vector for the bob.
    bob = np.zeros(bond_size, dtype=np.float64)
    c_i = 0
    for i, bond in enumerate(bonds):
        checker = check_bond(bags, bond)
        if checker[0]:
            for j, num in enumerate(sorted(bags[checker[1]][1:])[::-1]):
                # Use c_i as the index for bob if the zero padding should
                # be at the end of the vector instead of between each bond.
                # bob[i*size + j] = num
                bob[c_i] = num
                c_i += 1
        else:
            print(f'Error. Bond {bond} from bond list coudn\'t be found',
                  'in the bags list. This could be a case where the atom',
                  'is only present oncce in the molecule.')

    return bob
