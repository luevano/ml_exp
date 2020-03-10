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
        cr = 1.0

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge \
size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    cm = np.zeros((size, size), dtype=np.float64)

    # Actual calculation of the coulomb matrix.
    for i in range(n):
        cm[i, i] = 0.5*nc[i]**2.4

    # Calculates the values row-wise for faster timings.
    for i in range(n):
        rv = coords[i + 1:] - coords[i]
        r = np.linalg.norm(rv, axis=1)/cr
        val = nc[i]*nc[i +1:]/r
        cm[i, i + 1:n] = val
        cm[i + 1:n, i] = val

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
        cr = 1.0

    n = coords.shape[0]

    if not n == nc.shape[0]:
        raise ValueError('Compound size is different than the nuclear charge \
size. Arrays are not of the right shape.')

    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)',
              'instead of (size).')
        size = n

    lj = np.zeros((size, size), dtype=np.float64)

    # Actual calculation of the lennard-jones matrix.
    for i, xyz_i in enumerate(coords):
        for j, xyz_j in enumerate(coords):
            if i == j:
                if diag_value is None:
                    lj[i, j] = (0.5*nc[i]**2.4)
                else:
                    lj[i, j] = diag_value
            else:
                # Calculations are done after i==j is checked
                # so no division by zero is done.

                # A little play with r exponents
                # so no square root is calculated.
                # Conversion factor is included in r^2.
                rv = xyz_i - xyz_j
                r = np.linalg.norm(rv)/cr

                # 1/r^n
                r_2 = sigma**2/r**2
                r_6 = r_2**3
                r_12 = r_6**2
                lj[i, j] = (4*epsilon*(r_12 - r_6))

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
    NOTE: Bond distance of carbon to other elements
        are (for atoms present in the qm7 dataset):
            C: 1.19 - 1.54 A, 1.0
            H: 1.06 - 1.12 A, 1.0
            O: 1.43 - 2.15 A, 0.8
            N: 1.47 - 2.10 A, 1.0
            S: 1.81 - 2.55 A, 0.7
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

    # Possible bonds.
    cc_bond = ''.join(sorted(['C', 'C']))
    ch_bond = ''.join(sorted(['C', 'H']))
    co_bond = ''.join(sorted(['C', 'O']))
    cn_bond = ''.join(sorted(['C', 'N']))
    cs_bond = ''.join(sorted(['C', 'S']))
    pos_bonds = {cc_bond: (1.19, 1.54, 1.0), ch_bond: (1.06, 1.12, 1.0),
                 co_bond: (1.43, 2.15, 0.8), cn_bond: (1.47, 2.19, 1.0),
                 cs_bond: (1.81, 2.55, 0.7)}

    fnm = np.zeros((size, size), dtype=bool)
    bonds = []
    bonds_i = []
    bonds_k = []
    bonds_f = []
    for i, xyz_i in enumerate(coords):
        for j, xyz_j in enumerate(coords):
            # Ignore the diagonal.
            if i != j:
                bond = ''.join(sorted([atoms[i], atoms[j]]))
                if bond in pos_bonds.keys():
                    r_min = pos_bonds[bond][0]
                    r_max = pos_bonds[bond][1]
                    rv = xyz_i - xyz_j
                    r = np.linalg.norm(rv)/cr
                    if r >= r_min and r <= r_max:
                        fnm[i, j] = True
                        # Only add to the list if in the upper triangle.
                        if j > i:
                            bonds.append(bond)
                            bonds_i.append((i, j))
                            bonds_k.append(pos_bonds[bond][2])
                            bonds_f.append(rv*nc[i]*nc[j]/r**3)

    return fnm, bonds, bonds_i, bonds_k, bonds_f


def adjacency_matrix(bonds_i,
                     bonds_k,
                     bonds_f,
                     use_forces=False,
                     size=23):
    """
    Calculates the adjacency matrix given the bond list.
    bonds: list of bond names.
    bonds_i: list of bond indexes (tuple of indexes).
    bonds_k: list of k_cx values.
    bonds_f: list of force values.
    use_forces: if the use of forces instead of k_cx should be used.
    size: compund size.
    """
    if bonds_i is None:
        raise ValueError('The helping data hasn\'t been initialized for\
the current compound.')

    n = len(bonds_i)
    if size < n:
        print('Error. Compound size (n) is greater han (size). Using (n)\
              instead of (size).')
        size = n

    am = np.zeros((size, size), dtype=np.float64)

    for i, bond_i in enumerate(bonds_i):
        for j, bond_j in enumerate(bonds_i):
            # Ignore the diagonal.
            if i != j:
                if (bond_i[0] in bond_j) or (bond_i[1] in bond_j):
                    if use_forces:
                        am[i, j] = np.dot(bonds_f[i], bonds_f[j])
                    else:
                        am[i, j] = bonds_k[i]

    return am


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
