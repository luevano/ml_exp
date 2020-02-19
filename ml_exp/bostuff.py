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
from collections import Counter
from ml_exp.bob import check_bond


def bok_cx(adj_matrix,
           atoms,
           max_n=25,
           max_bond_len=325):
    """
    Creates the bag of k_cx using the adjacency matrix data.
    adj_matrix: adjacency matrix.
    max_n: maximum amount of atoms.
    max_bond_len: maximum amount of bonds in molecule.
    """
    n = len(atoms)
    bond_n = (n * n - n) / 2 + n
    n_r = range(n)

    if max_n < n:
        print(''.join(['Error. Molecule matrix dimension (mol_n) is ',
                       'greater than max_len. Using mol_n.']))
        max_n = n

    if max_bond_len < bond_n:
        print(''.join(['Error. Molecule bond length (bond_n) is ',
                       'greater than max_bond_len. Using bond_n.']))
        max_bond_len = bond_n

    # List where each bag data is stored.
    bags = []
    for i in n_r:
        for j in n_r:
            # Work only in the upper triangle of the coulomb matrix.
            if j >= i:
                # Get the string of the current bond.
                if i == j:
                    current_bond = atoms[i]
                else:
                    current_bond = ''.join(sorted([atoms[i], atoms[j]]))

                # Check if that bond is already in a bag.
                checker = check_bond(bags, current_bond)
                # Either create a new bag or add values to an existing one.
                if not checker[0]:
                    bags.append([current_bond, adj_matrix[i, j]])
                else:
                    bags[checker[1]].append(adj_matrix[i, j])

    # Create the actual bond list ordered.
    atom_counter = Counter(atoms)
    atom_list = sorted(list(set(atoms)))
    bonds = []
    for i, a_i in enumerate(atom_list):
        if atom_counter[a_i] > 1:
            for a_j in atom_list[i:]:
                bonds.append(''.join(sorted([a_i, a_j])))
    bonds = atom_list + bonds

    # Create the final vector for the bok_cx.
    bok_cx = array(zeros(max_bond_len), dtype=float)
    c_i = 0
    for i, bond in enumerate(bonds):
        checker = check_bond(bags, bond)
        if checker[0]:
            for j, num in enumerate(sorted(bags[checker[1]][1:])[::-1]):
                # Use c_i as the index for bok_cx if the zero padding should
                # be at the end of the vector instead of between each bond.
                bok_cx[i*max_n + j] = num
                c_i += 1
        # This is set to false because this was a debugging measure.
        else:
            print(''.join([f'Error. Bond {bond} from bond list coudn\'t',
                           ' be found in the bags list. This could be',
                           ' a case where the atom is only present once',
                           ' in the molecule.']))
    return bok_cx
