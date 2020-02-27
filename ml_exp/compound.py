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
from ml_exp.data import NUCLEAR_CHARGE
from ml_exp.representations import coulomb_matrix, lennard_jones_matrix,\
    first_neighbor_matrix, adjacency_matrix, bag_of_stuff


class Compound:
    def __init__(self,
                 xyz=None):
        """
        Initialization of the Compound.
        xyz: (path to) the xyz file.
        """
        self.name = None

        self.n = None
        self.extra = None  # In case a comment is in the compound file.
        self.atoms = None
        self.atoms_nc = None
        self.coordinates = None
        self.pbe0 = None
        self.delta = None

        self.cm = None
        self.ljm = None
        self.am = None
        self.bos = None

        if xyz is not None:
            self.read_xyz(xyz)

    def gen_cm(self,
               size=23,
               as_eig=True,
               bohr_ru=False):
        """
        Generate the Coulomb Matrix for the compund.
        size: compound size.
        as_eig: if the representation should be as the eigenvalues.
        bohr_ru: if radius units should be in bohr's radius units.
        """
        self.cm = coulomb_matrix(self.coordinates,
                                 self.atoms_nc,
                                 size=size,
                                 as_eig=as_eig,
                                 bohr_ru=bohr_ru)

    def gen_ljm(self,
                diag_value=None,
                sigma=1.0,
                epsilon=1.0,
                size=23,
                as_eig=True,
                bohr_ru=False):
        """
        Generate the Lennard-Jones Matrix for the compund.
        diag_value: if special diagonal value is to be used.
        sigma: sigma value.
        epsilon: epsilon value.
        size: compound size.
        as_eig: if the representation should be as the eigenvalues.
        bohr_ru: if radius units should be in bohr's radius units.
        """
        self.ljm = lennard_jones_matrix(self.coordinates,
                                        self.atoms_nc,
                                        diag_value=diag_value,
                                        sigma=sigma,
                                        epsilon=epsilon,
                                        size=size,
                                        as_eig=as_eig,
                                        bohr_ru=bohr_ru)

    def gen_am(self,
               use_forces=False,
               size=23,
               bohr_ru=False):
        """
        Generate the Adjacency Matrix for the compund.
        use_forces: if the use of forces instead of k_cx should be used.
        size: compound size.
        bohr_ru: if radius units should be in bohr's radius units.
        """
        # First, generate the first neighor matrix.
        fnm, bonds, forces = first_neighbor_matrix(self.coordinates,
                                                   self.atoms_nc,
                                                   self.atoms,
                                                   size=size,
                                                   use_forces=use_forces,
                                                   bohr_ru=bohr_ru)

        # Now, generate the adjacency matrix.
        self.am = adjacency_matrix(fnm,
                                   bonds,
                                   forces,
                                   size=size)

    def gen_bos(self,
                size=23,
                stuff='bonds'):
        """
        Generate the Bag of Stuff for the compound.
        size: compound size.
        stuff: elements of the bag, by default the known bag of bonds.
        """
        self.bos = bag_of_stuff(self.cm,
                                self.atoms,
                                size=size,
                                stuff=stuff)

    def read_xyz(self,
                 filename):
        """
        Reads an xyz file and adds the corresponding data to the Compound.
        filename: (path to) the xyz file.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.name = filename.split('/')[-1]
        self.n = int(lines[0])
        self.extra = lines[1]
        self.atoms = []
        self.atoms_nc = np.empty(self.n, dtype=int)
        self.coordinates = np.empty((self.n, 3), dtype=float)

        for i, atom in enumerate(lines[2:self.n + 2]):
            atom_d = atom.split()

            self.atoms.append(atom_d[0])
            self.atoms_nc[i] = NUCLEAR_CHARGE[atom_d[0]]
            self.coordinates[i] = np.asarray(atom_d[1:4], dtype=float)
