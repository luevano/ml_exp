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
    get_helping_data, adjacency_matrix, bag_of_bonds


class Compound:
    def __init__(self,
                 xyz=None):
        """
        Initialization of the Compound.
        xyz: (path to) the xyz file.
        """
        # General compound data.
        self.name = None
        self.n = None
        self.extra = None
        self.atoms = None
        self.nc = None
        self.coordinates = None
        self.pbe0 = None
        self.delta = None

        # Computed data.
        self.cm = None
        self.ljm = None
        self.am = None
        self.bob = None
        self.bo_atoms = None
        self.bok_cx = None
        self.bof = None

        # Helping data.
        self.fnm = None
        self.bonds = None
        self.bonds_i = None
        self.bonds_k = None
        self.bonds_f = None

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
                                 self.nc,
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
                                        self.nc,
                                        diag_value=diag_value,
                                        sigma=sigma,
                                        epsilon=epsilon,
                                        size=size,
                                        as_eig=as_eig,
                                        bohr_ru=bohr_ru)

    def gen_hd(self,
               size=23,
               bohr_ru=False):
        """
        Generate the helping data for use in Adjacency Matrix, for example.
        size: compund size.
        bohr_ru: if radius units should be in bohr's radius units.
        """
        hd = get_helping_data(self.coordinates,
                              self.atoms,
                              self.nc,
                              size=size,
                              bohr_ru=bohr_ru)

        self.fnm, self.bonds, self.bonds_i, self.bonds_k, self.bonds_f = hd

    def gen_am(self,
               use_forces=False,
               size=23):
        """
        Generate the Adjacency Matrix for the compund.
        use_forces: if the use of forces instead of k_cx should be used.
        size: compound size.
        """
        self.am = adjacency_matrix(self.bonds_i,
                                   self.bonds_k,
                                   self.bonds_f,
                                   use_forces=use_forces,
                                   size=size)

    def gen_bob(self,
                size=23):
        """
        Generate the Bag of Stuff for the compound.
        size: compound size.
        """
        self.bob = bag_of_bonds(self.cm,
                                self.atoms,
                                size=size)

    def read_xyz(self,
                 filename):
        """
        Reads an xyz file and adds the corresponding data to the Compound.
        filename: (path to) the xyz file.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        self.name = filename.split('/')[-1]
        self.n = np.int32(lines[0])
        self.extra = lines[1]
        self.atoms = []
        self.nc = np.empty(self.n, dtype=np.int64)
        self.coordinates = np.empty((self.n, 3), dtype=np.float64)

        for i, atom in enumerate(lines[2:self.n + 2]):
            atom_d = atom.split()

            self.atoms.append(atom_d[0])
            self.nc[i] = NUCLEAR_CHARGE[atom_d[0]]
            self.coordinates[i] = np.asarray(atom_d[1:4], dtype=np.float64)
