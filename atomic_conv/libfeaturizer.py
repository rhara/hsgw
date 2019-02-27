"""
Atomic coordinate featurizer.
"""

import logging
import numpy as np
import deepchem as dc
from deepchem.feat import Featurizer, ComplexFeaturizer
from deepchem.utils import rdkit_util, pad_array
from deepchem.utils.rdkit_util import MoleculeLoadException


def compute_neighbor_list(coords, neighbor_cutoff, max_num_neighbors, periodic_box_size):
    """Computes a neighbor list from atom coordinates."""
    N = coords.shape[0]
    import mdtraj
    traj = mdtraj.Trajectory(coords.reshape((1, N, 3)), None)
    box_size = None
    if periodic_box_size is not None:
        box_size = np.array(periodic_box_size)
        traj.unitcell_vectors = np.array([[[box_size[0], 0, 0], [0, box_size[1], 0], [0, 0, box_size[2]]]], dtype=np.float32)
    neighbors = mdtraj.geometry.compute_neighborlist(traj, neighbor_cutoff)
    neighbor_list = {}
    for i in range(N):
        if max_num_neighbors is not None and len(neighbors[i]) > max_num_neighbors:
          delta = coords[i] - coords.take(neighbors[i], axis=0)
          if box_size is not None:
              delta -= np.round(delta / box_size) * box_size
          dist = np.linalg.norm(delta, axis=1)
          sorted_neighbors = list(zip(dist, neighbors[i]))
          sorted_neighbors.sort()
          neighbor_list[i] = [sorted_neighbors[j][1] for j in range(max_num_neighbors)]
        else:
          neighbor_list[i] = list(neighbors[i])
    return neighbor_list


class ComplexNeighborListFragmentAtomicCoordinates(ComplexFeaturizer):
    """This class computes the featurization that corresponds to AtomicConvModel.

    This class computes featurizations needed for AtomicConvModel. Given a
    two molecular structures, it computes a number of useful geometric
    features. In particular, for each molecule and the global complex, it
    computes a coordinates matrix of size (N_atoms, 3) where N_atoms is the
    number of atoms. It also computes a neighbor-list, a dictionary with
    N_atoms elements where neighbor-list[i] is a list of the atoms the i-th
    atom has as neighbors. In addition, it computes a z-matrix for the
    molecule which is an array of shape (N_atoms,) that contains the atomic
    number of that atom.

    Since the featurization computes these three quantities for each of the
    two molecules and the complex, a total of 9 quantities are returned for
    each complex. Note that for efficiency, fragments of the molecules can be
    provided rather than the full molecules themselves.
    """

    def __init__(self, frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors, neighbor_cutoff, strip_hydrogens=True):
        print('@libfeaturizer.ComplexNeighborListFragmentAtomicCoordinates')
        self.frag1_num_atoms = frag1_num_atoms
        self.frag2_num_atoms = frag2_num_atoms
        self.complex_num_atoms = complex_num_atoms
        self.max_num_neighbors = max_num_neighbors
        self.neighbor_cutoff = neighbor_cutoff
        self.strip_hydrogens = strip_hydrogens

    def _featurize_complex(self, mol_pdb_file, protein_pdb_file):
        print('# read', mol_pdb_file, protein_pdb_file)
        try:
            frag1_coords, frag1_mol = rdkit_util.load_molecule(mol_pdb_file)
            frag2_coords, frag2_mol = rdkit_util.load_molecule(protein_pdb_file)
        except MoleculeLoadException:
            # Currently handles loading failures by returning None
            # TODO: Is there a better handling procedure?
            logging.warning("Some molecules cannot be loaded by Rdkit. Skipping")
            return None
        system_mol = rdkit_util.merge_molecules(frag1_mol, frag2_mol)
        system_coords = rdkit_util.get_xyz_from_mol(system_mol)

        frag1_coords, frag1_mol = self._strip_hydrogens(frag1_coords, frag1_mol)
        frag2_coords, frag2_mol = self._strip_hydrogens(frag2_coords, frag2_mol)
        system_coords, system_mol = self._strip_hydrogens(system_coords, system_mol)

        try:
            frag1_coords, frag1_neighbor_list, frag1_z = self.featurize_mol(frag1_coords, frag1_mol, self.frag1_num_atoms)
            frag2_coords, frag2_neighbor_list, frag2_z = self.featurize_mol(frag2_coords, frag2_mol, self.frag2_num_atoms)
            system_coords, system_neighbor_list, system_z = self.featurize_mol(system_coords, system_mol, self.complex_num_atoms)
        except ValueError as e:
            logging.warning("max_atoms was set too low. Some complexes too large and skipped")
            return None

        return frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords, frag2_neighbor_list, frag2_z, system_coords, system_neighbor_list, system_z

    def get_Z_matrix(self, mol, max_atoms):
        if len(mol.GetAtoms()) > max_atoms:
            raise ValueError("A molecule is larger than permitted by max_atoms. Increase max_atoms and try again.")
        return pad_array(np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_atoms)

    def featurize_mol(self, coords, mol, max_num_atoms):
        logging.info("Featurizing molecule of size: %d", len(mol.GetAtoms()))
        neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff, self.max_num_neighbors, None)
        z = self.get_Z_matrix(mol, max_num_atoms)
        z = pad_array(z, max_num_atoms)
        coords = pad_array(coords, (max_num_atoms, 3))
        return coords, neighbor_list, z

    def _strip_hydrogens(self, coords, mol):
        class MoleculeShim(object):
            """
            Shim of a Molecule which supports #GetAtoms()
            """
            def __init__(self, atoms):
                self.atoms = [AtomShim(x) for x in atoms]

            def GetAtoms(self):
                return self.atoms

        class AtomShim(object):
            def __init__(self, atomic_num):
              self.atomic_num = atomic_num

            def GetAtomicNum(self):
              return self.atomic_num

        if not self.strip_hydrogens:
            return coords, mol
        indexes_to_keep = []
        atomic_numbers = []
        for index, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() != 1:
                indexes_to_keep.append(index)
                atomic_numbers.append(atom.GetAtomicNum())
        mol = MoleculeShim(atomic_numbers)
        coords = coords[indexes_to_keep]
        return coords, mol
