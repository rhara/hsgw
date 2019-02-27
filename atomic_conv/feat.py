import sys, os, pickle
import mdtraj
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem.feat import ComplexFeaturizer
from deepchem.utils import rdkit_util, pad_array


class MoleculeLoadException(Exception): pass


def read_mol(fname, add_hydrogens=True, calc_charges=True, sanitize=False):
    if fname.endswith('.mol2'):
        mol = Chem.MolFromMol2File(fname, sanitize=False, removeHs=False)
    elif fname.endswith('.sdf'):
        mol = Chem.SDMolSupplier(fname, sanitize=False)[0]
    elif fname.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(fname, sanitize=False, removeHs=False)
    else:
        raise ValueError('Unrecognized file type')
    if mol is None:
        raise ValueError('Unable to read molecule')
    if add_hydrogens or calc_charges:
        mol = rdkit_util.add_hydrogens_to_mol(mol)
    if sanitize:
        Chem.SanitizeMol(mol)
    if calc_charges:
        try:
            AllChem.ComputeGasteigerCharges(mol)
        except Exception as e:
            raise MoleculeLoadException(e)
    return mol


class Featurizer:
    def __init__(self, frag1_natoms, frag2_natoms, max_nneighbors, neighbor_cutoff):
        self.frag1_natoms = frag1_natoms
        self.frag2_natoms = frag2_natoms
        self.complex_natoms = frag1_natoms + frag2_natoms
        self.max_nneighbors = max_nneighbors
        self.neighbor_cutoff = neighbor_cutoff

    @staticmethod
    def strip_hydrogens(coords, mol):
        class AtomShim:
            def __init__(self, atomic_num):
              self.atomic_num = atomic_num
            def GetAtomicNum(self):
              return self.atomic_num
    
        class MoleculeShim:
            def __init__(self, atoms):
                self.atoms = [AtomShim(x) for x in atoms]
            def GetAtoms(self):
                return self.atoms
    
        indexes_to_keep = []
        atomic_numbers = []
        for index, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() != 1:
                indexes_to_keep.append(index)
                atomic_numbers.append(atom.GetAtomicNum())
        mol = MoleculeShim(atomic_numbers)
        coords = coords[indexes_to_keep]
        return coords, mol

    def compute_neighbor_list(self, coords):
        N = coords.shape[0]
        traj = mdtraj.Trajectory(coords.reshape((1, N, 3)), None)
        neighbors = mdtraj.geometry.compute_neighborlist(traj, self.neighbor_cutoff)
        neighbor_list = {}
        for i in range(N):
            if self.max_nneighbors < len(neighbors[i]):
                delta = coords[i] - coords.take(neighbors[i], axis=0)
                dist = np.linalg.norm(delta, axis=1)
                sorted_neighbors = sorted(zip(dist, neighbors[i]))
                neighbor_list[i] = [sorted_neighbors[j][1] for j in range(self.max_nneighbors)]
            else:
                neighbor_list[i] = list(neighbors[i])
        return neighbor_list

    def featurize_mol(self, coords, mol, max_natoms):
        neighbor_list = self.compute_neighbor_list(coords)
        z = pad_array(np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_natoms)
        z = pad_array(z, max_natoms)
        coords = pad_array(coords, (max_natoms, 3))
        return coords, neighbor_list, z

    def __call__(self, frag1_mol, frag2_mol):
        frag1_coords = rdkit_util.get_xyz_from_mol(frag1_mol)
        frag2_coords = rdkit_util.get_xyz_from_mol(frag2_mol)
        complex_mol = rdkit_util.merge_molecules(frag1_mol, frag2_mol)
        complex_coords = rdkit_util.get_xyz_from_mol(complex_mol)
        
        frag1_coords, frag1_mol = Featurizer.strip_hydrogens(frag1_coords, frag1_mol)
        frag2_coords, frag2_mol = Featurizer.strip_hydrogens(frag2_coords, frag2_mol)
        complex_coords, complex_mol = Featurizer.strip_hydrogens(complex_coords, complex_mol)
        
        frag1_coords, frag1_neighbor_list, frag1_z = self.featurize_mol(frag1_coords, frag1_mol, self.frag1_natoms)
        frag2_coords, frag2_neighbor_list, frag2_z = self.featurize_mol(frag2_coords, frag2_mol, self.frag2_natoms)
        complex_coords, complex_neighbor_list, complex_z = self.featurize_mol(complex_coords, complex_mol, self.complex_natoms)
        return frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords, frag2_neighbor_list, frag2_z, complex_coords, complex_neighbor_list, complex_z


featurizer = Featurizer(frag1_natoms=70, frag2_natoms=24000, max_nneighbors=6, neighbor_cutoff=6.0)

ligand_fname = sys.argv[1]
protein_fname = sys.argv[2]
out_fname = sys.argv[3]

ligand = read_mol(ligand_fname)
protein = read_mol(protein_fname)

# Get features
features = featurizer(ligand, protein)
pickle.dump(features, open(out_fname, 'wb'))

# Print size of feature components
frag1_coords, frag1_neighbor_list, frag1_z, frag2_coords, frag2_neighbor_list, frag2_z, complex_coords, complex_neighbor_list, complex_z = features
print('frag1_coords', frag1_coords.shape)
print('frag1_neighbor_list (dict)', len(frag1_neighbor_list))
print('frag1_z', frag1_z.shape)
print(frag1_z)
print('frag2_coords', frag2_coords.shape)
print('frag2_neighbor_list (dict)', len(frag2_neighbor_list))
print('frag2_z', frag2_z.shape)
print('complex_coords', complex_coords.shape)
print('complex_neighbor_list (dict)', len(complex_neighbor_list))
print('complex_z', complex_z.shape)
