"""
## Pande Atomic Conv -- Not completed yet --
"""


import argparse
from rdkit import Chem
import numpy as np
from Featurizer import ComplexFeaturizer
from Timer import Timer

__VERSION__ = '0.1.0'

np.set_printoptions(precision=3, threshold=np.inf, linewidth=300)

parser = argparse.ArgumentParser(description='Convert molecule file')
parser.add_argument('--ligand', '-l', type=str, help='Ligand file')
parser.add_argument('--protein', '-p', type=str, help='Potein file')
args = parser.parse_args()

ligand_fname = args.ligand
protein_fname = args.protein

for mol_ligand in Chem.SDMolSupplier(args.ligand): break
mol_protein = Chem.MolFromPDBFile(args.protein)

print(f'{mol_ligand.GetNumAtoms()}, {mol_protein.GetNumAtoms()}')

timer = Timer()
featurizer = ComplexFeaturizer(M=24)
featurizer(mol_ligand, mol_protein)
lap = timer.elapsed()
R, Z, E, P = featurizer.R, featurizer.Z, featurizer.E, featurizer.P

print(f'Shapes: R:{R.shape}, Z:{Z.shape}, E:{E.shape}, P:{P.shape}')
print(f'{lap:.2f}s')

featurizer.save('out.npz')
