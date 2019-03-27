"""
## Pande Atomic Conv -- Not completed yet --
"""


import sys, os, math, time
import argparse
from rdkit import Chem
import numpy as np
# import pandas as pd

__VERSION__ = '0.1.3'

np.set_printoptions(precision=3, threshold=np.inf, linewidth=np.inf)


parser = argparse.ArgumentParser(description='Convert molecule file')
parser.add_argument('--ligand', '-l', type=str, help='Ligand file')
parser.add_argument('--protein', '-p', type=str, help='Potein file')
args = parser.parse_args()


"""
%load_ext autoreload
%autoreload 2
"""


"""
### Requires PandeAtomicConv.py
"""


from PandeAtomicConv import Featurizer
import numpy as np


"""
### Read mols
"""

from rdkit import Chem

ligand_fname = args.ligand
protein_fname = args.protein

for mol_ligand in Chem.SDMolSupplier(ligand_fname): break
mol_protein = Chem.MolFromPDBFile(protein_fname)
mol_complex = Chem.CombineMols(mol_ligand, mol_protein)

f'{mol_ligand.GetNumAtoms()}, {mol_protein.GetNumAtoms()}, {mol_complex.GetNumAtoms()}'


"""
### Setup featurizers

Calculate neighbor list of size (N, M) (M=12 as the default) in each molecule
"""

nmax_ligand = 70
nmax_protein = 12000
nmax_complex = nmax_ligand + nmax_protein

featurizer_L = Featurizer(mol_ligand, M=12)
featurizer_P = Featurizer(mol_protein, M=12)
featurizer_C = Featurizer(mol_complex, M=12)


"""
### Calculate convolution

Calculate R (neighbored distance matrix), Z (neighbored atom types) and E (atomic convolution). E is padded to the given size.
"""

featurizer_L.setConvolution(padding=nmax_ligand)
featurizer_P.setConvolution(padding=nmax_protein)
featurizer_C.setConvolution(padding=nmax_complex)

f'Atomic convolution E.shape: L:{featurizer_L.E.shape}, P:{featurizer_P.E.shape}, C:{featurizer_C.E.shape}'


"""
### Radial pooling layer

Calculate P (Radial pooling layer). radials are given as [1.5, 12.0] step 0.5. $r_s=1.5, \sigma_s = 1.0$ are scalar constants. $\beta_{n_r}$ (scaling) and $b_{n_r}$ (bias) are given as all 1.0 and all 0.0 respectively. These must be elucidated anyway.
"""

radials = np.arange(1.5, 12.+1e-7, .5)
radials


r_s, sigma_s = 1.5, 1.
beta = [1.]*len(radials)
bias = [0.]*len(radials)

featurizer_L.setRadialLayer(beta=beta, bias=bias, r_s=r_s, sigma_s=sigma_s, radials=radials)
featurizer_P.setRadialLayer(beta=beta, bias=bias, r_s=r_s, sigma_s=sigma_s, radials=radials)
featurizer_C.setRadialLayer(beta=beta, bias=bias, r_s=r_s, sigma_s=sigma_s, radials=radials)

print(f'Radial pooling layer P.shape: L:{featurizer_L.P.shape}, P:{featurizer_P.P.shape}, C:{featurizer_C.P.shape}')

