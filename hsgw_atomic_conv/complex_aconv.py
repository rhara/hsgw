"""
Atomic Conv for Complex inspired by Pande
"""
__VERSION__ = '0.2.0'
__AUTHOR__ = 'hara.ryuichiro@gmail.com'

import sys, argparse, time, os, signal
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import multiprocessing as mp

signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(-1))

np.set_printoptions(precision=3, threshold=np.inf, linewidth=300)

class ComplexFeaturizer:
    """
    Default values:
        radials = [1.5 2. 2.5 ... 12.]
        r_s = 1.5
        sigma_s = 1.
        beta = [1. 1. 1. ... 1.]
        bias = [0. 0. 0. ... 0.]
        M = 12
    """

    kwList = ['atomtype_list', 'radial_min', 'radial_max', 'radial_intv', 'r_s', 'sigma_s',
              'beta_init', 'bias_init', 'M']

    attrList = ['atomtype_list', 'radials', 'r_s', 'sigma_s', 'beta_init', 'bias_init',
                'mol_lig', 'mol_pro', 'R', 'Z', 'E', 'P']

    default_atomtypes = [6,7,8,9,11,12,15,16,17,20,25,30,35,53]

    def __init__(self, **kwargs):
        for kw in kwargs:
            assert kw in self.kwList

        self.atomtype_list = kwargs.get('atomtype_list', self.default_atomtypes)
        radial_min = kwargs.get('radial_min', 1.5)
        radial_max = kwargs.get('radial_max', 12.0)
        radial_intv = kwargs.get('radial_intv', .5)
        self.radials = np.arange(radial_min, radial_max+1e-7, radial_intv)
        self.r_s = kwargs.get('r_s', 1.5)
        self.sigma_s = kwargs.get('sigma_s', 1.)
        self.beta_init = kwargs.get('beta_init', 1.)
        self.bias_init = kwargs.get('bias_init', 0.)
        self.M = kwargs.get('M', 12)
        self.beta = None
        self.bias = None
        self.mol_lig = None
        self.mol_pro = None
        self.R = None
        self.Z = None
        self.E = None
        self.P = None

    def __call__(self, mol_ligand, mol_protein):
        """
        Calling this object with a given pair of molecules runs the calculation to generate an
        initial layer in steps;
            1) make R and Z
            2) convolution
            3) radial pooling
        """

        self.mol_lig = mol_ligand
        self.mol_pro = mol_protein

        self.neighborListConstruction()
        self.atomTypeConvolution()
        self.radialPooling()

    def neighborListConstruction(self):
        """
        Make R (distance) and Z (atomtype) of neighboring atoms
        Note: neighbor size is defined as self.M
        """

        atomTypes_lig = np.array([a.GetAtomicNum() for a in self.mol_lig.GetAtoms()])
        atomTypes_pro = np.array([a.GetAtomicNum() for a in self.mol_pro.GetAtoms()])

        N_lig = self.mol_lig.GetNumAtoms()
        N_pro = self.mol_pro.GetNumAtoms()

        conf_lig = self.mol_lig.GetConformer(0)
        coords_lig = conf_lig.GetPositions()

        conf_pro = self.mol_pro.GetConformer(0)
        coords_pro = conf_pro.GetPositions()

        self.R = np.zeros([N_lig, self.M])

        distance_mat = np.zeros([N_lig, N_pro])

        for i in range(N_lig):
            ci = np.tile(coords_lig[i], N_pro).reshape((N_pro, 3))
            distance_mat[i] = np.linalg.norm(ci - coords_pro, axis=1)

        neighbors = np.zeros([N_lig, self.M], dtype=np.int)
        self.Z = np.zeros([N_lig, self.M], dtype=np.int)
        for i in range(N_lig):
            neighbors[i] = np.argsort(distance_mat[i])[:self.M]
            self.R[i] = distance_mat[i, neighbors[i]]
            self.Z[i] = atomTypes_pro[neighbors[i]]

    def atomTypeConvolution(self):
        """
        Do the atom type convolution
        R(N,M), Z(N,M) -> E(N,M,Nat)
        """

        Na = len(self.atomtype_list)
        stacks = []
        for k in range(Na):
            atomtype = self.atomtype_list[k]
            Ka = (self.Z == atomtype).astype(np.int)
            v = self.R*Ka
            stacks.append(v)
        self.E = np.dstack(stacks)

    def radialPooling(self):
        """
        Do the radialPooling
        E(N,M,Nat) -> P(N,Nat,Nr)
        """

        def fc(r, cutoff):
            return (r < cutoff)*(np.cos(np.pi*r/cutoff) + 1)

        def fs(r, cutoff):
            return np.exp(-(r-r_s)*(r-r_s)*fc(r, cutoff)/sigma_s/sigma_s)

        r_s = self.r_s
        sigma_s = self.sigma_s
        beta = np.ones(len(self.radials))*self.beta_init
        bias = np.ones(len(self.radials))*self.bias_init

        N = self.E.shape[0]
        Na = len(self.atomtype_list)
        Nr = len(self.radials)

        self.P = np.zeros([N, Na, Nr])
        for i in range(Nr):
            cutoff = self.radials[i]
            r0_list = np.zeros(Na)
            v0 = beta[i]*np.sum(fs(r0_list, cutoff)) + bias[i]
            for j in range(N):
                for k in range(Na):
                    r_list = self.E[j,:,k]
                    if np.all(r_list == 0.):
                        self.P[j, k, i] = v0
                    else:
                        self.P[j, k, i] = beta[i]*np.sum(fs(r_list, cutoff)) + bias[i]

    def save(self, fname):
        """
        Save attributes in npz format
        """

        np.savez(fname, **vars(self))


def worker(par):
    for ligand in Chem.SDMolSupplier(par.sdfname): break
    protein = Chem.MolFromPDBFile(par.pdbname)

    if not (ligand and protein):
        return par, None

    featurizer = ComplexFeaturizer(M=par.neighbor_size,
                                   radial_min=par.rmin,
                                   radial_max=par.rmax,
                                   radial_intv=par.rint)

    featurizer(ligand, protein)

    return par, featurizer

    # featurizer.save(args.output)


def main():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser(description='Convert pair of bound-complex molecules into '
                                                 'Pande-type atomic conv pooling layer')
    parser.add_argument('basedir', type=str, help='Base directory to find dataset')
    parser.add_argument('--neighbor-size', '-M', type=int, default=12, help='Neighbor size')
    parser.add_argument('--radials-setup', '-R', type=str, default='1.5 12.0 0.5', help='Radials setup')
    parser.add_argument('--output', '-o', type=str, default='out.npz', help='Output file')
    args = parser.parse_args()

    args.rmin, args.rmax, args.rint = map(float, args.radials_setup.split())

    def data_iter():
        for root, dir, fnames in os.walk(args.basedir):
            par = argparse.Namespace(**vars(args))
            SDFNAME = None
            PDBNAME = None
            for fname in fnames:
                if fname.endswith('.sdf'):
                    SDFNAME = fname
                elif fname.endswith('.pdb'):
                    PDBNAME = fname
            if PDBNAME and SDFNAME:
                args.pdbname = os.path.abspath(f'{root}/{PDBNAME}')
                args.sdfname = os.path.abspath(f'{root}/{SDFNAME}')
                args.root = os.path.basename(os.path.abspath(root))
                yield args



    pool = mp.Pool(mp.cpu_count())

    count = 0
    feats = []
    for par, feat in pool.imap_unordered(worker, data_iter()):
        if feat:
            count += 1
            print(f'{count} {par.root} => R:{feat.R.shape}, Z:{feat.Z.shape}, E:{feat.E.shape}, P:{feat.P.shape}')
            feats.append(feat)
        else:
            print(f'[Warning] {par.root} skipped for invalid molecule')

if __name__ == '__main__':
    main()
