import sys, os, math, time
import argparse
from rdkit import Chem
import numpy as np

__VERSION__ = '0.1.2'

np.set_printoptions(precision=3, threshold=np.inf, linewidth=200)

class AnalyticMol:
    def __init__(self, mol, **kwargs):
        """
        Initialize mol for featuring

        Ordered args:
            mol
        Keyword args:
            confidx
            atomtypes
            radials
            beta
            bias
            M
        Attributes:
            mol
            atomtypes
            radials
            beta
            bias
            M
            distance_mat
        """
        self.mol = mol
        self.atomtypes = kwargs.get('atomtypes',
                                     [6,7,8,9,11,12,15,16,17,20,25,30,35,53])
        self.radials = kwargs.get('radials',
                                  [1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,
                                   7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0])
        self.beta = kwargs.get('beta',
                               [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                                1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        self.bias = kwargs.get('bias',
                               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                                0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.M = kwargs.get('M', 12)

        N = mol.GetNumAtoms()

        # Fix size for small ligand
        self.M = min(N - 1, self.M)

        confidx = kwargs.get('confidx', 0)
        conf = mol.GetConformer(confidx)
        coords = conf.GetPositions()

        self.distance_mat = np.zeros([N, N])
        for i in range(N):
            ci = np.tile(coords[i], N).reshape(N, 3)
            self.distance_mat[i] = np.linalg.norm(coords - ci, axis=1)

    def getNeighborhood(self):
        """
        Returns neighborhood information

        Returns:
            Nmap
            R
            Z
        """
        N = self.mol.GetNumAtoms()
        Nmap = np.zeros([N, self.M], dtype=np.int)
        R = np.zeros([N, self.M])
        Z = np.zeros([N, self.M], dtype=np.int)
        for i in range(N):
            Nmap[i] = np.argsort(self.distance_mat[i])[1:self.M+1]
            R[i] = np.sort(self.distance_mat[i])[1:self.M+1]
            for j in range(self.M):
                atom = self.mol.GetAtomWithIdx(int(Nmap[i,j]))
                Z[i,j] = atom.GetAtomicNum()

        return Nmap, R, Z

    def getConvolution(self, padding=None):
        """
        Returns convolution

        Returns:
            KR
        """
        Nmap, R, Z = self.getNeighborhood()

        Na = len(self.atomtypes)
        stacks = []
        for k in range(Na):
            atomtype = self.atomtypes[k]
            Ka = (Z == atomtype).astype(np.int)
            v = R*Ka
            stacks.append(v)
        KR = np.dstack(stacks)
        if padding is not None:
            n1 = KR.shape[0]
            n1 = padding - n1
            pad = np.zeros([n1, self.M, Na])
            KR = np.vstack((KR, pad))
        return KR

    def getRadialPoolingLayer(self, r_s, sigma_s):
        """
        Returns radial pooling layer

        Ordered args:
            r_s
            sigma_s
        Returns:
            P
        """
        def fc(r, cutoff):
            v = 0.0
            if r < cutoff:
                v = 0.5*(math.cos(math.pi*r/cutoff) + 1)
            return 0.0

        def fs(r, rs, sigma, cutoff):
            return math.exp(-(r-rs)*(r-rs)*fc(r, cutoff)/sigma/sigma)

        N = self.mol.GetNumAtoms()
        Na = len(self.atomtypes)
        Nr = len(self.radials)
        conv = self.getConvolution()
        P = np.zeros([N, Na, Nr])

        for _nr in range(Nr):
            cutoff = self.radials[_nr]
            for _n in range(N):
                for _na in range(Na):
                    r_list = conv[_n,:,_na]
                    v = np.sum(fs(r_list, r_s, sigma_s, cutoff))
                    v = self.beta[_nr]*v + self.bias[_nr]
                    P[_n, _na, _nr] = v
        return P

    @staticmethod
    def FromMolCombination(mol1, mol2, **kwargs):
        combined_mol = Chem.CombineMols(mol1, mol2)
        return AnalyticMol(combined_mol, **kwargs)

    @staticmethod
    def FromFile(fname, **kwargs):
        verbose = kwargs.get('verbose', False)
        sanitize = kwargs.get('sanitize', True)

        try:
            if fname.endswith('.sdf'):
                suppl = Chem.SDMolSupplier(fname)
                for mol in suppl:
                    break
            elif fname.endswith('.mol2'):
                mol = Chem.MolFromMol2File(fname)
            elif fname.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(fname)
            else:
                raise Exception('Unknown input file type')

            if mol is None:
                raise Exception(f'Molecule should not be None: {fname}')

            if sanitize:
                Chem.SanitizeMol(mol)

            N = mol.GetNumAtoms()
            if 10000 <= N:
                raise Exception(f'Molecule too big error: {fname} {N} atoms')

            if verbose:
                print(Chem.MolToSmiles(mol))
        except Exception as e:
            raise Exception(f'Read error: {fname}')

        return AnalyticMol(mol, **kwargs)

def main():
    print(f'{os.path.abspath(sys.argv[0])} {__VERSION__}')
    parser = argparse.ArgumentParser(description='Pande 2017')
    parser.add_argument('--ligand', '-l', type=str, help='Ligand file')
    parser.add_argument('--protein', '-p', type=str, help='Protein file')
    parser.add_argument('--out', '-o', type=str, default='out.npz', help='Output file')
    args = parser.parse_args()

    t = time.time()

    analytic_ligand = AnalyticMol.FromFile(args.ligand)
    analytic_protein = AnalyticMol.FromFile(args.protein)
    analytic_complex = AnalyticMol.FromMolCombination(analytic_ligand.mol, analytic_protein.mol)

    ligand_size = 70
    protein_size = 12000

    print('Atoms: '
          f'{analytic_ligand.mol.GetNumAtoms()}/'
          f'{analytic_protein.mol.GetNumAtoms()}/'
          f'{analytic_complex.mol.GetNumAtoms()}')

    print('ligand', end=' ')
    E_ligand = analytic_ligand.getConvolution(padding=ligand_size)
    print(E_ligand.shape)

    print('protein', end=' ')
    E_protein = analytic_protein.getConvolution(padding=protein_size)
    print(E_protein.shape)

    print('complex', end=' ')
    E_complex = analytic_complex.getConvolution(padding=ligand_size+protein_size)
    print(E_complex.shape)

    T = time.time()
    np.savez(args.out,
             ligand_file=args.ligand,
             protein_file=args.out,
             E_ligand=E_ligand,
             E_protein=E_protein,
             E_complex=E_complex)
    print(f'Elapsed: {T-t:.2f}s, written to {args.out}')

if __name__ == '__main__':
    main()
