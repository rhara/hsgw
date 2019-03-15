from rdkit import Chem
import numpy as np

class Featurizer:
    def __init__(self, mol, **kwargs):
        self.mol = mol
        self.atomtype_list = kwargs.get('atomtype_list', [6,7,8,9,11,12,15,16,17,20,25,30,35,53])
        self.M = kwargs.get('M', 12)
        confidx = kwargs.get('confidx', 0)

        N = mol.GetNumAtoms()
        # Fix size for small ligand
        self.M = min(N - 1, self.M)

        conf = mol.GetConformer(confidx)
        coords = conf.GetPositions()

        self.atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        self.distance_mat = np.zeros([N, N])
        for i in range(N):
            ci = np.tile(coords[i], N).reshape(N, 3)
            self.distance_mat[i] = np.linalg.norm(coords - ci, axis=1)

        self.neighbors = None # will be set in self._setNeighborhood
        self.R = None         # will be set in self._setRandZ
        self.Z = None         # will be set in self._setRandZ
        self.E = None         # will be set in self.setConvolution (mandatory)
        self.P = None         # will be set in self.setRadialLayer succeedingly

    def _setNeighborhood(self):
        N = self.mol.GetNumAtoms()
        self.neighbors = np.zeros([N, self.M], dtype=np.int)
        for i in range(N):
            self.neighbors[i] = np.argsort(self.distance_mat[i])[1:self.M+1]

    def _setRandZ(self):
        N = self.mol.GetNumAtoms()
        self.R = np.zeros((N, self.M))
        self.Z = np.zeros((N, self.M), dtype=np.int)
        for i in range(N):
            self.R[i] = self.distance_mat[i, self.neighbors[i]]
            self.Z[i] = self.atom_types[self.neighbors[i]]

    def setConvolution(self, padding=None):
        self._setNeighborhood()
        self._setRandZ()

        Na = len(self.atomtype_list)
        stacks = []
        for k in range(Na):
            atomtype = self.atomtype_list[k]
            Ka = (self.Z == atomtype).astype(np.int)
            v = self.R*Ka
            stacks.append(v)
        self.E = np.dstack(stacks)
        if padding is not None:
            n1 = self.E.shape[0]
            n1 = padding - n1
            pad = np.zeros([n1, self.M, Na])
            self.E = np.vstack((self.E, pad))

    def setRadialLayer(self, beta, bias, r_s=1., sigma_s=1., radials=np.arange(1.5,12.+1e-7,.5)):
        def fc(r, cutoff):
            return (r < cutoff)*(np.cos(np.pi*r/cutoff) + 1)

        def fs(r, cutoff):
            return np.exp(-(r-r_s)*(r-r_s)*fc(r, cutoff)/sigma_s/sigma_s)

        N = self.E.shape[0]
        Na = len(self.atomtype_list)
        Nr = len(radials)

        self.P = np.zeros([N, Na, Nr])
        for i in range(Nr):
            cutoff = radials[i]
            r0_list = np.zeros(Na)
            v0 = beta[i]*np.sum(fs(r0_list, cutoff)) + bias[i]
            for j in range(N):
                for k in range(Na):
                    r_list = self.E[j,:,k]
                    if np.all(r_list == 0.):
                        self.P[j, k, i] = v0
                    else:
                        self.P[j, k, i] = beta[i]*np.sum(fs(r_list, cutoff)) + bias[i]
