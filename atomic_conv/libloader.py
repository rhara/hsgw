"""
PDBBind dataset loader.
"""

import os, logging
import deepchem as dc
import numpy as np
from libfeaturizer import ComplexNeighborListFragmentAtomicCoordinates

logger = logging.getLogger(__name__)


def atomic_loader(frag1_num_atoms=70, frag2_num_atoms=24000, max_num_neighbors=4, neighbor_cutoff=4, trunc_dataset_size=None, reload=True):
    """
    Load and featurize raw PDBBind dataset.
    """

    featurizer = 'atomic'
    split = 'random'
    subset = 'core'

    complex_num_atoms = frag1_num_atoms + frag2_num_atoms

    pdbbind_tasks = ["-logKd/Ki"]
    data_dir = dc.utils.get_data_dir()
    if reload:
        save_dir = os.path.join(data_dir, "pdbbind/" + featurizer + "/" + str(split))
        print('save_dir', save_dir)
        loaded, all_dataset, transformers = dc.utils.save.load_dataset_from_disk(save_dir)
        if loaded:
            return pdbbind_tasks, all_dataset, transformers
    else:
      save_dir = os.path.join(data_dir, "pdbbind/" + featurizer + "/" + str(split))

    print('data_dir', data_dir)
    print('save_dir', save_dir)
    data_folder = os.path.join(data_dir, "v2015")

    index_file = os.path.join(data_folder, "INDEX_core_name.2013")
    labels_file = os.path.join(data_folder, "INDEX_core_data.2013")

    # Extract locations of data
    pdbs = []
    with open(index_file, "r") as g:
        for line in g.readlines():
            line = line.split(" ")
            pdb = line[0]
            if len(pdb) == 4:
                pdbs.append(pdb)

    print('len(pdbs)', len(pdbs))

    protein_files = [os.path.join(data_folder, pdb, "%s_protein.pdb" % pdb) for pdb in pdbs]
    ligand_files = [os.path.join(data_folder, pdb, "%s_ligand.sdf" % pdb) for pdb in pdbs]

    # Extract labels
    labels = []
    with open(labels_file, "r") as f:
        for line in f.readlines():
            if line[0] == "#":
                continue
            # Lines have format
            # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
            line = line.split()
            # The base-10 logarithm, -log kd/pk
            log_label = float(line[3])
            labels.append(log_label)
    labels = np.array(labels)

    # Pulled from PDB files. For larger datasets with more PDBs, would use
    # max num atoms instead of exact.

    featurizer = ComplexNeighborListFragmentAtomicCoordinates(frag1_num_atoms, frag2_num_atoms, complex_num_atoms, max_num_neighbors, neighbor_cutoff)

    print("Featurizing Complexes")

    # make data smaller for testing
    if trunc_dataset_size is not None:
        protein_files = protein_files[:trunc_dataset_size]
        ligand_files = ligand_files[:trunc_dataset_size]
        labels = labels[:trunc_dataset_size]
    print(labels)

    features, failures = featurizer.featurize_complexes(ligand_files, protein_files)
    print('failures', failures)
    # Delete labels for failing elements
    labels = np.delete(labels, failures)
    labels = labels.reshape((len(labels), 1))
    dataset = dc.data.DiskDataset.from_numpy(features, labels)
    print('Featurization complete.')
    # No transformations of data
    transformers = []

    # TODO(rbharath): This should be modified to contain a cluster split so
    # structures of the same protein aren't in both train/test
    splitters = {
        'index': dc.splits.IndexSplitter(),
        'random': dc.splits.RandomSplitter(),
    }
    splitter = splitters[split]
    train, valid, test = splitter.train_valid_test_split(dataset)
    all_dataset = (train, valid, test)
    if reload:
        dc.utils.save.save_dataset_to_disk(save_dir, train, valid, test, transformers)
    return pdbbind_tasks, all_dataset, transformers
