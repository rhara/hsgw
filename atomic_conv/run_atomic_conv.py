"""
Script that trains Atomic Conv models on PDBbind dataset.
"""

import os
import deepchem as dc
import numpy as np
from libmodel import AtomicConvModel
from libloader import atomic_loader
import tensorflow as tf

# For stable runs
np.random.seed(123)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

frag1_num_atoms = 70
frag2_num_atoms = 24000
max_num_neighbors = 6
neighbor_cutoff = 6.0
batch_size = 1
trunc_dataset_size = None
# trunc_dataset_size = 20
epochs = 10

tasks, datasets, transformers = atomic_loader(frag1_num_atoms=frag1_num_atoms,
                                              frag2_num_atoms=frag2_num_atoms,
                                              max_num_neighbors=max_num_neighbors,
                                              neighbor_cutoff=neighbor_cutoff,
                                              trunc_dataset_size=trunc_dataset_size,
                                              reload=True)

train_dataset, valid_dataset, test_dataset = datasets
print('train_dataset X{} -> y{}'.format(train_dataset.X.shape, train_dataset.y.shape))
print('valid_dataset X{} -> y{}'.format(valid_dataset.X.shape, valid_dataset.y.shape))
print('test_dataset X{} -> y{}'.format(test_dataset.X.shape, test_dataset.y.shape))

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print(metric.name)

model = AtomicConvModel(batch_size=batch_size,
                        frag1_num_atoms=frag1_num_atoms,
                        frag2_num_atoms=frag2_num_atoms,
                        epochs=epochs)

print('Fitting model on train dataset')
model.fit(train_dataset)
model.save()

print('Evaluating model')
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print('Train scores')
print(train_scores)

print('Validation scores')
print(valid_scores)
