"""
Script that trains Sklearn RF models on PDBbind Pockets dataset.
"""
import os
import deepchem as dc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from libloader import load_pockets

# For stable runs
np.random.seed(123)

split = 'random'
subset = 'core'
trunc_dataset_size = 20

tasks, datasets, transformers = load_pockets(split=split, subset=subset, trunc_dataset_size=trunc_dataset_size)
train_dataset, valid_dataset, test_dataset = datasets

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'pocket_%s_%s_RF' % (split, subset))

sklearn_model = RandomForestClassifier(n_estimators=500)
model = dc.models.SklearnModel(sklearn_model, model_dir=model_dir)

# Fit trained model
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
