# Enhancing Transparency in Hoeffding Trees: Integrate Model-Specific and Model-Agnostic Techniques

## Overview
This project contains the Hoeffding Pruning Tree (HPT), an extension of the Hoeffding Tree that incorporates incremental
Permutation Feature Importance (iPFI) to prune the tree effectively. Pruning can be executed using two distinct
strategies: complete and selective, both of which utilize feature importance values to determine which branches
to prune. HPT aims to reduce the tree size and enhance transparency. This work is presented as the Bachelor's thesis
of K. Händler.

## Implementation
- Hoeffding Pruning Tree
  - Complete and selective pruning strategy
- Incremental PFI with seeds for reproducibility
  - Only for the sampling strategies and marginal imputer
- Modified versions of Agrawal and SEA generators to specify concept drifts.
- A fixed version of the Extremely Fast Hoeffding Tree from river version 0.20.0.

The implementation is based on [river](https://github.com/online-ml/river)
and [ixai](https://github.com/mmschlk/iXAI). The Hoeffding Pruning Tree is implemented as a Hoeffding tree classifier 
and should support all river functionalities.

## Installation
The Hoeffding Pruning Tree requires **Python 3.8 or above**. Installation of the requirements can be done via `pip`:
```sh
pip install -r requirements.txt 
```
**Note**: The project works with **river version 0.16.0**. Later versions may not be supported by ixai.

## Example Code
```python
>>> from river.metrics import RollingROCAUC

>>> from data.modified_agrawal import SuddenDriftAgrawal
>>> from tree.hoeffding_pruning_tree import HoeffdingPruningTree

>>> dataset = SuddenDriftAgrawal(drift_instance=25000)
>>> model = HoeffdingPruningTree(importance_threshold=0.02, pruner="complete")

>>> metric = RollingROCAUC()
>>> for i, (x, y) in enumerate(dataset.take(50000), start=1):
...    y_pred = model.predict_proba_one(x)     # predicting
...    metric.update(y, y_pred)                # updating metric
...    model.learn_one(x, y)                   # learning
...
...    if i % 10000 == 0:
...        print(f"{i}: Accuracy: {metric.get():.3f}, PFI: {model.importance_values}")
        
10000: Accuracy: 0.798, PFI: {'salary': 0.218, 'age': 0.031, 'elevel': 0.001, 'commission': 0.003, 'loan': 0.002, 'hvalue': -0.002, 'zipcode': 0.002, 'car': 0.006, 'hyears': 0.002}
20000: Accuracy: 0.905, PFI: {'salary': 0.290, 'age': 0.179, 'elevel': -0.001, 'commission': 0.026, 'loan': -0.001, 'hvalue': -0.001, 'zipcode': -0.001, 'car': 0.001, 'hyears': 0.002}
30000: Accuracy: 0.657, PFI: {'salary': 0.002, 'age': 0.062, 'elevel': 0.111, 'commission': -0.001, 'loan': 0.001, 'hvalue': 0.001, 'zipcode': 0.000, 'car': 0.004, 'hyears': 0.004}
40000: Accuracy: 1.000, PFI: {'salary': -0.000, 'age': 0.457, 'elevel': 0.455, 'commission': -0.000, 'loan': -0.000, 'hvalue': 0.000, 'zipcode': -0.001, 'car': -0.001, 'hyears': 0.000}
50000: Accuracy: 1.000, PFI: {'salary': -0.000, 'age': 0.442, 'elevel': 0.461, 'commission': -0.000, 'loan': -0.000, 'hvalue': 0.000, 'zipcode': -0.000, 'car': -0.000, 'hyears': 0.000}
```

## Abstract of my Bachelor's Thesis
...