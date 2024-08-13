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
The following code shows how to evaluate multiple models.
```python
>>> from data.modified_agrawal import SuddenDriftAgrawal
>>> from utils.evaluate_multiple import evaluate_multiple

>>> from tree.hoeffding_pruning_tree import HoeffdingPruningTree
>>> from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier


>>> num_instances = 200000
>>> dataset = SuddenDriftAgrawal(drift_instance=10000)
>>> data_name = "SuddenDriftAgrawal"

>>> ht = HoeffdingTreeClassifier()
>>> hpt_cpl_02 = HoeffdingPruningTree(importance_threshold=0.02, pruner="complete")

>>> models = [ht, hpt_cpl_02]
>>> model_names = ["HT", "HPT_cpl_0.02"]

# Training and evaluating.
>>> results = evaluate_multiple(models, dataset, model_names, data_name, num_instances)
```

## Abstract of my Bachelor's Thesis
Explainable artificial intelligence has gained significant attention in recent
years, with decision trees playing a key role due to their inherent interpretability. Hoeffding trees are particularly popular for data streams where efficiency
and transparency are crucial. However, as data streams can be unbounded,
Hoeffding trees tend to grow indefinitely, increasing complexity. An approach
to explain complex models, is to use model-agnostic methods like incremental permutation feature importance (iPFI). IPFI provides insight into feature
impact without being tied to a specific model. This thesis introduces the Hoeffding pruning tree (HPT), a novel approach that integrates Hoeffding trees
with iPFI to manage unbounded growth and improve adaptability in nonstationary data streams. HPT uses feature importance to transparently prune
the tree, employing either a complete or selective pruning strategy to simplify the pruning process. Experiments show that HPT achieves performance
comparable to state-of-the-art adaptive Hoeffding trees, while maintaining a
lower node count through transparent pruning, effectively balancing interpretability and adaptability in data stream processing.
