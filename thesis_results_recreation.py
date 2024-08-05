from data.utils import load_datasets
from utils.evaluate_multiple import evaluate_multiple

from tree.EFDT import EFDT
from tree.hoeffding_pruning_tree import HoeffdingPruningTree

from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from river.tree.hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier


num_instances = 1000000
seeds = [40, 41, 42, 43, 44]
results = {"seed40": {}, "seed41": {}, "seed42": {}, "seed43": {}, "seed44": {}}
for seed in seeds:
    datasets = load_datasets(num_instances=num_instances, pertubation=0.2, seed=seed)
    datasets = [(datasets[0], "agrawal_base"),
                (datasets[1], "agrawal_sudden"),
                (datasets[2], "agrawal_gradual"),
                (datasets[3], "agrawal_feature"),
                (datasets[4], "agrawal_recurring"),
                (datasets[5], "agrawal_mixed"),
                (datasets[6], "sea"),
                (datasets[7], "hyperplane")
                ]

    for data, data_name in datasets:

        ht = HoeffdingTreeClassifier()
        efdt = EFDT()
        hatc = HoeffdingAdaptiveTreeClassifier(bootstrap_sampling=False, seed=42)

        hpt_sel_05 = HoeffdingPruningTree(importance_threshold=0.05,pruner="selective", seed=42)
        hpt_sel_02 = HoeffdingPruningTree(importance_threshold=0.02, pruner="selective", seed=42)
        hpt_sel_00 = HoeffdingPruningTree(importance_threshold=0, pruner="selective", seed=42)

        hpt_cpl_05 = HoeffdingPruningTree(importance_threshold=0.05, pruner="complete", seed=42)
        hpt_cpl_02 = HoeffdingPruningTree(importance_threshold=0.02, pruner="complete", seed=42)
        hpt_cpl_00 = HoeffdingPruningTree(importance_threshold=0, pruner="complete", seed=42)

        models = [ht, efdt, hatc,
                  hpt_sel_05, hpt_sel_02, hpt_sel_00,
                  hpt_cpl_05,
                  hpt_cpl_02, hpt_cpl_00
                  ]
        model_names = ["base_ht", "efdt", "hatc",
                       "hpt_sel_05", "hpt_sel_02", "hpt_sel_00",
                       "hpt_cpl_05",
                       "hpt_cpl_02", "hpt_cpl_00"
                       ]

        # Training and evaluating.
        results[f"seed{seed}"] = evaluate_multiple(models, data, model_names, data_name, num_instances)



