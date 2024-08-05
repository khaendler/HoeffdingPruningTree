from utils.evaluate_multiple import evaluate_multiple
from data.modified_agrawal import SuddenDriftAgrawal
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier
from tree.hoeffding_pruning_tree import HoeffdingPruningTree


num_instances = 200000
dataset = SuddenDriftAgrawal(drift_instance=10000)
data_name = "SuddenDriftAgrawal"

ht = HoeffdingTreeClassifier()
hpt_cpl_02 = HoeffdingPruningTree(importance_threshold=0.02, pruner="complete", seed=42)

models = [ht, hpt_cpl_02]
model_names = ["ht", "hpt_cpl_02"]

# Training and evaluating.
results = evaluate_multiple(models, dataset, model_names, data_name, num_instances)
