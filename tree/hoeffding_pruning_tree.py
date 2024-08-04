from river.metrics import Accuracy
from river.tree import HoeffdingTreeClassifier
from pfi_with_seeds.ipfi import IPFI
from ixai.utils.wrappers import RiverWrapper
from ixai.visualization import FeatureImportancePlotter

from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch
from river.tree.splitter import Splitter

from pruner.complete_pruner import CompletePruner
from pruner.selective_pruner import SelectivePruner


class HoeffdingPruningTree(HoeffdingTreeClassifier):
    """Hoeffding Pruning Tree using the VFDT classifier with incremental PFI to prune the tree.

    Note: This works with river versions 0.16.0. Later versions may not be supported.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.
    importance_threshold
        The threshold value that determines whether a feature is important or not.
    pruner
        The Pruner to use.</br>
        - 'selective' - SelectivePruner</br>
        - 'complete' - CompletePruner</br>
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        importance_threshold: float = 0.05,
        pruner: str = "complete",
        seed: int = None
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.incremental_pfi = None
        self.pfi_plotter = None
        self.importance_threshold = importance_threshold

        self.feature_names = None
        self.important_features = None
        self.last_important_features = None

        self.pruner = self._set_pruner(pruner)
        self.seed = seed

    @property
    def root(self):
        return self._root

    @property
    def importance_values(self):
        return self.incremental_pfi.importance_values

    def set_new_root(self, node: HTLeaf | DTBranch):
        self._root = node

    def create_new_leaf(self, initial_stats: dict | None = None, parent: HTLeaf | DTBranch | None = None):
        return self._new_leaf(initial_stats, parent)

    def _update_leaf_counts(self):
        leaves = [leaf for leaf in self._root.iter_leaves()]
        self._n_active_leaves = sum(1 for leaf in leaves if leaf.is_active())
        self._n_inactive_leaves = sum(1 for leaf in leaves if not leaf.is_active())

    def _set_pruner(self, pruner):
        if pruner == "selective":
            return SelectivePruner(self)
        elif pruner == "complete":
            return CompletePruner(self)
        else:
            raise ValueError(f"Invalid pruner type: {pruner}. Valid options are 'selective' or 'complete'.")

    def learn_one(self, x, y, *, sample_weight=1.0):
        # Initialize the incremental PFI instance.
        if self.incremental_pfi is None:
            self.feature_names = list(x.keys())
            self.important_features = self.feature_names
            self.last_important_features = set(self.feature_names)
            self._create_ipfi()

        self._update_ipfi(x, y)

        # Prune the tree if the set of important features has changed.
        if self.last_important_features != set(self.important_features):
            self.pruner.prune_tree()
            self._update_leaf_counts()

        self.last_important_features = set(self.important_features)

        # learning
        super().learn_one(x, y, sample_weight=1.0)

        return self

    def _create_ipfi(self):
        self.incremental_pfi = IPFI(
            model_function=RiverWrapper(self.predict_one),
            loss_function=Accuracy(),
            feature_names=self.feature_names,
            smoothing_alpha=0.001,
            n_inner_samples=5,
            seed=self.seed
        )

        self.pfi_plotter = FeatureImportancePlotter(feature_names=self.feature_names)

    def _update_ipfi(self, x, y):
        """Updates iPFI, PFI plotter and the list of current important features based on the importance threshold."""

        # explaining
        inc_fi_pfi = self.incremental_pfi.explain_one(x, y)

        # update visualizer
        self.pfi_plotter.update(inc_fi_pfi)

        # Check if any feature's importance value meets or exceeds the threshold.
        # If so, select those features as important. Otherwise, consider all features as important.
        if any(value >= self.importance_threshold for value in self.incremental_pfi.importance_values.values()):
            self.important_features = [k for k, v in self.incremental_pfi.importance_values.items()
                                       if v >= self.importance_threshold]
        else:
            self.important_features = self.feature_names

    def plot_pfi(
            self,
            names_to_highlight: list,
            title: str = None,
            model_performances: list = None,
            metric_name: str = None,
            save_name: str = None
    ):
        """Plots the feature importance values up to the current point in time.

        Parameters
        ----------
        names_to_highlight
            The names of the features to highlight in the plot.
        title
            The title of the plot.
        model_performances
            A list of performances of the model. If given, a performance plot will be added.
        metric_name
            The name of the metric used for the performance values.
        save_name
            If given, saves the plot with the name given.
        """

        metric_name = "Perf." if metric_name is None else metric_name
        performance_kw = {
            "y_min": 0, "y_max": 1, "y_label": metric_name
        }

        fi_kw = {
            "names_to_highlight": names_to_highlight,
            "legend_style": {
                "fontsize": "small", 'title': 'features', "ncol": 1,
                "loc": 'upper left', "bbox_to_anchor": (0, 1)},
            "title": title
        }
        model_performances = None if model_performances is None else {metric_name: model_performances}
        self.pfi_plotter.plot(
            save_name=save_name,
            model_performances=model_performances,
            performance_kw=performance_kw,
            **fi_kw
        )



