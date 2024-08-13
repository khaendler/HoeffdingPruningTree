from river.tree.splitter import Splitter
from river.tree.utils import BranchFactory
from river.tree.extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier


class EFDT(ExtremelyFastDecisionTreeClassifier):
    """Fixed version of the Extremely Fast Decision Tree classifier from river version 0.20.0.
    In earlier versions, EFDT had a bug where the split re-evaluation failed when the current branch's
    feature was not available as a split option. The fix also enable the tree to pre-prune a leaf via
    the tie-breaking mechanism.

    Also referred to as Hoeffding AnyTime Tree (HATT) classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    min_samples_reevaluate
        Number of instances a node should observe before reevaluating the best split.
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
        List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
        should be treated as continuous.
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

    Notes
    -----
    The Extremely Fast Decision Tree (EFDT) [^1] constructs a tree incrementally. The EFDT seeks to
    select and deploy a split as soon as it is confident the split is useful, and then revisits
    that decision, replacing the split if it subsequently becomes evident that a better split is
    available. The EFDT learns rapidly from a stationary distribution and eventually it learns the
    asymptotic batch tree if the distribution from which the data are drawn is stationary.

    References
    ----------
    [^1]:  C. Manapragada, G. Webb, and M. Salehi. Extremely Fast Decision Tree.
    In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data
    Mining (KDD '18). ACM, New York, NY, USA, 1953-1962.
    DOI: https://doi.org/10.1145/3219819.3220005
    """

    def __init__(
        self,
        grace_period: int = 200,
        max_depth: int = None,
        min_samples_reevaluate: int = 20,
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
    ):

        super().__init__(
            grace_period=grace_period,
            max_depth=max_depth,
            min_samples_reevaluate=min_samples_reevaluate,
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

    def _reevaluate_best_split(self, node, parent, branch_index, **kwargs):
        """Reevaluate the best split for a node.

        If the samples seen so far are not from the same class then:
        1. Find split candidates and select the best one.
        2. Compute the Hoeffding bound.
        3. If the null split candidate is higher than the top split candidate:
            3.1 Kill subtree and replace it with a leaf.
            3.2 Update the tree.
            3.3 Update tree's metrics
        4. If the difference between the top split candidate and the current split is larger than
        the Hoeffding bound:
           4.1 Create a new split node.
           4.2 Update the tree.
           4.3 Update tree's metrics
        5. If the top split candidate is the current split but with different split test:
           5.1 Update the split test of the current split.

        Parameters
        ----------
        node
            The node to reevaluate.
        parent
            The node's parent.
        branch_index
            Parent node's branch index.
        kwargs
            Other parameters passed to the branch node.

        Returns
        -------
            Flag to stop moving in depth.
        """
        stop_flag = False
        if not node.observed_class_distribution_is_pure():
            split_criterion = self._new_split_criterion()
            best_split_suggestions = node.best_split_suggestions(split_criterion, self)
            if len(best_split_suggestions) > 0:
                # Sort the attribute accordingly to their split merit for each attribute
                # (except the null one)
                best_split_suggestions.sort()

                # x_best is the attribute with the highest merit
                x_best = best_split_suggestions[-1]
                id_best = x_best.feature

                # Best split candidate is the null split
                if x_best.feature is None:
                    return True

                # x_current is the current attribute used in this SplitNode
                id_current = node.feature
                x_current = node.find_attribute(id_current, best_split_suggestions)

                # Get x_null
                x_null = BranchFactory(merit=0)

                # Compute Hoeffding bound
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(node.stats),
                    self.delta,
                    node.total_weight,
                )

                if x_null.merit - x_best.merit > hoeffding_bound:
                    # Kill subtree & replace the branch by a leaf
                    best_split = self._kill_subtree(node)

                    # update EFDT
                    if parent is None:
                        # Root case : replace the root node by a new split node
                        self._root = best_split
                    else:
                        parent.children[branch_index] = best_split

                    n_active = n_inactive = 0
                    for leaf in node.iter_leaves():
                        if leaf.is_active():
                            n_active += 1
                        else:
                            n_inactive += 1

                    self._n_active_leaves += 1
                    self._n_active_leaves -= n_active
                    self._n_inactive_leaves -= n_inactive
                    stop_flag = True

                    # Manage memory
                    self._enforce_size_limit()

                elif x_current is not None:
                    if (
                        x_best.merit - x_current.merit > hoeffding_bound
                        or hoeffding_bound < self.tau
                    ) and (id_current != id_best):

                        # Create a new branch
                        branch = self._branch_selector(
                            x_best.numerical_feature, x_best.multiway_split
                        )
                        leaves = tuple(
                            self._new_leaf(initial_stats, parent=node)
                            for initial_stats in x_best.children_stats
                        )

                        new_split = x_best.assemble(
                            branch, node.stats, node.depth, *leaves, **kwargs
                        )

                        # Update weights in new_split
                        new_split.last_split_reevaluation_at = node.total_weight

                        n_active = n_inactive = 0
                        for leaf in node.iter_leaves():
                            if leaf.is_active():
                                n_active += 1
                            else:
                                n_inactive += 1

                        self._n_active_leaves -= n_active
                        self._n_inactive_leaves -= n_inactive
                        self._n_active_leaves += len(leaves)

                        if parent is None:
                            # Root case : replace the root node by a new split node
                            self._root = new_split
                        else:
                            parent.children[branch_index] = new_split

                        stop_flag = True

                        # Manage memory
                        self._enforce_size_limit()

                    elif (
                        x_best.merit - x_current.merit > hoeffding_bound
                        or hoeffding_bound < self.tau
                    ) and (id_current == id_best):

                        branch = self._branch_selector(
                            x_best.numerical_feature, x_best.multiway_split
                        )
                        # Change the branch but keep the existing children nodes
                        new_split = x_best.assemble(
                            branch, node.stats, node.depth, *tuple(node.children), **kwargs
                        )
                        # Update weights in new_split
                        new_split.last_split_reevaluation_at = node.total_weight

                        if parent is None:
                            # Root case : replace the root node by a new split node
                            self._root = new_split
                        else:
                            parent.children[branch_index] = new_split

        return stop_flag

