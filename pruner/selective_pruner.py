from pruner.base import BasePruner
from river.tree.nodes.leaf import HTLeaf
from river.tree.nodes.branch import DTBranch


class SelectivePruner(BasePruner):
    """ Pruner that uses postorder traversal to prune the HoeffdingPruningTree based on feature importance.
    A node that was split on an unimportant feature will be replaced by the children's subtree with the highest
    amount of average promise in its leaves. Promise defines how likely it is that a leaf is going to split.

    Parameters
    ----------
    tree
        The HoeffdingPruningTree that should be pruned.
    """

    def __init__(self, tree):
        self.tree = tree

    def prune_tree(self):
        """ Start the pruning process. """

        # Starts pruning from the root node if it is not a leaf.
        if isinstance(self.tree.root, DTBranch):
            self.tree.set_new_root(self.prune_subtree(self.tree.root))

    def prune_subtree(self, node):
        """ Prune the subtree of the given node.

        Parameters
        ----------
        node
            The node which should be pruned.

        Returns
        -------
            The child with the highest amount of average promise in its subtree.
        """
        # First prune every child that is a branch.
        pruned_children = [self.prune_subtree(child) if not isinstance(child, HTLeaf)
                           else child for child in node.children]

        # Then decide whether to replace this node based on its splitting feature.
        if node.feature not in self.tree.important_features:
            return self._choose_child(pruned_children)
        else:
            node.children = pruned_children
            return node

    def _choose_child(self, children):
        """ Choose the child with the highest amount of average promise in its subtree.

        Parameters
        ----------
        children
            A list of child nodes.

        Returns
        -------
            The child with the highest amount of average promise in its subtree.

        """

        # Calculate average promises for each child.
        promises = [self._calculate_avg_promise(child) if not isinstance(child, HTLeaf)
                    else child.calculate_promise() for child in children]

        # Find the child with the highest average promise
        max_avg_promise_index = promises.index(max(promises))
        return children[max_avg_promise_index]

    def _calculate_avg_promise(self, node):
        """ Calculate the average promise of the node's subtree."""

        leaves = [leaf for leaf in node.iter_leaves() if isinstance(leaf, HTLeaf)]
        total_promise = sum(leaf.calculate_promise() for leaf in leaves)
        return total_promise / len(leaves)

