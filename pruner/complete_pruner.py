from pruner.base import BasePruner
from river.tree.nodes.branch import DTBranch


class CompletePruner(BasePruner):
    """ Pruner that uses preorder traversal to prune the HoeffdingPruningTree based on feature importance.
    A node that was split on an unimportant feature will be replaced by a new leaf containing the node's statistics.

    Parameters
    ----------
    tree
        The HoeffdingPruningTree that should be pruned.
    """

    def __init__(self, tree):
        self.tree = tree

    def prune_tree(self):
        """ Start the pruning process. """

        # Starts pruning from the root node if it is a branch.
        if isinstance(self.tree.root, DTBranch):
            self.prune_subtree(self.tree.root, None, 0)

    def prune_subtree(self, node, parent, index):
        """ Prune the subtree of the given node.

        Parameters
        ----------
        node
            The node which should be pruned.
        parent
            The parent of node.
        index
            An index indicating the position of the child in the parent.
        """

        if isinstance(node, DTBranch):
            # Prune the node if its splitting feature is not important.
            if node.feature not in self.tree.important_features:
                new_leaf = self.tree.create_new_leaf(initial_stats=node.stats, parent=parent)
                if parent is None:
                    self.tree.set_new_root(new_leaf)
                else:
                    parent.children[index] = new_leaf
            else:
                # Recursively prune the children.
                for i, child in enumerate(node.children):
                    self.prune_subtree(child, node, i)
