from abc import abstractmethod, ABC


class BasePruner(ABC):
    """ Base class for HoeffdingPruningTree pruner.

    Warning: This class should not be used directly.
    """

    @abstractmethod
    def __init__(self, tree):
        self.tree = tree

    @abstractmethod
    def prune_tree(self):
        pass
