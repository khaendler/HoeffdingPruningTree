import numpy as np
from typing import Optional, Any

from ixai.storage.reservoir_storage import ReservoirStorage


class GeometricReservoirStorage(ReservoirStorage):
    """ Geometric Reservoir Storage using a seed"""

    def __init__(self, size: int, constant_probability: float = None, store_targets: bool = False, seed: int = None):
        super().__init__(size=size, store_targets=store_targets)

        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1 / self.size

        self.rng = np.random.RandomState(seed)

    def update(self, x: dict, y: Optional[Any] = None):
        if len(self._storage_x) < self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            random_float = self.rng.random()
            if random_float <= self.constant_probability:
                rand_idx = self.rng.randint(self.size)
                self._storage_x[rand_idx] = x
                if self.store_targets:
                    self._storage_y[rand_idx] = y


class UniformReservoirStorage(ReservoirStorage):
    """ Uniform Reservoir Storage using a seed

    Summarizes a data stream by keeping track of a fixed length reservoir of observations.
    Each past observation of the stream has an equal probability of being in the reservoir at
    the current time.
    For more information we refer to https://en.wikipedia.org/wiki/Reservoir_sampling.

    Attributes:
        stored_samples int: Number of samples observed in the stream.
    """

    def __init__(self, size: int = 1000, store_targets: bool = False, seed: int = None):
        """
        Args:
            size (int): Length of the reservoir to store samples. Defaults to 1000.
            store_targets (bool): Flag if target labels should also be stored. Defaults to False.
        """
        super().__init__(
            size=size,
            store_targets=store_targets
        )
        self.rng = np.random.RandomState(seed)
        self.stored_samples: int = 0
        self._algo_wt = np.exp(np.log(self.rng.random()) / self.size)
        self._algo_l_counter: int = (
                self.size + (np.floor(np.log(self.rng.random()) / np.log(1 - self._algo_wt)) + 1)
        )

    def update(self, x: dict, y: Optional[Any] = None):
        """Updates the reservoir with the current sample if necessary.

        The update mechanism follows the optimal algorithm as stated here:
        https://en.wikipedia.org/wiki/Reservoir_sampling#Optimal:_Algorithm_L.

        Args:
            x (dict): Current observation's features.
            y (Any, optional): Current observation's label. Defaults to None.
        """
        self.stored_samples += 1
        if self.stored_samples <= self.size:
            self._storage_x.append(x)
            if self.store_targets:
                self._storage_y.append(y)
        else:
            if self._algo_l_counter == self.stored_samples:
                self._algo_l_counter += (np.floor(
                    np.log(self.rng.random()) / np.log(1 - self._algo_wt)) + 1)
                rand_idx = self.rng.randint(self.size)
                self._storage_x[rand_idx] = x
                if self.store_targets:
                    self._storage_y[rand_idx] = y
                self._algo_wt *= np.exp(np.log(self.rng.random()) / self.size)
