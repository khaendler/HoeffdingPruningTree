from __future__ import annotations
import numpy as np

from river.datasets.synth import SEA


class SuddenDriftSEA(SEA):
    """SEA synthetic dataset having a sudden concept drift at a given instance.

    Implementation of the data stream with abrupt drift described in [^1]. Each observation is
    composed of 3 features. Only the first two features are relevant. The target is binary, and is
    positive if the sum of the features exceeds a certain threshold. There are 4 thresholds to
    choose from. Concept drift can be introduced by switching the threshold anytime during the
    stream.

    * **Variant 0**: `True` if $att1 + att2 > 8$

    * **Variant 1**: `True` if $att1 + att2 > 9$

    * **Variant 2**: `True` if $att1 + att2 > 7$

    * **Variant 3**: `True` if $att1 + att2 > 9.5$

    Parameters
    ----------
    variant
        Determines the classification function to use. Possible choices are 0, 1, 2, 3.
    drift_variant
        Determines the drift classification function to use. Possible choices are 0, 1, 2, 3.
    drift_instance
        Determines the instance at which a sudden drift should occur.
    noise
        Determines the amount of observations for which the target sign will be flipped.
    seed
        Random seed number used for reproducibility.

    References
    ----------
    [^1]: [A Streaming Ensemble Algorithm (SEA) for Large-Scale Classification](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.482.3991&rep=rep1&type=pdf)

    """

    def __init__(self, variant=1, drift_variant=2, drift_instance=1000000, noise=0.0, seed: int | None = None):
        super().__init__(variant, noise, seed)
        self.drift_variant = drift_variant
        self.drift_instance = drift_instance
        self.instance_counter = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        while True:
            self.instance_counter += 1
            if self.instance_counter == self.drift_instance:
                self.generate_drift(self.drift_variant)

            x = {f"attr{i+1}": rng.uniform(0, 10) for i in range(3)}
            y = x["attr1"] + x["attr2"] > self._threshold

            if self.noise and rng.random() < self.noise:
                y = not y

            yield x, y

    def generate_drift(self, drift_variant):
        self.variant = drift_variant
        self._threshold = {0: 8, 1: 9, 2: 7, 3: 9.5}[self.variant]
