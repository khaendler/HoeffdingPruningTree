import numpy as np
from river import datasets
from river.datasets.synth import ConceptDriftStream


class GradualConceptDriftStream(ConceptDriftStream):
    """Custom ConceptDriftStream with gradual concept drift between two stream generators.
    This generator ensures that if both generators produce the same features values,
    they remain consistent, and only the target values transition from one generator
    to the other over the specified width.

    A stream generator that adds concept drift or change by joining two
    streams. This is done by building a weighted combination of two pure
    distributions that characterizes the target concepts before and after
    the change.

    The sigmoid function is an elegant and practical solution to define the
    probability that each new instance of the stream belongs to the new
    concept after the drift. The sigmoid function introduces a gradual, smooth
    transition whose duration is controlled with two parameters:

    - $p$, the position of the change.

    - $w$, the width of the transition.

    The sigmoid function at sample $t$ is

    $$f(t) = 1/(1+e^{-4(t-p)/w})$$

    Parameters
    ----------
    stream
        Original agrawal stream
    drift_stream
        Drift agrawal stream
    seed
        Random seed for reproducibility.
    alpha
        Angle of change used to estimate the width of concept drift change.
        If set, it will override the width parameter. Valid values are in the
        range (0.0, 90.0].
    position
        Central position of the concept drift change.
    width
        Width of concept drift change.

    Notes
    -----
    An optional way to estimate the width of the transition $w$ is based on
    the angle $\\alpha$, $w = 1/ tan(\\alpha)$. Since width corresponds to
    the number of samples for the transition, the width is rounded to the
    nearest smaller integer. Notice that larger values of $\\alpha$ result in
    smaller widths. For $\\alpha > 45.0$, the width is smaller than 1 so values
    are rounded to 1 to avoid division by zero errors.
    """
    def __init__(
        self,
        stream: datasets.base.SyntheticDataset | None = None,
        drift_stream: datasets.base.SyntheticDataset | None = None,
        position: int = 5000,
        width: int = 1000,
        seed: int | None = None,
        alpha: float | None = None,
    ):
        super().__init__(stream, drift_stream, position, width, seed, alpha)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        stream_generator = iter(self.stream)
        drift_stream_generator = iter(self.drift_stream)
        sample_idx = 0

        while True:
            sample_idx += 1
            v = -4.0 * float(sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(v))
            try:
                if rng.random() > probability_drift:
                    x, y = next(stream_generator)
                    next(drift_stream_generator)
                else:
                    x, y = next(drift_stream_generator)
                    next(stream_generator)
            except StopIteration:
                break
            yield x, y
