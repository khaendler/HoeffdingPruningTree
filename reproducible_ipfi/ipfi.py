import copy
from typing import Union, Sequence, Dict, Callable, Any, Optional

from river.metrics.base import Metric

from ixai.explainer import IncrementalPFI
from ixai.explainer.base import BaseIncrementalExplainer
from ixai.imputer import BaseImputer, MarginalImputer
from ixai.storage.base import BaseStorage
from ixai.utils.tracker.base import Tracker
from ixai.utils.tracker import MultiValueTracker, WelfordTracker, ExponentialSmoothingTracker
from ixai.utils.validators.loss import validate_loss_function

from reproducible_ipfi.marginal_imputer import MarginalImputer
from reproducible_ipfi.reservoir import GeometricReservoirStorage, UniformReservoirStorage


class IPFI(IncrementalPFI):
    """Incremental PFI Explainer using a seed.

    Note: This  works with river versions 0.16.0. Later versions may not be supported.

    Computes PFI importance values incrementally by applying exponential smoothing.
    For each input instance tuple x_i, y_i one update of the explanation procedure is performed.

    Parameters
    ----------
    n_inner_samples (int):
        The number of inner_samples used for removing features.
    model_function (Callable[[Any], Any]):
        The Model function to be explained (e.g. model.predict_one (river), model.predict_proba (sklearn)).
    loss_function (Union[Metric, Callable[[Any, Dict], float]]):
        The loss function for which the importance values are calculated.
        This can either be a callable function or a predefined river.metric.base.Metric.<br>
        - river.metric.base.Metric: Any Metric implemented in river (e.g.
            river.metrics.CrossEntropy() for classification or river.metrics.MSE() for
            regression).<br>
        - callable function: The loss_function needs to follow the signature of
            loss_function(y_true, y_pred) and handle the output dimensions of the model
            function. Smaller values are interpreted as being better if not overriden with
            `loss_bigger_is_better=True`. `y_pred` is passed as a dict.
    feature_names (Sequence[Union[str, int, float]]):
        List of feature names to be explained for the model.
    smoothing_alpha (float, optional):
        The smoothing parameter for the exponential smoothing of the importance values.
        Should be in the interval between ]0,1]. Defaults to 0.001.
    storage (BaseStorage, optional):
        Optional incremental data storage Mechanism. Defaults to `GeometricReservoirStorage(size=100)`
         for dynamic modelling settings (`dynamic_setting=True`) and `UniformReservoirStorage(size=100)`
         in static modelling settings (`dynamic_setting=False`).
    imputer (BaseImputer, optional):
        Incremental imputing strategy to be used. Defaults to `MarginalImputer(sampling_strategy='joint')`.
    n_inner_samples (int):
        Number of model evaluation per feature and explanation step (observation). Defaults to 1.
    dynamic_setting (bool):
        Flag to indicate if the modelling setting is dynamic `True` (changing model, and adaptive explanation)
        or a static modelling setting `False` (all observations contribute equally to the final importance).
        Defaults to `True`.
    """
    def __init__(
            self,
            model_function: Callable[[Any], Any],
            loss_function: Union[Metric, Callable[[Any, Dict], float]],
            feature_names: Sequence[Union[str, int, float]],
            storage: Optional[BaseStorage] = None,
            imputer: Optional[BaseImputer] = None,
            n_inner_samples: int = 1,
            smoothing_alpha: float = 0.001,
            dynamic_setting: bool = True,
            seed: int = None
    ):
        BaseIncrementalExplainer.__init__(self, model_function, feature_names)
        self._loss_function = validate_loss_function(loss_function)

        self._smoothing_alpha = 0.001 if smoothing_alpha is None else smoothing_alpha
        if dynamic_setting:
            assert 0. < smoothing_alpha <= 1., f"The smoothing parameter needs to be in the range" \
                                               f" of ']0,1]' and not " \
                                               f"'{self._smoothing_alpha}'."
            base_tracker = ExponentialSmoothingTracker(alpha=self._smoothing_alpha)
        else:
            base_tracker = WelfordTracker()
        self._marginal_loss_tracker: Tracker = copy.deepcopy(base_tracker)
        self._model_loss_tracker: Tracker = copy.deepcopy(base_tracker)
        self._marginal_prediction_tracker: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._importance_trackers: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._variance_trackers: MultiValueTracker = MultiValueTracker(copy.deepcopy(base_tracker))
        self._storage: BaseStorage = storage
        if self._storage is None:
            if dynamic_setting:
                self._storage = GeometricReservoirStorage(store_targets=False, size=100, seed=seed)
            else:
                self._storage = UniformReservoirStorage(store_targets=False, size=100, seed=seed)
        self._imputer: BaseImputer = imputer
        if self._imputer is None:
            self._imputer = MarginalImputer(self._model_function, 'joint', self._storage, seed=seed)

        self.n_inner_samples = n_inner_samples


