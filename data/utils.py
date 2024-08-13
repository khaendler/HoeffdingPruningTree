from data.modified_agrawal import *
from data.modified_concept_drift_stream import GradualConceptDriftStream
from data.modified_sea import SuddenDriftSEA
from river.datasets.synth import Hyperplane


def salary_drift(num_instances: int, instance_idx: int, rng: np.random.RandomState):
    """ Creates an incremental drift for Agrawal by adding up to 20k to the salary.

    :param num_instances: Number of instances in the dataset.
    :param instance_idx: Idx of the current instance.
    :param rng: An object of np.random.RandomState.
    :return:
    """
    if num_instances < 40000:
        raise ValueError(f"Got {num_instances} num_instances for salary_drift, but at least 40000 are needed.")
    if instance_idx >= int(num_instances / 4):
        inflation_value = min((40000 * instance_idx / num_instances) - 10000, 20000)
        return inflation_value + 20000 + 130000 * rng.random()
    return 20000 + 130000 * rng.random()


def load_thesis_datasets(num_instances=1000000, pertubation=0, seed=None):
    """ Loads the datasets used in the thesis.

    :param num_instances: Number of instances in the dataset.
    :param pertubation: Noise percentage.
    :param seed: Random seed for reproducibility.
    :return: A list containing the 8 datasets used in the thesis.
    """
    base = CustomAgrawal(classification_function=1, perturbation=pertubation, seed=seed)
    sudden = SuddenDriftAgrawal(classification_function=1, drift_classification_function=2,
                                drift_instance=int(num_instances/2), perturbation=pertubation, seed=seed)

    _agrawal1 = CustomAgrawal(classification_function=1, perturbation=pertubation, seed=seed)
    _agrawal2 = CustomAgrawal(classification_function=2, perturbation=pertubation, seed=seed)
    gradual = GradualConceptDriftStream(_agrawal1, _agrawal2, seed=seed, position=int(num_instances / 2),
                                        width=int(num_instances/10))

    feature = FeatureDriftAgrawal(classification_function=3, salary_drift=salary_drift,
                                  perturbation=pertubation, seed=seed, num_instances=num_instances)

    recurring = RecurringDriftAgrawal(classification_function=1, recurring_drift_classification_function=2,
                                      drift_interval=int(num_instances / 4), perturbation=pertubation, seed=seed)

    _sudden1 = SuddenDriftAgrawal(classification_function=1, drift_classification_function=2,
                                  drift_instance=int(num_instances/4), perturbation=pertubation, seed=seed)
    _agrawal2 = CustomAgrawal(classification_function=1, perturbation=pertubation, seed=seed)
    mixed = GradualConceptDriftStream(_sudden1, _agrawal2, seed=seed, position=int(num_instances * 0.75),
                                      width=int(num_instances/10))

    sea = SuddenDriftSEA(variant=1, drift_variant=2, drift_instance=int(num_instances/2), noise=pertubation, seed=seed)
    hyperplane = Hyperplane(n_features=20, n_drift_features=10, mag_change=0.001, noise_percentage=pertubation, seed=seed)

    datasets = [base, sudden, gradual, feature, recurring, mixed, sea, hyperplane]

    return datasets
