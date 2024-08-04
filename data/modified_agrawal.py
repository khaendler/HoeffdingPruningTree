from typing import Optional, Callable
import numpy as np

from river.datasets.synth import Agrawal


class CustomAgrawal(Agrawal):
    """Custom Agrawal stream generator which can change to a specified classification function.

    Parameters
    ----------
    classification_function
        The classification function to use for the generation.
        Valid values are from 0 to 9.
    seed
        Random seed for reproducibility.
    balance_classes
        If True, the class distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. Each new
        sample will be perturbed by the magnitude of `perturbation`.
        Valid values are in the range [0.0 to 1.0].
    """

    def __init__(
        self,
        classification_function: int = 1,
        seed: int | None = None,
        balance_classes: bool = False,
        perturbation: float = 0.0,
    ):
        super().__init__(classification_function, seed, balance_classes, perturbation)

    def generate_drift(self, new_function: int | None = None):
        """
        Generate drift by switching to the given classification function.
        If none is given, the classification function is switched randomly.

        Parameters
        ----------
        new_function
            A dictionary of features.

        """
        if new_function is not None:
            if new_function not in range(10):
                raise ValueError(
                    f"classification_function takes values from 0 to 9 "
                    f"and {new_function} was passed"
                )
            self.classification_function = new_function
        else:
            super().generate_drift()


class SuddenDriftAgrawal(CustomAgrawal):
    """Agrawal stream generator with a user-specified sudden concept drift.

    Parameters
    ----------
    classification_function
        The classification function to use for the generation.
        Valid values are from 0 to 9.
    drift_classification_function
        The classification function to use for the sudden concept drift.
    drift_instance
        The instance at which the sudden drift should be performed.
    seed
        Random seed for reproducibility.
    balance_classes
        If True, the class distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. Each new
        sample will be perturbed by the magnitude of `perturbation`.
        Valid values are in the range [0.0 to 1.0].
    """
    def __init__(
        self,
        classification_function: int = 1,
        drift_classification_function: int = 2,
        drift_instance: int = 1000,
        seed: Optional[int] = None,
        balance_classes: bool = False,
        perturbation: float = 0.0,
    ):
        super().__init__(classification_function, seed, balance_classes, perturbation)
        self.drift_instance = drift_instance
        self.drift_classification_function = drift_classification_function

    def __iter__(self):
        self._rng = np.random.RandomState(self.seed)
        self._next_class_should_be_zero = False
        self.instance_counter = 0

        while True:
            y = 0
            desired_class_found = False
            self.instance_counter += 1
            while not desired_class_found:
                salary = 20000 + 130000 * self._rng.random()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self._rng.random())
                age = self._rng.randint(20, 80)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(1, 20)
                zipcode = self._rng.randint(0, 8)
                hvalue = (8 - zipcode) * 100000 * (0.5 + self._rng.random())
                hyears = self._rng.randint(1, 30)
                loan = self._rng.random() * 500000

                # Switches to the desired classification function
                if self.instance_counter == self.drift_instance:
                    self.generate_drift(self.drift_classification_function)

                y = self._classification_functions[self.classification_function](
                    salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
                )

                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (y == 0)) or (
                            (not self._next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = np.round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = np.round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y


class RecurringDriftAgrawal(CustomAgrawal):
    """Agrawal stream generator with a user-specified recurring concept drift. The recurring concept occurs after every
    second drift.

    Parameters
    ----------
    classification_function
        The beginning classification function to use for the generation.
        Valid values are from 0 to 9.
    recurring_drift_classification_function
        The classification function to reoccur after a drift.
    drift_interval
        The interval at which a concept drift should be performed.
    seed
        Random seed for reproducibility.
    balance_classes
        If True, the class distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. Each new
        sample will be perturbed by the magnitude of `perturbation`.
        Valid values are in the range [0.0 to 1.0].
    """
    def __init__(
        self,
        classification_function: int = 1,
        recurring_drift_classification_function: int = 2,
        drift_interval: int = 10000,
        seed: Optional[int] = None,
        balance_classes: bool = False,
        perturbation: float = 0.0,
    ):
        super().__init__(classification_function, seed, balance_classes, perturbation)
        self.recurring_drift_classification_function = recurring_drift_classification_function
        self.drift_interval = drift_interval

    def __iter__(self):
        self._rng = np.random.RandomState(self.seed)
        self._next_class_should_be_zero = False
        self.currently_recurring = False
        self.instance_counter = 0

        while True:
            y = 0
            desired_class_found = False
            self.instance_counter += 1
            while not desired_class_found:
                salary = 20000 + 130000 * self._rng.random()
                commission = 0 if (salary >= 75000) else (10000 + 75000 * self._rng.random())
                age = self._rng.randint(20, 80)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(1, 20)
                zipcode = self._rng.randint(0, 8)
                hvalue = (8 - zipcode) * 100000 * (0.5 + self._rng.random())
                hyears = self._rng.randint(1, 30)
                loan = self._rng.random() * 500000
                if (self.instance_counter % self.drift_interval) == 0:
                    # Switches to the recurring classification function
                    if not self.currently_recurring:
                        self.generate_drift(self.recurring_drift_classification_function)
                        self.currently_recurring = True
                    # Switches back to a random classification function after drift_interval instances.
                    else:
                        self.generate_drift()
                        self.currently_recurring = False

                y = self._classification_functions[self.classification_function](
                    salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
                )
                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (y == 0)) or (
                            (not self._next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = np.round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = np.round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y


class FeatureDriftAgrawal(CustomAgrawal):
    """Agrawal stream generator with a user-specified virtual/feature concept drift.

    Parameters
    ----------
    classification_function
        The beginning classification function to use for the generation.
        Valid values are from 0 to 9.
    seed
        Random seed for reproducibility.
    balance_classes
        If True, the class distribution will converge to a uniform distribution.
    perturbation
        The probability that noise will happen in the generation. Each new
        sample will be perturbed by the magnitude of `perturbation`.
        Valid values are in the range [0.0 to 1.0].
    salary_drift, commission_drift, age_drift, hvalue_drift, hyears_drift, loan_drift
        Optional user-specified functions to introduce feature drift in respective attributes.
        Each function should take three parameters:
            - num_instances (int): The number of instance of the stream.
            - instance_idx (int): The index of the current instance.
            - rng (np.random.RandomState): A random state for reproducibility.
        Each function should return the modified value for the respective attribute.
        If None, no additional drift is applied to that attribute.
    """
    def __init__(
        self,
        classification_function: int = 1,
        seed: Optional[int] = None,
        balance_classes: bool = False,
        perturbation: float = 0.0,
        num_instances: int = 1000000,
        salary_drift: Optional[Callable[[int, int, np.random.RandomState], float]] = None,
        commission_drift: Optional[Callable[[int, int, np.random.RandomState], float]] = None,
        age_drift: Optional[Callable[[int, int, np.random.RandomState], int]] = None,
        hvalue_drift: Optional[Callable[[int, int, np.random.RandomState], float]] = None,
        hyears_drift: Optional[Callable[[int, int, np.random.RandomState], int]] = None,
        loan_drift: Optional[Callable[[int, int, np.random.RandomState], float]] = None
    ):
        super().__init__(classification_function, seed, balance_classes, perturbation)
        self._rng = np.random.RandomState(self.seed)
        self.num_instances = num_instances
        self.salary_drift = salary_drift
        self.commission_drift = commission_drift
        self.age_drift = age_drift
        self.hvalue_drift = hvalue_drift
        self.hyears_drift = hyears_drift
        self.loan_drift = loan_drift

    def __iter__(self):
        self._next_class_should_be_zero = False
        self.instance_counter = 0

        while True:
            y = 0
            desired_class_found = False
            while not desired_class_found:
                self.instance_counter += 1

                salary = self._assign_value(20000 + 130000 * self._rng.random(),
                                            self.salary_drift, self.instance_counter, self._rng)
                commission = self._assign_value(0 if (salary >= 75000) else (10000 + 75000 * self._rng.random()),
                                                self.commission_drift, self.instance_counter, self._rng)
                age = self._assign_value(self._rng.randint(20, 80), self.age_drift, self.instance_counter, self._rng)
                elevel = self._rng.randint(0, 4)
                car = self._rng.randint(1, 20)
                zipcode = self._rng.randint(0, 8)
                hvalue = self._assign_value((8 - zipcode) * 100000 * (0.5 + self._rng.random()),
                                            self.hvalue_drift, self.instance_counter, self._rng)
                hyears = self._assign_value(self._rng.randint(1, 30), self.hyears_drift,
                                            self.instance_counter, self._rng)
                loan = self._assign_value(self._rng.random() * 500000, self.loan_drift,
                                          self.instance_counter, self._rng)

                y = self._classification_functions[self.classification_function](
                    salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
                )
                if not self.balance_classes:
                    desired_class_found = True
                else:
                    if (self._next_class_should_be_zero and (y == 0)) or (
                            (not self._next_class_should_be_zero) and (y == 1)
                    ):
                        desired_class_found = True
                        self._next_class_should_be_zero = not self._next_class_should_be_zero

            if self.perturbation > 0.0:
                salary = self._perturb_value(salary, 20000, 150000)
                if commission > 0:
                    commission = self._perturb_value(commission, 10000, 75000)
                age = np.round(self._perturb_value(age, 20, 80))
                hvalue = self._perturb_value(hvalue, (9 - zipcode) * 100000, 0, 135000)
                hyears = np.round(self._perturb_value(hyears, 1, 30))
                loan = self._perturb_value(loan, 0, 500000)

            x = dict()
            for feature in self.feature_names:
                x[feature] = eval(feature)

            yield x, y

    def _assign_value(self, default_value, drift_function, instance_counter, rng):
        if drift_function is None:
            return default_value
        else:
            return drift_function(self.num_instances, instance_counter, rng)


