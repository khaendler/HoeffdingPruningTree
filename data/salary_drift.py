from data.modified_agrawal import *


def salary_drift(num_instances: int, instance_idx: int, rng: np.random.RandomState):
    if num_instances < 40000:
        raise ValueError(f"Got {num_instances} num_instances for salary_drift, but at least 40000 are needed.")
    if instance_idx >= int(num_instances / 4):
        inflation_value = min((40000 * instance_idx / num_instances) - 10000, 20000)
        return inflation_value + 20000 + 130000 * rng.random()
    return 20000 + 130000 * rng.random()
