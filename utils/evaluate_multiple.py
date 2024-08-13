import time
import numpy as np
import pandas as pd
from river.metrics import RollingROCAUC


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def evaluate_multiple(models, data, model_names, data_name, num_instances):
    """ Trains and evaluate multiple models at the same time.

    :param models: A list of river classification models.
    :param data: The dataset.
    :param model_names: A list with the model names.
    :param data_name: The name of the dataset.
    :param num_instances: The number of instances to train the models.
    :return: A dict containing the number of nodes, metric_values and learning times for each model for each instance in the form of
    {"n_nodes": {model_name1: [...], ...}, "metric_values": ***, "learn_times": ***}.
    """
    print(f"Starting training {model_names} on {data_name} with seed={data.seed} for {num_instances} instances:\n")
    learn_times = {model_name: [] for model_name in model_names}
    n_nodes = {model_name: [] for model_name in model_names}
    metrics = {model_name: RollingROCAUC() for model_name in model_names}
    metric_values = {model_name: [] for model_name in model_names}
    for (n, (x, y)) in enumerate(data, start=1):
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            # Step 1: Evaluate the model on a metric.
            metrics[model_name].update(y, model.predict_proba_one(x))
            metric_values[model_name].append(round(metrics[model_name].get(), 3))

            # Step 2: Train the model on the current instance and measure time.
            start_time = time.perf_counter()
            model.learn_one(x, y)
            end_time = time.perf_counter()
            learn_time = (end_time - start_time) * 1000
            learn_times[model_name].append(learn_time)

            # Step 3: Append number of current nodes in the model.
            n_nodes[model_name].append(model.n_nodes)

        if n % (num_instances/100) == 0:
            df = create_feedback(model_names, n_nodes, metric_values, learn_times)
            print(f"{n}: \n{df.to_string()} \n")

        if n == num_instances:
            df = create_result(model_names, n_nodes, metric_values, learn_times)
            print(f"\nCompleted training {model_names} on {data_name} for {num_instances} instances.\n")
            print(f"Summary: \n{df.to_string()} \n\n\n")
            break

    return {"n_nodes": n_nodes, "metric_values": metric_values, "learn_times": learn_times}


def create_feedback(model_names, n_nodes, metric_values, learn_times):
    transposed_summaries = {
        "n_nodes": {},
        "metric": {},
        "time per instance (ms)": {}
    }
    for i, model_name in enumerate(model_names):
        transposed_summaries["n_nodes"][model_name] = f"{n_nodes[model_name][-1]}"
        transposed_summaries["metric"][model_name] = f"{round(metric_values[model_name][-1], 3)}"
        transposed_summaries["time per instance (ms)"][model_name] = f"{learn_times[model_name][-1]:.2f}"

    df = pd.DataFrame.from_dict(transposed_summaries, orient='index')

    return df


def create_result(model_names, n_nodes, metric_values, learn_times):
    transposed_summaries = {
        "n_nodes": {},
        "metric": {},
        "time per instance (s)": {}
    }
    for i, model_name in enumerate(model_names):
        transposed_summaries["n_nodes"][model_name] = (f"{int(np.mean(n_nodes[model_name]))}"
                                                       f" +- {int(np.std(n_nodes[model_name]))}")
        transposed_summaries["metric"][model_name] = (f"{round(np.mean(metric_values[model_name]), 3)}"
                                                      f" +- {round(np.std(metric_values[model_name]), 3)}")
        transposed_summaries["time per instance (ms)"][model_name] = (f"{np.mean(learn_times[model_name]):.2f}"
                                                                      f" +- {np.std(learn_times[model_name]):.2f}")

    df = pd.DataFrame.from_dict(transposed_summaries, orient='index')

    return df






