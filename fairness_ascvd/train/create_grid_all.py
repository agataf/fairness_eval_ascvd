'''
adapted from group_robustness_fairness/run_tuning.py in https://github.com/som-shahlab/group_robustness_fairness/
'''

import numpy as np
import os
import random
import pandas as pd
import configargparse as argparse
import itertools

from fairness_ascvd.prediction_utils.util import yaml_write
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--data_path", type=str, help="The root data path",
)

parser.add_argument(
    "--experiment_name_prefix",
    type=str,
    default="scratch",
    help="The name of the experiment",
)


parser.add_argument(
    "--grid_size",
    type=int,
    default=None,
    help="The number of elements in the random grid",
)

parser.add_argument(
    "--num_folds", type=int, required=False, default=10,
)

parser.add_argument("--seed", type=int, default=234)


def generate_grid(
    global_tuning_params_dict,
    model_tuning_params_dict=None,
    experiment_params_dict=None,
    grid_size=None,
    seed=None,
):

    the_grid = list(ParameterGrid(global_tuning_params_dict))
    if model_tuning_params_dict is not None:
        local_grid = []
        for i, pair_of_grids in enumerate(
            itertools.product(the_grid, list(ParameterGrid(model_tuning_params_dict)))
        ):
            local_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        the_grid = local_grid

    if grid_size is not None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        np.random.shuffle(the_grid)
        the_grid = the_grid[:grid_size]

    if experiment_params_dict is not None:
        outer_grid = list(ParameterGrid(experiment_params_dict))
        final_grid = []
        for i, pair_of_grids in enumerate(itertools.product(outer_grid, the_grid)):
            final_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        return final_grid
    else:
        return the_grid


if __name__ == "__main__":

    args = parser.parse_args()

    common_grid = {
        "global": {
            "lr": [1e-4],
            "batch_size": [128],
            "num_epochs": [150],
            "gamma": [1.0],
            "early_stopping": [True],
            "early_stopping_patience": [25],
        },
        "model_specific": {
            # added linear_layer
            "logistic_regression": {"num_hidden": [0], "weight_decay": [0], "linear_layer": [True]},
        },
         "experiment": {
            # here, each fold results in a different config file - I'm doing one experiment per fold
            # "fold_id": [str(i + 1) for i in range(args.num_folds)],
         },
    }

    grids = {
        "erm": {
            "global": common_grid["global"],
            "model_specific": common_grid["model_specific"]["logistic_regression"],
            "experiment": {
                **{
                    "group_objective_type": ["standard"],
                    "selection_metric": ["loss"],
                },
            },
        },
        "eq_odds_thr": {
            "global": common_grid["global"],
            "model_specific": common_grid["model_specific"]["logistic_regression"],
            "experiment": {
                **{
                    "lambda_group_regularization": [
                        float(x) for x in np.logspace(-1, 0, num=4)
                    ],
                    "group_objective_type": ["regularized"],
                    "group_objective_metric": ["threshold_rate"],
                    'threshold_mode': ['conditional'],
                    'thresholds': [[0.075, 0.2]],
                    "sensitive_attribute": ['group'],
                    "balance_groups": [False],
                    "selection_metric": ["loss"],
                },
            },
        }
    }           

    for experiment_name, value in grids.items():
        print(experiment_name)
        the_grid = generate_grid(
            grids[experiment_name]["global"],
            grids[experiment_name]["model_specific"],
            grids[experiment_name]["experiment"],
        )
        print("{}, length: {}".format(experiment_name, len(the_grid)))

        experiment_dir_name = "{}_{}".format(
            args.experiment_name_prefix, experiment_name
        )

        grid_df = pd.DataFrame(the_grid)
        config_path = os.path.join(
            args.data_path, "experiments", experiment_dir_name, "config"
        )
        os.makedirs(config_path, exist_ok=True)
        grid_df.to_csv(os.path.join(config_path, "config.csv"), index_label="id")

        for i, config_dict in enumerate(the_grid):
            yaml_write(config_dict, os.path.join(config_path, "{}.yaml".format(i)))