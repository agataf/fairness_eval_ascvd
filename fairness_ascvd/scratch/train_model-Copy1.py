import numpy as np
import pandas as pd
import os
import joblib
import configargparse as argparse
import copy
import torch
import logging
import sys

from prediction_utils.pytorch_utils.models import TorchModel
from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator
from lifelines import KaplanMeierFitter, LogNormalFitter, WeibullFitter

from prediction_utils.util import yaml_write

from prediction_utils.pytorch_utils.group_fairness import group_regularized_model
from prediction_utils.pytorch_utils.robustness import group_robust_model
from prediction_utils.pytorch_utils.lagrangian import group_lagrangian_model

from prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
    FairOVAEvaluator,
)

#from fairness_utility.train import train_utils
import train_utils

parser = argparse.ArgumentParser(config_file_parser_class=argparse.YAMLConfigFileParser)

parser.add_argument("--config_path", required=False, is_config_file=True)

# Path configuration
parser.add_argument(
    "--data_path", type=str, default="", help="The root path where data is stored",
)

# parser.add_argument(
#     "--features_path", type=str, default="", help="The root path where data is stored",
# )

parser.add_argument(
    "--cohort_path",
    type=str,
    default="",
    help="File name for the file containing metadata",
)

# parser.add_argument(
#     "--vocab_path",
#     type=str,
#     default="",
#     help="File name for the file containing feature information",
# )

# parser.add_argument(
#     "--features_row_id_map_path",
#     type=str,
#     default="",
#     help="Maps identifiers in the cohort dataframe to rows of the feature matrix",
# )

parser.add_argument(
    "--logging_path", type=str, default=None, help="A path to store logs",
)

parser.add_argument(
    "--result_path", type=str, required=True, help="A path where results will be stored"
)

# Hyperparameters - training dynamics
parser.add_argument(
    "--num_epochs", type=int, default=10, help="The number of epochs of training"
)

parser.add_argument(
    "--iters_per_epoch",
    type=int,
    default=100,
    help="The number of batches to run per epoch",
)

parser.add_argument("--batch_size", type=int, default=256, help="The batch size")

parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")

parser.add_argument(
    "--gamma", type=float, default=0.95, help="Learning rate decay (exponential)"
)

parser.add_argument(
    "--early_stopping",
    dest="early_stopping",
    action="store_true",
    help="Whether to use early stopping",
)

parser.add_argument("--early_stopping_patience", type=int, default=5)

parser.add_argument(
    "--selection_metric",
    type=str,
    default="loss",
    help="The metric to use for model selection",
)

# Hyperparameters - model architecture
parser.add_argument(
    "--num_hidden", type=int, default=1, help="The number of hidden layers"
)

parser.add_argument(
    "--hidden_dim", type=int, default=128, help="The dimension of the hidden layers"
)

parser.add_argument(
    "--normalize", dest="normalize", action="store_true", help="Use layer normalization"
)

parser.add_argument(
    "--drop_prob", type=float, default=0.25, help="The dropout probability"
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=0
)

parser.add_argument(
    "--weighted_loss", dest="weighted_loss", action="store_true", help="Apply sample weights to loss"
)


# Experiment configuration
parser.add_argument(
    "--fold_id", type=str, default="1", help="The fold id to use for early stopping"
)

parser.add_argument(
    "--label_col",
    type=str,
    default="labels",
    help="The name of a column in cohort to use as the label",
)

parser.add_argument(
    "--data_mode", type=str, default="array", help="Which mode of source data to use",
)

parser.add_argument("--sparse_mode", type=str, default="csr", help="The sparse mode")

parser.add_argument(
    "--num_workers",
    type=int,
    default=5,
    help="The number of workers to use for data loading during training",
)

parser.add_argument(
    "--deterministic",
    dest="deterministic",
    action="store_true",
    help="Whether to use deterministic training",
)

parser.add_argument(
    "--seed", type=int, default=2020, help="The seed",
)

parser.add_argument(
    "--cuda_device", type=int, default=0, help="The cuda device id",
)

parser.add_argument(
    "--print_every", type=int, default=10, help="Print evaluation every ... epochs",
)

parser.add_argument(
    "--logging_metrics",
    type=str,
    nargs="*",
    required=False,
    default=['auc', 'auprc', 'brier', 'loss_bce'],
    help="metrics to use for logging during training",
)

parser.add_argument(
    "--logging_threshold_metrics",
    type=str,
    nargs="*",
    required=False,
    default=['specificity', 'recall', 'positive_rate'],
    help="threshold metrics to use for logging during training",
)

parser.add_argument(
    "--logging_thresholds",
    type=str,
    nargs="*",
    required=False,
    default=[0.075, 0.2],
    help="thresholds to use for threshold-based logging metrics",
)

parser.add_argument(
    "--eval_metrics",
    type=str,
    nargs="*",
    required=False,
    default=['auc', 'auprc', 'brier', 'loss_bce', 'ace_rmse_logistic_log', 'ace_abs_logistic_log'],
    help="metrics to use for evaluation",
)

parser.add_argument(
    "--eval_threshold_metrics",
    type=str,
    nargs="*",
    required=False,
    default=['specificity', 'recall', 'positive_rate'],
    help="threshold metrics to use for evaluation",
)


parser.add_argument(
    "--eval_attributes",
    type=str,
    nargs="+",
    required=False,
    default='group',
    help="The attributes to use to perform a stratified evaluation. Refers to columns in the cohort dataframe",
)

parser.add_argument(
    "--eval_thresholds",
    type=float,
    nargs="+",
    required=False,
    default=[0.075, 0.2],
    help="The thresholds to apply for threshold-based evaluation metrics",
)

parser.add_argument(
    "--weighted_evaluation", action="store_true", help="Evaluate using sample weights"
)

parser.add_argument(
    "--sample_keys", type=str, nargs="*", required=False, default=None, help=""
)

parser.add_argument(
    "--replicate_id", type=str, default="", help="Optional replicate id"
)


## Arguments related to group fairness and robustness

parser.add_argument(
    "--linear_layer",
    dest="linear_layer",
    action="store_true",
    help="Whether to use log reg",
)

parser.add_argument(
    "--sensitive_attribute",
    type=str,
    default=None,
    help="The attribute to be fair with respect to",
)

parser.add_argument(
    "--balance_groups",
    dest="balance_groups",
    action="store_true",
    help="Whether to rebalance the data so that data from each group is sampled with equal probability",
)

parser.add_argument(
    "--group_objective_type",
    type=str,
    default="standard",
    help="""
    Options:
        standard: train with a standard ERM objective over the entire dataset
        regularized: train with a fairness regularizer
        dro: train with a practical group distributionally robust strategy
        lagrangian: train with a practical proxy-Lagrangian strategy
    """,
)

parser.add_argument(
    "--group_objective_metric",
    type=str,
    default="loss",
    help="""The metric used to construct the group objective. 
    Refer to the implementation of group_regularized_model, group_robust_model, and group_lagrangian_model""",
)

## Arguments specific to regularized models
parser.add_argument(
    "--lambda_group_regularization",
    type=float,
    default=1e-1,
    help="The extent to which to penalize group differences in the regularization_metric",
)

parser.add_argument(
    "--mmd_mode",
    type=str,
    default="unconditional",
    help="""
    Specific to MMDModel
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--mean_prediction_mode",
    type=str,
    default="unconditional",
    help="""
    Specific to EqualMeanPredictionModel
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--threshold_mode",
    type=str,
    default="unconditional",
    help="""
    Specific to EqualThresholdRateModel
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="The threshold used for models that rely on threshold-based surrogates",
)

parser.add_argument(
    "--group_regularization_mode",
    type=str,
    default="overall",
    help="The type of group regularization used when comparing a metric across groups. Valid options are `overall` and `group`",
)

## Arguments specific to DRO/Lagrangian
parser.add_argument(
    "--lr_lambda", type=float, default=1e-2, help="The learning rate for DRO"
)

parser.add_argument(
    "--update_lambda_on_val",
    dest="update_lambda_on_val",
    action="store_true",
    help="Whether to update the lambdas on the validation set",
)

# Arguments related to Lagrangian
parser.add_argument(
    "--constraint_slack", type=float, default=0.05, help="The constraint slack",
)

parser.add_argument(
    "--multiplier_bound",
    type=float,
    default=1,
    help="The maximum norm of the lambda vector",
)

parser.add_argument(
    "--thresholds",
    type=float,
    nargs="*",
    required=False,
    default=[0.1],
    help="The threshold used for the MultiLagrangianThresholdModel",
)

parser.add_argument(
    "--constraint_metrics",
    type=str,
    nargs="*",
    required=False,
    default=["tpr", "fpr"],
    help="The metrics to use for the MultiLagrangianThresholdModel",
)

# Arguments related to adversarial models
parser.add_argument(
    "--adversarial_mode",
    type=str,
    default="unconditional",
    help="""
    Options: `conditional`, `unconditional`. Conditional corresponds to equalized odds, unconditional corresponds to demographic parity.
    """,
)

parser.add_argument(
    "--lr_discriminator",
    type=float,
    default=1e-4,
    help="The learning rate for the adversarial discriminator",
)

parser.add_argument(
    "--num_hidden_discriminator",
    type=int,
    default=3,
    help="The number of hidden layers",
)

parser.add_argument(
    "--hidden_dim_discriminator",
    type=int,
    default=128,
    help="The dimension of the hidden layers",
)

parser.add_argument(
    "--drop_prob_discriminator", type=float, default=0.0, help="The dropout probability"
)

parser.add_argument(
    "--spectral_norm",
    dest="spectral_norm",
    action="store_true",
    help="Whether to apply spectral normalization to the discriminator",
)

# Args related to subsetting data
parser.add_argument("--subset_attribute", type=str, default=None)
parser.add_argument("--subset_group", type=str, default=None)

# Boolean arguments
parser.add_argument(
    "--save_outputs",
    dest="save_outputs",
    action="store_true",
    help="Whether to save the outputs of evaluation",
)

parser.add_argument(
    "--save_model_weights",
    dest="save_model_weights",
    action="store_true",
    help="Whether to save the model weights",
)

parser.add_argument(
    "--logging_evaluate_by_group",
    dest="logging_evaluate_by_group",
    action="store_true",
    help="Whether to evaluate the model for each group during training",
)

parser.add_argument(
    "--evaluate_by_group",
    dest="evaluate_by_group",
    action="store_true",
    help="Whether to evaluate the model for each group during evaluation",
)

parser.add_argument(
    "--run_evaluation",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the trained model",
)

parser.add_argument(
    "--run_evaluation_group",
    dest="run_evaluation_group",
    action="store_true",
    help="Whether to evaluate the trained model for each group",
)

parser.add_argument(
    "--run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_true",
    help="Whether to evaluate the model with the standard evaluator",
)

parser.add_argument(
    "--run_evaluation_group_fair_ova",
    dest="run_evaluation_group_fair_ova",
    action="store_true",
    help="Whether to evaluate the model with the group fairness one-vs-all evaluator",
)

parser.add_argument(
    "--print_debug",
    dest="print_debug",
    action="store_true",
    help="Whether to print debugging information",
)

parser.add_argument(
    "--disable_metric_logging",
    dest="disable_metric_logging",
    action="store_true",
    help="Whether to disable metric logging during training",
)

parser.add_argument(
    "--censoring_by_group",
    action="store_true",
    help="Whether model propensity for censoring for each group separately (currently defined as grp variable)",
)

parser.add_argument(
    "--censoring_model_type",
    type=str,
    default='KM',
    help="what model to use for estimating propensity for censoring. Currently, only Kaplan Meier (KM) impemented",
)

parser.set_defaults(
    normalize=False,
    weighted_loss=True,
    weighted_evaluation=True,
    early_stopping=False,
    save_model_weights=False,
    save_outputs=False,
    balance_groups=False,
    logging_evaluate_by_group=False,
    evaluate_by_group=True,
    run_evaluation=True,
    run_evaluation_group=True,
    run_evaluation_group_standard=True,
    run_evaluation_group_fair_ova=True,
    spectral_norm=False,
    deterministic=True,
    update_lambda_on_val=False,
    print_debug=False,
    disable_metric_logging=False,
    censoring_by_group=True,

)


# def filter_cohort(cohort, subset_attribute=None, subset_group=None):
#     # Global filter
#     if "gender_concept_name" in cohort.columns:
#         cohort = cohort.query('gender_concept_name != "No matching concept"')

#     # Custom filter
#     if (subset_attribute is not None) and (subset_group is not None):
#         if not (subset_attribute in cohort.columns):
#             raise ValueError("subset_attribute not in cohort columns")
#         cohort = cohort.query(
#             "{subset_attribute} == '{subset_group}'".format(
#                 subset_attribute=subset_attribute, subset_group=subset_group
#             )
#         )
#     return cohort


def read_file(filename, columns=None, **kwargs):
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns, **kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)
    
def censoring_weights(df, model_type = 'KM'):

    if model_type == 'KM':
        censoring_model = KaplanMeierFitter()
    else:
        raise ValueError("censoring_model not defined")
    
    censoring_model.fit(df.query('is_train==1').event_time, 1.0*~df.query('is_train==1').event_indicator)
    
    weights = 1 / censoring_model.survival_function_at_times(df.event_time_10yr.values - 1e-5)
    weights_dict = dict(zip(df.index.values, weights.values))
    return weights_dict

def get_censoring(df, by_group=True, model_type='KM'):
    
    if by_group:
        weight_dict = {}
        for group in [1, 2, 3, 4]:
            group_df = df.query('grp==@group')
            group_weights_dict = censoring_weights(group_df, model_type)
            weight_dict.update(group_weights_dict)
            
    else:
        weight_dict = censoring_weights(cohort, censoring_model_type)

    weights = pd.Series(weight_dict, name='weights') 
    return weights


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.result_path, exist_ok=True)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.deterministic:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config_dict = copy.deepcopy(args.__dict__)

    logger = logging.getLogger(__name__)
    if config_dict.get("logging_path") is not None:
        logging.basicConfig(
            filename=config_dict.get("logging_path"),
            level="DEBUG" if args.print_debug else "INFO",
            format="%(message)s",
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level="DEBUG" if args.print_debug else "INFO",
            format="%(message)s",
        )

    cohort = read_file(args.cohort_path)

    logging.info("Result path: {}".format(args.result_path))
    os.makedirs(args.result_path, exist_ok=True)

    cohort = cohort.assign(is_train = lambda x: np.where((x.fold_id != config_dict.get('fold_id')) & (x.fold_id != "test"),
                                                         1, 0))

    all_weights = get_censoring(cohort, by_group = args.censoring_by_group, model_type = args.censoring_model_type)
    cohort = cohort.join(all_weights)

    # TODO: replace with Stephen's loader eventually
    data = train_utils.Dataset(cohort, deg=2,
                               feature_columns = ['age', 'totchol', 'hdlc', 'sysbp', 'rxsbp', 'unrxsbp', 'bmi', 'diabt126', 'cursmoke', 'race_black', 'gender_male'],
                               val_fold_id = config_dict.get('fold_id'),
                               test_fold_id = 'test',
                               batch_size = config_dict.get('batch_size'))

    config_dict.update({'input_dim': data.features_dict_uncensored_scaled['train'].shape[1]})

    if args.group_objective_type == "standard":
        model_class = TorchModel
    else:
        assert args.sensitive_attribute is not None
        if args.group_objective_type == "regularized":
            model_class = group_regularized_model(config_dict["group_objective_metric"])
        elif args.group_objective_type == "dro":
            model_class = group_robust_model(config_dict["group_objective_metric"])
        elif args.group_objective_type == "lagrangian":
            model_class = group_lagrangian_model(config_dict["group_objective_metric"])
        else:
            raise ValueError("group_objective_type not defined")

    model = model_class(**config_dict)

    logging.info(model.config_dict)

    # Write the resulting config
    yaml_write(config_dict, os.path.join(args.result_path, "config.yaml"))
    
    result_df = model.train(data.loaders_dict)["performance"]

    # Dump training results to disk
    result_df.to_parquet(
        os.path.join(args.result_path, "result_df_training.parquet"),
        index=False,
        engine="pyarrow",
    )

    if args.save_model_weights:
        torch.save(model.model.state_dict(), os.path.join(args.result_path, "state_dict.pt"))

    if args.run_evaluation:
        logging.info("Evaluating model")
        predict_dict = model.predict(data.loaders_dict_predict, phases=['val', 'test'])

        output_df_eval, result_df_eval = (
            predict_dict["outputs"],
            predict_dict["performance"],
        )
        logging.info(result_df_eval)
                               
        output_df_eval = (train_utils.add_ranges(output_df_eval)
                          .rename(columns={'row_id': 'person_id'})
                          .merge(cohort.filter(['person_id', 'ldlc']), how='inner', on='person_id')
                          .assign(relative_risk = lambda x: train_utils.treat_relative_risk(x))
                      )

        # Dump evaluation result to disk
        result_df_eval.to_parquet(
            os.path.join(args.result_path, "result_df_training_eval.parquet"),
            index=False,
            engine="pyarrow",
        )
        if args.save_outputs:
            output_df_eval.to_parquet(
                os.path.join(args.result_path, "output_df.parquet"),
                index=False,
                engine="pyarrow",
            )
        if args.run_evaluation_group:
            logging.info("Running evaluation on groups")
            if args.eval_attributes is None:
                raise ValueError(
                    "If using run_evaluation_group, must specify eval_attributes"
                )

            if args.run_evaluation_group_standard:

                evaluator = StandardEvaluator(threshold_metrics = args.eval_threshold_metrics, thresholds=args.eval_thresholds, metrics=args.eval_metrics)
                result_df_group_standard_eval = evaluator.get_result_df(
                    df = output_df_eval, 
                    weight_var = 'weights'
                )
                logging.info(result_df_group_standard_eval)
                result_df_group_standard_eval.to_parquet(
                    os.path.join(
                        args.result_path, "result_df_group_standard_eval.parquet"
                    ),
                    engine="pyarrow",
                    index=False,
                )
            if args.run_evaluation_group_fair_ova:
                evaluator = FairOVAEvaluator()
                result_df_group_fair_ova = evaluator.get_result_df(
                    df = output_df_eval, 
                    weight_var = 'weights'
                )
                logging.info(result_df_group_fair_ova)
                result_df_group_fair_ova.to_parquet(
                    os.path.join(args.result_path, "result_df_group_fair_ova.parquet"),
                    engine="pyarrow",
                    index=False,
                )