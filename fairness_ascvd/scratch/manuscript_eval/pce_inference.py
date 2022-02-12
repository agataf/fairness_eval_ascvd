import pandas as pd
import os
import numpy as np
import argparse
from prediction_utils.pytorch_utils.metrics import StandardEvaluator, FairOVAEvaluator
import utils
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument('--base_path', type=str, default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts')
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv')
parser.add_argument('--logging_threshold_metrics', default=['sensitivity', 'specificity'])
parser.add_argument('--logging_thresholds', default=[0.075, 0.2])
parser.add_argument('--run_evaluation', default=False)

args = parser.parse_args()
args = copy.deepcopy(args.__dict__)

result_path = os.path.join(args['base_path'], 'experiments', args['experiment_name'], 'performance')

os.makedirs(result_path, exist_ok=True)
os.makedirs(os.path.join(result_path, 'all'), exist_ok=True)

cohort = pd.read_csv(args['cohort_path'])
cohort = cohort.assign(sysbp = lambda x: x.rxsbp+x.unrxsbp,
                       rxbp = lambda x: (x.rxsbp>0).astype(int),
                       is_train = lambda x: np.where((x.fold_id != 'eval') & (x.fold_id != "test"),
                                                         1, 0),
                       labels = lambda x: x.ascvd_10yr.astype(int),
                       model_type = args['experiment_name'],
                       weights = lambda x: utils.get_censoring(x, by_group = True, model_type = 'KM'))

if args['experiment_name'] == 'original_pce':
    risks = utils.run_pce_model(cohort)
elif args['experiment_name'] == 'revised_pce':
    risks = utils.run_revised_pce_model(cohort)

output_df_eval = (cohort
                  .rename(columns={'fold_id': 'phase',
                                   'grp': 'group'})
                  .join(risks)
                  .assign(treat = lambda x: utils.add_ranges(x),
                         relative_risk = lambda x: utils.treat_relative_risk(x)
                         )
                  #.rename(columns={'row_id': 'person_id'})
                  .filter(['phase', 'pred_probs', 'labels', 'weights',
                           'group', 'model_type', 'person_id', 'treat',
                          'relative_risk'])
            )



output_df_eval.to_csv(
    os.path.join(result_path, 'all', 'predictions.csv'),
    index=False
)

if args['run_evaluation']:

    evaluator = StandardEvaluator(threshold_metrics = args['logging_threshold_metrics'],
                                  thresholds = args['logging_thresholds'],
                                  metrics = ['auc', 'auprc', 'loss_bce', 
                                             'ace_rmse_logistic_log',
                                             'ace_abs_logistic_log']
                                 )

    eval_general_args = {'df': output_df_eval,
                         'label_var': 'labels',
                         'pred_prob_var': 'pred_probs',
                         'weight_var': 'weights', 
                         'strata_vars': ['phase'],
                         'group_var_name': 'group'}

    result_df_overall = evaluator.get_result_df(**eval_general_args)

    evaluator = FairOVAEvaluator(threshold_metrics = args['logging_threshold_metrics'],
                                 thresholds = args['logging_thresholds'])

    eval_fair_args = {'df': output_df_eval,
                      'label_var': 'labels',
                      'pred_prob_var': 'pred_probs',
                      'weight_var': 'weights',
                      'group_var_name': 'group',
                      'strata_vars': ['phase']}

    result_df_group_fair_ova = evaluator.get_result_df(**eval_fair_args)

    result_df_overall.to_csv(
        os.path.join(result_path, 
                     'all',
                     'standard_evaluation.csv'
                     ),
        index=False
    )
    result_df_group_fair_ova.to_csv(
        os.path.join(result_path,
                     'all',
                     'fairness_evaluation.csv'
                    ),
        index=False
    )