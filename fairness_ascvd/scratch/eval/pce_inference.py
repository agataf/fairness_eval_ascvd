import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import copy
import configargparse as argparse
from prediction_utils.pytorch_utils.metrics import StandardEvaluator, FairOVAEvaluator
from prediction_utils.util import yaml_write
from lifelines import KaplanMeierFitter, LogNormalFitter, WeibullFitter


import train_utils
import yaml

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

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv')
parser.add_argument('--result_path', type=str)
parser.add_argument('--base_config_path', type=str)

parser.set_defaults(
    save_outputs=False,
    run_evaluation=True,
    run_evaluation_group_standard=True,
    run_evaluation_group_fair_ova=True,
    print_debug=True,
    save_model_weights=False,
    data_query = '',
    num_epochs = 0
)

args = parser.parse_args()
args = copy.deepcopy(args.__dict__)

os.makedirs(args['result_path'], exist_ok=True)
os.makedirs(os.path.join(args['result_path'], 'all'), exist_ok=True)

cohort = pd.read_csv(args['cohort_path'])
cohort = cohort.assign(sysbp = lambda x: x.rxsbp+x.unrxsbp,
         rxbp = lambda x: (x.rxsbp>0).astype(int))
config_dict = yaml.load(open(args['base_config_path']), Loader=yaml.FullLoader)

coefs = {1: [17.114, 0, 0.94, 0, -18.920, 4.475, 29.291, -6.432, 27.820, -6.087, 0.691, 0, 0.874],
              2: [-29.799, 4.884, 13.54, -3.114, -13.578, 3.149, 2.019, 0, 1.957, 0, 7.574, -1.665, 0.661],
              3: [2.469, 0, 0.302, 0, -0.307, 0, 1.916, 0, 1.809, 0, 0.549, 0, 0.645],
              4: [12.344, 0, 11.853, -2.664, -7.990, 1.769, 1.797, 0, 1.7864, 0, 7.837, -1.795, 0.658]}
mean_risk = {1: 86.61, 2:-29.18, 3:19.54, 4:61.18}
baseline_survival = {1:0.9533, 2:0.9665, 3:0.8954, 4:0.9144}

data_df = (pd.DataFrame({'log(age)': np.log(cohort.age),
                    'log(age)^2': np.log(cohort.age)**2,
                    'log(totchol)': np.log(cohort.totchol),
                    'log(age)*log(totchol)': np.log(cohort.age)*np.log(cohort.totchol),
                    'log(hdlc)': np.log(cohort.hdlc),
                    'log(age)*log(hdlc)': np.log(cohort.age)*np.log(cohort.hdlc),
                    'rxbp*log(sysbp)': cohort.rxbp*np.log(cohort.sysbp),
                    'rxbp*log(age)*log(sysbp)': cohort.rxbp*np.log(cohort.age)*np.log(cohort.sysbp),
                    '(1-rxbp)*log(sysbp)': (1-cohort.rxbp)*np.log(cohort.sysbp),
                    '(1-rxbp)*log(age)*log(sysbp)': (1-cohort.rxbp)*np.log(cohort.age)*np.log(cohort.sysbp),
                    'cursmoke': cohort.cursmoke,
                    'log(age)*cursmoke': cohort.cursmoke*np.log(cohort.age),
                    'diabt126': cohort.diabt126
                   }
                  )
     )

risks = []
for group in [1,2,3,4]:
    relative_risk = (data_df
                     .iloc[np.where(cohort.grp==group)]
                     .multiply(coefs[group])
                     .sum(axis=1)
                     .sub(mean_risk[group])
                     .transform(np.exp)
                    )
    risk = 1 - pow(baseline_survival[group], relative_risk)
    risks.append(risk)
    
risks = pd.concat(risks)

cohort = cohort.assign(is_train = lambda x: np.where((x.fold_id != config_dict.get('fold_id')) & (x.fold_id != "test") 
                                                         & (x.fold_id != "eval"),
                                                         1, 0))
all_weights = get_censoring(cohort, by_group = True, model_type = 'KM')
#df = df.join(all_weights)


output_df_eval = (cohort
           .rename(columns={'fold_id': 'phase',
                            'grp': 'group'})
           .assign(labels = lambda x: x.ascvd_10yr.astype(int),
                   pred_probs = risks,
                   weights = all_weights,
                   model_type = 'original_pce',
                  )
           .filter(['phase', 'pred_probs', 'labels', 'weights', 'group', 'model_type', 'person_id'])
            )

output_df_eval = (train_utils.add_ranges(output_df_eval)
                          .merge(cohort.filter(['person_id', 'ldlc']), how='inner', on='person_id')
                          .assign(relative_risk = lambda x: train_utils.treat_relative_risk(x))
                          .rename(columns={'row_id': 'person_id'})
                      )


output_df_eval.to_parquet(
    os.path.join(args['result_path'], "output_df.parquet"),
    index=False,
    engine="pyarrow"
)

output_df_eval.to_csv(
    os.path.join(args['result_path'], 'all', 'predictions.csv'),
    index=False
)

evaluator = StandardEvaluator(threshold_metrics = config_dict['logging_threshold_metrics'],
                              thresholds = config_dict['logging_thresholds'],
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

# result_df_overall.to_parquet(
#     os.path.join(args['result_path'], "result_df_group_standard_eval.parquet"),
#     engine="pyarrow",
#     index=False
# )

evaluator = FairOVAEvaluator(threshold_metrics = config_dict['logging_threshold_metrics'],
                             thresholds = config_dict['logging_thresholds'])
    
eval_fair_args = {'df': output_df_eval,
                  'label_var': 'labels',
                  'pred_prob_var': 'pred_probs',
                  'weight_var': 'weights',
                  'group_var_name': 'group',
                  'strata_vars': ['phase']}
        
result_df_group_fair_ova = evaluator.get_result_df(**eval_fair_args)

result_df_overall.to_csv(os.path.join(args['result_path'], 'all', 'standard_evaluation.csv'), index=False)
result_df_group_fair_ova.to_csv(os.path.join(args['result_path'], 'all', 'fairness_evaluation.csv'), index=False)

# result_df_group_fair_ova.to_parquet(
#     os.path.join(args['result_path'], "result_df_group_fair_ova.parquet"),
#     engine="pyarrow",
#     index=False
# )