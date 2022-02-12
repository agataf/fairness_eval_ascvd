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

def log_reg(x):
    return 1/(1+np.exp(-1*x))


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

coefs = {'women': [0.106501, 0.432440, 0.000056, 0.017666, 0.731678, 0.943970, 1.009790, 0.151318,
                   -0.008580, -0.003647, 0.006208, 0.152968, -0.000153, 0.115232, -0.092231, 0.070498,
                   -0.000173, -0.000094, -12.823110],
         'men': [0.064200, 0.482835, -0.000061, 0.038950, 2.055533, 0.842209, 0.895589, 0.193307,
                 0, -0.014207, 0.011609, -0.119460, 0.000025, -0.077214, -0.226771, -0.117749,
                 0.004190, -0.000199, -11.679980]}
                  
groups_dict = {1: 'women', 2: 'women', 3: 'men', 4: 'men'}

data_df = (pd.DataFrame({'sex': cohort.grp.map(groups_dict),
                         'age': cohort.age,
                         'black': cohort.race_black,
                         'sysbp^2': cohort.sysbp**2,
                         'sysbp': cohort.sysbp,
                         'rxbp': cohort.rxbp,
                         'diabt': cohort.diabt126,
                         'cursmoke': cohort.cursmoke,
                         'totchol/hdlc': cohort.totchol/cohort.hdlc,
                         'age_if_black': cohort.age*cohort.race_black,#only women
                         'sysbp_if_rxbp': cohort.sysbp*cohort.rxbp,
                         'sysbp_if_black': cohort.sysbp*cohort.race_black,
                         'black_and_rxbp': cohort.rxbp*cohort.race_black, 
                         'age*sysbp': cohort.age*cohort.sysbp, 
                         'black_and_diabt': cohort.diabt126*cohort.race_black, 
                         'black_and_cursmoke': cohort.cursmoke*cohort.race_black,
                         'totchol/hdlc_if_black': cohort.totchol/cohort.hdlc*cohort.race_black,
                         'sysbp_if_black_and_rxbp': cohort.sysbp*cohort.rxbp*cohort.race_black,
                         'age*sysbp_if_black': cohort.sysbp*cohort.age*cohort.race_black}
                       )
           .assign(intercept=1)
     )

risks = []
for sex in ['women','men']:
    risk = (data_df
            .query("sex==@sex")
            .drop(columns='sex')
            .multiply(coefs[sex])
            .sum(axis=1)
            .apply(lambda x:log_reg(x))
                    )
    risks.append(risk)
    
risks = pd.concat(risks).sort_index()
risks.name='pred_probs'

cohort = cohort.assign(is_train = lambda x: np.where((x.fold_id != 'eval') & (x.fold_id != "test") 
                                                         & (x.fold_id != "eval"),
                                                         1, 0),
                       labels = lambda x: x.ascvd_10yr.astype(int),
                       model_type = 'revised_pce')

all_weights = get_censoring(cohort, by_group = True, model_type = 'KM').sort_index()

output_df_eval = (cohort
                  .rename(columns={'fold_id': 'phase',
                                   'grp': 'group'})
                  .join(all_weights)
                  .join(risks)
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


eval_args = {'label_var': 'labels',
             'pred_prob_var': 'pred_probs',
             'weight_var': 'weights', 
             'strata_vars': ['phase'],
             'group_var_name': 'group'}

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
