import numpy as np
import pandas as pd
import os

import argparse
import copy

from fairness_ascvd.prediction_utils.pytorch_utils.metrics import StandardEvaluator
from fairness_ascvd.prediction_utils.pytorch_utils.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp_path', type=str, help='path where experimental results are stored')
parser.add_argument('--cohort_path', type=str, help='path where input cohorts are stored')
parser.add_argument('--output_path', type=str, help='path where evaluation results will be stored')
parser.add_argument('--n_boot', type=int, help='number of bootstrap resamplings for evaluation', default=1000)

args = parser.parse_args()

preds_all = []

# only consider values of lambda up to 0.1

model_type_names = {'original_pce': 'PCE',
                    'revised_pce': 'rPCE',
                    'final_erm': 'UC',
                    'final_erm_rec_logit': 'rUC',
                    'final_eq_odds_thr': 'EO'
                   }

eqodds_threshold = 0.1
for experiment in model_type_names.keys():
    aggregate_path = os.path.join(args.exp_path, experiment, 'performance', 'all')
    preds_path = os.path.join(aggregate_path, 'predictions.csv')

    preds = pd.read_csv(preds_path)
    if 'model_id' not in preds.columns:
        preds = preds.assign(model_id=0)
    if 'fold_id' not in preds.columns:
        preds = preds.assign(fold_id=0)
        
    preds = preds.assign(model_type = model_type_names[experiment])
                   
    if experiment == 'EO':
        preds = preds.query('model_id >= @eqodds_threshold')
        
    preds_all.append(preds) 
preds_all = pd.concat(preds_all)


# Fixing inconsistent race coding
cohort = pd.read_csv(args.cohort_path)
df = (cohort
      .assign(grp = np.where((cohort.grp==1) & (cohort.race_black==0), 2,
                              np.where((cohort.grp==3) & (cohort.race_black==0), 4, cohort.grp)
                             )
              )
      .assign(labels=lambda x: x.event_indicator.astype(int))
      .filter(['person_id', 'labels', 'grp'])
     )

preds_all = (preds_all
              .merge(df, on=['person_id', 'labels'], how='left')
              .rename(columns={'group': 'group_old', 'grp': 'group'})
             )

# run evaluation

evaluator = StandardEvaluator(thresholds = [0.075, 0.2],
                                 metrics = ['auc', 
                                            'ace_rmse_logistic_logit',
                                            'loss_bce'],
                                 threshold_metrics = ['specificity',
                                                      'recall',
                                                      'tce_signed_logistic_logit']
                                )

        
result_df_ci = evaluator.bootstrap_evaluate(
    df=preds_all.query("phase=='test'"),
    n_boot=args.n_boot,
    strata_vars_eval=['phase', 'model_type', 'model_id', 'fold_id', 'group'],
    strata_vars_boot=['phase', 'labels', 'group'],
    strata_var_replicate='fold_id',
    replicate_aggregation_mode=None,
    baseline_experiment_name=0,
    strata_var_group='group',
    weight_var='weights',
    n_jobs=-1,
    compute_overall=True,
)

# post-processing to standardize model and metric names for later analysis

metric_names = {'auc':                             'auc', 
                'ace_rmse_logistic_logit':         'ace_logit',
                'loss_bce':                        'loss',
                'recall_0.075':                    'sensitivity',
                'recall_0.2':                      'sensitivity',
                'specificity_0.075':               'specificity',
                'specificity_0.2':                 'specificity',
                'tce_signed_logistic_logit_0.075': 'tce_logit',
                'tce_signed_logistic_logit_0.2':   'tce_logit'
                
               }

thresholds = {'recall_0.075':      0.075,
              'recall_0.2':        0.2,
              'specificity_0.075': 0.075,
              'specificity_0.2':   0.2,
              'tce_signed_logistic_logit_0.075': 0.075,
              'tce_signed_logistic_logit_0.2': 0.2
             }

plot_df = (result_df_ci
           .assign(#model_type = lambda x: x.model_type.map(model_type_names),
                   thresholds = lambda x: x.metric.map(thresholds),
                   metric     = lambda x: x.metric.map(metric_names)
                  )
          )

# update tce_logit to threshold_error, and sensitivity/specificity to fnr/fpr

plot_df = (plot_df
           .append(plot_df.query("metric=='tce_logit'")
                                     .assign(CI_med = lambda x: -1*x.CI_med, 
                                             CI_lower = lambda x: -1*x.CI_lower, 
                                             CI_upper = lambda x: -1*x.CI_upper, 
                                             metric='threshold_error')
                             )
           .append(plot_df.query("metric==['sensitivity', 'specificity']")
                                     .assign(CI_med = lambda x: 1-x.CI_med, 
                                             CI_lower = lambda x: 1-x.CI_lower, 
                                             CI_upper = lambda x: 1-x.CI_upper, 
                                             metric = lambda x: np.where(
                                                 x.metric=='sensitivity', 'fnr', 'fpr'
                                             )
                                            )
                  )
          )


# update model names so they don't have to be referred to by lambda values

model_type = plot_df.model_type
for lambda_val, model_name in zip(np.logspace(-1, 0, num=4), ['EO1', 'EO2', 'EO3', 'EO4']):
    model_type = np.where((plot_df.model_type=='EO') & (plot_df.model_id==lambda_val),
                          model_name,
                          model_type)

plot_df = (plot_df
           .assign(model_type = pd.Categorical(model_type, 
                                               categories = ['PCE', 'rPCE', 
                                                             'UC', 'rUC',
                                                             'EO1', 'EO2',
                                                             'EO3', 'EO4'],
                                               ordered=True)
                  )
           .drop(columns = ['model_id'])
          )

# save evaluation result

os.makedirs(args.output_path, exist_ok=True)
plot_df.to_csv(os.path.join(args.output_path, 'bootstrap_standard_eval.csv'), index=False)


# generate IGSD:

grouped_plot_df = (plot_df
                   .query("metric==['fpr', 'fnr', 'threshold_error']")
                   .drop(columns=['phase', 'CI_lower', 'CI_upper'])
                   .groupby(['model_type', 'thresholds', 'metric', 'group']).sum()
                   .reset_index()
                   .pivot(index=['model_type', 'thresholds', 'metric'], columns='group', values='CI_med')
                   .drop(columns=['overall'])
                  )

means = grouped_plot_df.apply(np.mean, axis=1)
stds = grouped_plot_df.apply(np.std, axis=1)

coeff_var = ((stds/means)
             .to_frame()
             .reset_index()
             .pivot(index=['thresholds','model_type'], columns=['metric'], values=0)
             .reset_index()
             .set_index(['thresholds', 'model_type', 'threshold_error'])
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns = {0: 'Inter group coefficient of variation',
                               'threshold_error': 'IGCV in threshold calibration error',
                               'model_type': 'Model',
                               'metric': 'Metric',
                               'thresholds': 't'})
             .assign(t = lambda x: x.t.map({0.075: '7.5%', 0.2: '20%'}),
                     Metric = lambda x: x.Metric.map({'fnr': 'IGCV in FNR',
                                                      'fpr': 'IGCV in FPR'}
                                                    )
                    )
            )

std_frame = (stds
             .to_frame()
             .reset_index()
             .pivot(index=['thresholds','model_type'], columns=['metric'], values=0)
             .reset_index()
             .set_index(['thresholds', 'model_type', 'threshold_error'])
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns = {0: 'IGSD in error rates',
                               'threshold_error': 'IGSD in TCE',
                               'model_type': 'Model',
                               'metric': 'Metric',
                               'thresholds': 't'})
             .assign(t = lambda x: x.t.map({0.075: '7.5%', 0.2: '20%'}),
                     Metric = lambda x: x.Metric.map({'fnr': 'IGSD in FNR',
                                                      'fpr': 'IGSD in FPR'}
                                                    )
                    )
            )

std_frame.to_csv(os.path.join(args.output_path, 'IGSD_results.csv'), index=False)



