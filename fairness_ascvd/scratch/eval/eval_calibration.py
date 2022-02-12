import pandas as pd
import os
import numpy as np
import logging
import sys
import torch
import copy
import yaml
import random
import argparse

from prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
    FairOVAEvaluator,
    CalibrationEvaluator
)

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("ticks")

grp_label_dict = {1: 'Black women', 2: 'White women', 3: 'Black men', 4: 'White men'} 

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str)
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv')
parser.add_argument('--base_path', type=str, default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts')
parser.add_argument('--n_bootstrap', type=int, default=1)
parser.add_argument('--eval_fold', type=str, default='test')
args = parser.parse_args()

aggregate_path = os.path.join(args.base_path, 'experiments', 
                              args.experiment_name, 'performance',
                              'all')

preds_path = os.path.join(aggregate_path, 'predictions.csv')

os.makedirs(os.path.join(aggregate_path, 'calibration' ,'logx_logreg'), exist_ok=True)
    
preds = pd.read_csv(preds_path)
if 'fold_id' not in preds.columns:
    preds = preds.assign(fold_id=0)
if 'model_id' not in preds.columns:
    preds = preds.assign(model_id=0)

def get_calib_probs(model, x, transform=None):
    
    if transform=='log':
        model_input = np.log(x)
    else:
        model_input = x
        
    calibration_density = model.predict_proba(model_input.reshape(-1, 1))[:, -1]
                    
    df = pd.DataFrame({'pred_probs': x,
                       'model_input': model_input,
                       'calibration_density': calibration_density})  
    return df
    
def get_calib_model(labels, pred_probs, weights, transform=None):
    
    evaluator = CalibrationEvaluator()
    _, model = evaluator.get_calibration_density_df(labels, 
                                                    pred_probs,
                                                    weights,
                                                    transform = transform)

    return model

df_to_calibrate = preds[preds.phase==args.eval_fold].reset_index(drop=True)
lin_calibs=[]
thr_calibs=[]
for iter_idx in range(args.n_bootstrap):
    cohort_bootstrap_sample = (df_to_calibrate
                               .groupby(['labels', 'group'])
                               .sample(frac=1, replace=True))
    
    for model_id in preds.model_id.unique():
        for fold_id in preds.fold_id.unique(): 
            for group in [1,2,3,4, 'overall']:
                if group=='all':
                    df = cohort_bootstrap_sample.query("(model_id==@model_id) & (fold_id==@fold_id)")    
                else:
                    df = cohort_bootstrap_sample.query("(group==@group) & (model_id==@model_id) & (fold_id==@fold_id)")
                max_pred_prob = df.pred_probs.values.max()
                    
                loop_kwargs = {'group': group,
                               'fold_id': fold_id,
                               'phase': args.eval_fold,
                               'model_type': preds.model_type.unique()[0],
                               'model_id' : model_id}

                model = get_calib_model(df.labels, df.pred_probs, df.weights, transform='log')
                    
                lin_calib = (get_calib_probs(model, np.append([1e-15], np.linspace(0.025, int(max_pred_prob/0.025)*0.025, int((max_pred_prob)/0.025))), 'log')
                                 .assign(**loop_kwargs))
                lin_calibs.append(lin_calib)
                    
                thr_calib = (get_calib_probs(model, [0.075, 0.2], 'log')
                                 .assign(**loop_kwargs))
                thr_calibs.append(thr_calib)
    print(iter_idx)

lin_calibs = pd.concat(lin_calibs)
lin_calibs.to_csv(os.path.join(aggregate_path, 'calibration' ,'logx_logreg', 'calibration_sensitivity_test_raw.csv'), index=False)

thr_calibs = pd.concat(thr_calibs)
thr_calibs.to_csv(os.path.join(aggregate_path, 'calibration' ,'logx_logreg', 'calibration_sensitivity_thresholds_raw.csv'), index=False)