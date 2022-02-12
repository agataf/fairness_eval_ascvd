import pandas as pd
import os
import numpy as np
import eval_utils
import argparse
from prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
    FairOVAEvaluator,
)

def add_ranges(df, one_hot=False, threshold1 = 0.075, threshold2 = 0.2):
    
    range1 = (df.pred_probs < threshold1).astype(int)
    range2 = ((df.pred_probs >= threshold1) & (df.pred_probs < threshold2)).astype(int)
    range3 = ((df.pred_probs >= threshold2)).astype(int)

    if one_hot:
        df = df.assign(treat0=range1, treat1=range2, treat2=range3)
    else:
        rang = 1*range2 + 2*range3
        df = df.assign(treat=rang)
        
    return df


parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--new_model_type", type=str, required=True)
parser.add_argument("--new_model_id", type=str, required=True)
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv')
parser.add_argument("--base_path", type=str, required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts')
parser.add_argument("--recalibrate_on_fold", type=str, required=False,
                   default='eval')
args = parser.parse_args()

aggregate_path = os.path.join(args.base_path, 'experiments', 
                              args.experiment_name, 'performance',
                              'all')
new_experiment_name = '_'.join((args.experiment_name, 'recalib'))
new_aggregate_path = os.path.join(args.base_path, 'experiments', 
                              new_experiment_name, 'performance',
                              'all')
os.makedirs(new_aggregate_path, exist_ok = True)

preds = pd.read_csv(os.path.join(aggregate_path, 'predictions.csv'))

#lin_calibs=[]
test_calibs=[]
for group in [1,2,3,4]:
    for fold_id in range(1,11):    
        max_pred_prob = preds.query("(group==@group)").pred_probs.values.max()
        group_df = preds.query("(group==@group) & (fold_id==@fold_id)")
        group_test = group_df.query("phase=='test'").reset_index(drop=True)
        
        group_recalib = group_df[group_df.phase == args.recalibrate_on_fold].reset_index(drop=True)
        
        model = eval_utils.get_calib_model(group_recalib, transform='log')

#         lin_calib = (eval_utils.get_calib_probs(model, 
#                                      np.linspace(1e-15, max_pred_prob, 30),
#                                      'log')
#                      .assign(group=group)
#                     )
        
        test_calib = (eval_utils.get_calib_probs(model, 
                                      group_test.pred_probs.values, 
                                      'log')
                      .merge(group_test)
                      .drop(['pred_probs', 'model_input'], axis=1)
                      .rename(columns={'calibration_density': 'pred_probs'})
                     )
        print(group, fold_id, test_calib.shape, group_test.shape)


        #lin_calibs.append(lin_calib)
        test_calibs.append(test_calib)
        
#lin_calibs = pd.concat(lin_calibs)
test_calibs = pd.concat(test_calibs)
print(test_calibs.groupby(['fold_id','labels']).count())

    
test_calibs = (add_ranges(test_calibs).assign(model_type = args.new_model_type,
                                              model_id = args.new_model_id))

print(test_calibs.groupby(['fold_id','labels']).count())

standard_evaluator = StandardEvaluator()
fair_evaluator = FairOVAEvaluator()

standard_eval = standard_evaluator.get_result_df(df = test_calibs, 
                                                 weight_var = 'weights')

fair_ova_eval = fair_evaluator.get_result_df(df = test_calibs, 
                                             weight_var = 'weights')

standard_eval.to_csv(os.path.join(new_aggregate_path, 'standard_evaluation.csv'), index=False)
fair_ova_eval.to_csv(os.path.join(new_aggregate_path, 'fairness_evaluation.csv'), index=False)
test_calibs.to_csv(os.path.join(new_aggregate_path, 'predictions.csv'), index=False)

print(os.path.join(new_aggregate_path, 'predictions.csv'))