import numpy as np
import pandas as pd
import os

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.pytorch_utils.metrics import *
from collections import ChainMap

class CalibrationEvaluatorNew(CalibrationEvaluator):

    def observation_rate_at_point(
        self,
        point,
        labels,
        pred_probs,
        sample_weight=None,
        model_type="logistic",
        transform=None,
    ):

        df, model = self.get_calibration_density_df(
            labels,
            pred_probs,
            sample_weight=sample_weight,
            model_type=model_type,
            transform=transform,
        )
        
        valid_transforms = ["log", "c_log_log"]
        
        if transform is None:
            point = np.array(point).reshape(-1, 1)
        elif transform in valid_transforms:
            if transform == "log":
                point = np.array(np.log(point)).reshape(-1, 1)
            elif transform == "c_log_log":
                point = np.array(self.c_log_log(point)).reshape(-1, 1)
        else:
            raise ValueError("Invalid transform provided")
        
        calibration_density = model.predict_proba(point)
        if len(calibration_density.shape) > 1:
            calibration_density = calibration_density[:, -1]
            
        return calibration_density[0]

class StandardEvaluatorNew(StandardEvaluator):
    def get_threshold_metrics(
        self,
        threshold_metrics=None,
        thresholds=[0.01, 0.05, 0.1, 0.2, 0.5],
        weighted=False,
    ):
        """
        Returns a set of metric functions that are defined with respect to a set of thresholds
        """
        if thresholds is None:
            return {}

        if threshold_metrics is None:
            threshold_metrics = [
                "recall",
                "precision",
                "specificity",
            ]  # acts as default value

        result = {}

        if "recall" in threshold_metrics:
            result["recall"] = {
                "recall_{}".format(threshold): generate_recall_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }
        if "precision" in threshold_metrics:
            result["precision"] = {
                "precision_{}".format(threshold): generate_precision_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }
        if "specificity" in threshold_metrics:
            result["specificity"] = {
                "specificity_{}".format(threshold): generate_specificity_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }
        if "observation_rate" in threshold_metrics:
            result["observation_rate"] = {
                "observation_rate_{}".format(threshold): generate_observation_rate_at_threshold(
                    threshold, weighted=weighted
                )
                for threshold in thresholds
            }  
            
        if len(result) > 0:
            return dict(ChainMap(*result.values()))
        else:
            return result
        
def generate_observation_rate_at_threshold(threshold, weighted=False):
    """
    Returns a lambda function that computes the specificity at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    return (
            lambda labels, pred_probs, sample_weight: (
             observation_rate_at_point(threshold, labels, pred_probs,
                                                      sample_weight,
                                                     model_type="logistic", 
                                                     transform='log')))

def observation_rate_at_point(*args, **kwargs):
    evaluator = CalibrationEvaluatorNew()
    return evaluator.observation_rate_at_point(*args, **kwargs)


grp_label_dict = {1: "Black women", 2: "White women", 3: "Black men", 4: "White men"}

args = {
    "cohort_path": "/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv",
    "base_path": "/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts",
    "eval_fold": "test",
}

preds_all = []
eqodds_threshold = 0.1
for experiment in ['original_pce', 'revised_pce', 'apr14_erm', 'apr14_erm_recalib', 'scratch_thr']:
    aggregate_path = os.path.join(args['base_path'], 'experiments', 
                                  experiment, 'performance',
                                  'all')
    preds_path = os.path.join(aggregate_path, 'predictions.csv')

    preds = pd.read_csv(preds_path)
    if 'model_id' not in preds.columns:
        preds = preds.assign(model_id=0)
    if 'fold_id' not in preds.columns:
        preds = preds.assign(fold_id=0)
    if experiment in ['apr14_mmd', 'apr14_thr', 'scratch_thr']:
        preds = preds.query('model_id >= @eqodds_threshold')
        
    preds_all.append(preds) 
preds_all = pd.concat(preds_all)

evaluator = StandardEvaluatorNew(thresholds = [0.075, 0.2],
                                              metrics = ['auc', 'auprc', 'ace_rmse_logistic_log', 'ace_rmse_bin_log', 'loss_bce'],
                                         threshold_metrics=['observation_rate', 'specificity', 'recall'])

        
result_df_ci = evaluator.bootstrap_evaluate(
    df=preds_all.query("phase=='test'"),
    n_boot=1000,
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


model_type_names = {'original_pce': 'PCE',
                    'revised_pce': 'rPCE',
                    'erm': 'UC',
                    'recalib_erm': 'rUC',
                    'eqodds_thr': 'EO'
                   }

metric_names = {'auc':                    'auc', 
                'auprc':                  'auprc',
                'ace_rmse_logistic_log':  'ace',
                'ace_rmse_bin_log':       'ace_bin',
                'loss_bce':               'loss',
                'recall_0.075':           'sensitivity',
                'recall_0.2':             'sensitivity',
                'specificity_0.075':      'specificity',
                'specificity_0.2':        'specificity',
                'observation_rate_0.075': 'impl_threshold',
                'observation_rate_0.2':   'impl_threshold'
                
               }

thresholds = {'recall_0.075':           0.075,
              'recall_0.2':             0.2,
              'specificity_0.075':      0.075,
              'specificity_0.2':        0.2,
              'observation_rate_0.075': 0.075,
              'observation_rate_0.2':   0.2
             }

plot_df = (result_df_ci
           .assign(model_type = lambda x: x.model_type.map(model_type_names),
                   thresholds = lambda x: x.metric.map(thresholds),
                   metric = lambda x: x.metric.map(metric_names)
                  )
          )

model_type = np.where((plot_df.model_type=='EO') & (plot_df.model_id==0.1), 'EO1', plot_df.model_type)
model_type = np.where((plot_df.model_type=='EO') & (plot_df.model_id==0.21544346900318825), 'EO2', model_type)
model_type = np.where((plot_df.model_type=='EO') & (plot_df.model_id==0.4641588833612778), 'EO3', model_type)
model_type = np.where((plot_df.model_type=='EO') & (plot_df.model_id==1.0), 'EO4', model_type)


plot_df = (plot_df
           .assign(model_type = pd.Categorical(model_type, 
                                               categories = ['PCE', 'rPCE', 'UC', 'rUC', 'EO1', 'EO2', 'EO3', 'EO4'],
                                               ordered=True)
                  )
           .drop(columns = ['model_id'])
          )


aggregate_path_all = '/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/experiments/bmj_manuscript/'
os.makedirs(aggregate_path_all, exist_ok=True)
result_df_ci.to_csv(os.path.join(aggregate_path_all, 'bootstrap_standard_eval_raw_new.csv'), index=False)

aggregate_path_all = '/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/experiments/bmj_manuscript/'
os.makedirs(aggregate_path_all, exist_ok=True)

plot_df.to_csv(os.path.join(aggregate_path_all, 'bootstrap_standard_eval_new.csv'), index=False)

