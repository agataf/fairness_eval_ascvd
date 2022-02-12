import pandas as pd
import os
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, choices=['original_pce', 'revised_pce'])
parser.add_argument('--cohort_path', type=str, help='path where input cohorts are stored')
parser.add_argument('--result_path', type=str, help='path where inference results will be stored')

args = parser.parse_args()

cohort = pd.read_csv(args.cohort_path)

# preprocess data to put it in a model-compatible format

cohort = cohort.assign(sysbp = lambda x: x.rxsbp+x.unrxsbp,
                       rxbp = lambda x: (x.rxsbp>0).astype(int),
                       is_train = lambda x: np.where((x.fold_id != 'eval') & (x.fold_id != 'test'),
                                                         1, 0),
                       labels = lambda x: x.ascvd_10yr.astype(int),
                       model_type = args.experiment_name,
                       weights = lambda x: utils.get_censoring(x, by_group = True, model_type = 'KM'))

# Run selected model on input data
# (Models are saved in the utils file)

if args.experiment_name == 'original_pce':
    risks = utils.run_pce_model(cohort)
elif args.experiment_name == 'revised_pce':
    risks = utils.run_revised_pce_model(cohort)

# post-process file

output_df_eval = (cohort
                  .rename(columns={'fold_id': 'phase',
                                   'grp': 'group'})
                  .join(risks)
                  .filter(['phase', 'pred_probs', 'labels', 'weights',
                           'group', 'model_type', 'person_id'])
            )

# Save results

os.makedirs(args.result_path, exist_ok=True)
os.makedirs(os.path.join(args.result_path, 'performance', 'all'), exist_ok=True)
output_df_eval.to_csv(
    os.path.join(args.result_path, 'performance', 'all', 'predictions.csv'),
    index=False
)