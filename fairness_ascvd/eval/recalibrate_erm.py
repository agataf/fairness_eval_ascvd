import pandas as pd
import os
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--new_model_type", type=str, required=True)
parser.add_argument('--input_model_path', type=str, help='path where model we want to recalibrate is stored')
parser.add_argument('--result_path', type=str, help='path where inference results will be stored')
parser.add_argument("--model_type", type=str, required=False, default='logistic')
parser.add_argument("--transform", type=str, required=False, default=None)
args = parser.parse_args()

aggregate_path = os.path.join(args.input_model_path, 'performance', 'all')
new_aggregate_path = os.path.join(args.result_path, 'performance', 'all')
os.makedirs(new_aggregate_path, exist_ok = True)

preds = pd.read_csv(os.path.join(aggregate_path, 'predictions.csv'))

# For each group, for each of the 10 models (each corresponding to a training fold)
# we use predictions generated the model on the recalibration (eval) set, and train 
# a recalibration model, generating predictions on the test set
 
test_calibs = []
for group in [1,2,3,4]:
    for fold_id in range(1,11):    
        group_df = preds.query("(group==@group) & (fold_id==@fold_id)")
        group_test = group_df.query("phase=='test'").reset_index(drop=True)
        group_recalib = group_df.query("phase=='eval'").reset_index(drop=True)

        # train the appropriate recalibration model on the group_recalib set
        calib_model = utils.get_calib_model(to_calibrate = group_recalib, 
                                            transform    = args.transform, 
                                            model_type   = args.model_type)
        
        # generate recalibrated value on the test set
        calibrated_values = utils.get_calib_probs(model     = calib_model, 
                                                  x         = group_test.pred_probs.values, 
                                                  transform = args.transform)
        test_calib = (calibrated_values
                      .merge(group_test)
                      .drop(['pred_probs', 'model_input'], axis=1)
                      .rename(columns={'calibration_density': 'pred_probs'})
                     )
        test_calibs.append(test_calib)
        
test_calibs = (pd
               .concat(test_calibs) 
               .assign(model_type = args.new_model_type,
                       model_id = args.transform)
              )

test_calibs.to_csv(os.path.join(new_aggregate_path, 'predictions.csv'), index=False)