import numpy as np
import pandas as pd
import os
import git
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False)
parser.add_argument("--output_path", type=str, help="path where aggregated data should be stored", required=False)
parser.add_argument("--test_frac", type=float, help="fraction of data that should go into test", required=False,
                   default=0.1)
parser.add_argument("--eval_frac", type=float, help="fraction of data that should go into evaluation", required=False,
                   default=0.1)

args = parser.parse_args()

def patient_split(df, test_frac=0.1, eval_frac=0.1, nfold=10, seed=657):

    assert (test_frac > 0.0) & (test_frac < 1.0)

    # Shuffle the patients
    patient_df = df.sample(frac=1, random_state=seed)

    # Record the number of samples in each split
    num_test = int(np.floor(test_frac * patient_df.shape[0]))
    num_eval = int(np.floor(eval_frac * patient_df.shape[0]))
    num_train = patient_df.shape[0] - (num_test + num_eval)

    # Get the number of patients in each fold
    test_patient_df = patient_df.iloc[0:num_test].assign(fold_id="test")
    eval_patient_df = patient_df.iloc[num_test:(num_test+num_eval)].assign(fold_id="eval")
    train_patient_df = patient_df.iloc[(num_test + num_eval):]

    train_patient_df = train_patient_df.assign(
        fold_id=lambda x: np.tile(
            np.arange(1, nfold + 1), int(np.ceil(num_train / nfold))
        )[: x.shape[0]]
    )
    train_patient_df["fold_id"] = train_patient_df["fold_id"].astype(str)
    patient_df = pd.concat([train_patient_df, test_patient_df, eval_patient_df], ignore_index=True)

    df = df.merge(patient_df)
    return df

df = (pd
      .read_csv(os.path.join(args.cohort_path, 'all.csv'))
      .assign(event_time      = lambda x: x.event_time_10yr,
              event_indicator = lambda x: x.ascvd_10yr)
     )

# Stratified splitting

result = {}
for (grp_name, grp_df) in df.groupby(['censored_10yr', 'ascvd_10yr', 'race_black', 'gender_male', 'study']):
    result[grp_name] = patient_split(grp_df, test_frac=args.test_frac, eval_frac=args.eval_frac)

data_df = (pd
           .concat(result, ignore_index=True)
           .reset_index(drop=True)
           .reset_index(drop=False)
           .rename(columns = {'index': 'person_id'})
          )

data_df.to_csv(os.path.join(args.output_path), index = False)
