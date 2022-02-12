import pandas as pd
import numpy as np

def patient_split(df, test_frac=0.1, nfold=10, seed=657):

    assert (test_frac > 0.0) & (test_frac < 1.0)


    # Shuffle the patients
    patient_df = df.sample(frac=1, random_state=seed)

    # Record the number of samples in each split
    num_test = int(np.floor(test_frac * patient_df.shape[0]))
    num_train = patient_df.shape[0] - num_test

    # Get the number of patients in each fold
    test_patient_df = patient_df.iloc[0:num_test].assign(fold_id="test")

    train_patient_df = patient_df.iloc[num_test:]

    train_patient_df = train_patient_df.assign(
        fold_id=lambda x: np.tile(
            np.arange(1, nfold + 1), int(np.ceil(num_train / nfold))
        )[: x.shape[0]]
    )
    train_patient_df["fold_id"] = train_patient_df["fold_id"].astype(str)
    patient_df = pd.concat([train_patient_df, test_patient_df], ignore_index=True)

    df = df.merge(patient_df)
    return df