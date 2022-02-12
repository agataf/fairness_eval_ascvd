import pandas as pd 
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--print_info', type=bool)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

input_vars = ['timetomi', 'timetostrk', 'timetochddeath', 'timetostrkdeath', 'timetodth',
                             'lastexam', 'lastfu', 
                             'mi', 'strk', 'chddeath', 'strkdeath', 'death',
                             'prevmi', 'prevstrk', 'prevproc', 'prevchf', 'prevcvd',
                             'prevap', 'prevang', 'prevchd', 'prevafib',
                             'cohort_pid', 'racegrp', 'gender', 'age', 'study',
                             'cursmoke', 'hyptmdsr', 'cholmed', 'diabt126',  'totchol',
                             'ldlc', 'trigly', 'hdlc', 'sysbp' , 'diabp', 'bmi']

model_vars = ['cohort_pid', 'age', 'race_black', 'gender_male',
     'grp', 'hdlc', 'ldlc', 'trigly', 'totchol', 'sysbp',
     'cursmoke', 'diabt126', 'unrxsbp', 'rxsbp', 'study', 
     'ascvd_10yr', 'censored_10yr', 'event_time_10yr', 'bmi']

df = pd.read_csv(os.path.join(args.input_dir, '.'.join((args.study, 'csv'))))

def exclude(df):    
    excl_race = df.racegrp.isna()
    excl_age = ~df.age.between(40,79)
    excl_prevcond = ((df.prevcond== 1) | (df.max_time <= 0))
    excl_statin = (df.cholmed == 1)
    
    extr_sysbp = ~df.sysbp.between(90,200)
    extr_totchol = ~df.totchol.between(130,320)
    extr_hdlc = ~df.hdlc.between(20,100)
    
    excl_missing = df.filter(items = ['age', 'totchol', 'hdl', 'sysbp', 'rxbp', 
              'dm', 'cursmoke', 'race', 'gender',
              'event_time_10yr', 'censored_10yr', 'ascvd_10yr']).isnull().any(axis=1)
    
    excl_all = (excl_race | excl_age | excl_prevcond | excl_statin | excl_missing)
    extr_all = ((extr_sysbp | extr_totchol | extr_hdlc) & ~excl_all)
    
    
    if args.print_info:
        print(sum(excl_all), '\t removed: exclusion criteria')
        print(sum(extr_all), '\t removed: extreme values')
        
        
    df = df[~(extr_all | excl_all)]
    
    return df

def calc_ldlc(df):
    new_ldlc = df.totchol - (df.hdlc + df.trigly/5)
    return new_ldlc


df_final = (df.
            assign(timetoascvd = lambda x: (x
                                            .filter(items=['timetomi', 'timetostrk', 'timetochddeath',
                                                           'timetostrkdeath'])
                                            .replace(0.0, 1e-18)
                                            .min(axis=1)
                                           ),
                   ascvd       = lambda x: (x
                                            .filter(items=['mi', 'strk', 'chddeath', 'strkdeath'])
                                            .any(axis=1)
                                            .astype(int)
                                           ),
                   max_time    = lambda x: (x
                                            .filter(items=['lastexam', 'lastfu', 'timetodth', 'timetoascvd'])
                                            .max(axis=1)
                                           ),
                   event_time  = lambda x: np.minimum(x.timetoascvd.replace(np.nan, float('inf')), 
                                                      x.max_time.replace(np.nan, float('inf'))
                                                     ),
                   prevcond    = lambda x: x.filter(items=['prevmi', 'prevstrk', 'prevproc', 'prevchf',
                                                           'prevcvd','prevchd', 'prevafib']
                                                   ).any(axis=1).astype(int),
                   unrxsbp     = lambda x: x.sysbp*(1-x.hyptmdsr),
                   rxsbp       = lambda x: x.sysbp*(x.hyptmdsr),
                   race        = lambda x: x.racegrp.map({'B': 'black', 'W': 'white'}),
                   gender      = lambda x: x.gender.map({'M': 'male', 'F': 'female'}),
                   race_black  = lambda x: 1.0 * (x.race == 'black'), 
                   gender_male = lambda x: 1.0 * (x.gender == 'male'),
                   grp         = lambda x: (x.race != 'black')*1 + (x.gender == 'male')*2 + 1,
                   ascvd_10yr  = lambda x: (x.timetoascvd <= 10) & (x.ascvd == 1),
                   censored_10yr   = lambda x: (x.event_time <= 10) & (x.ascvd == 0),
                   event_time_10yr = lambda x: np.minimum(x.event_time, 10)
                  )
            .filter(items = ['cohort_pid', 'age', 'race_black', 'gender_male',
                             'grp', 'hdlc', 'ldlc', 'trigly', 'totchol', 
                             'cursmoke', 'diabt126', 'unrxsbp', 'rxsbp', 'study', 
                             'ascvd_10yr', 'censored_10yr', 'event_time_10yr',
                             'prevcond', 'max_time', 'cholmed', 'sysbp', 'bmi', 'racegrp'])
           )

if args.print_info:
    print(args.study, 'cohort extraction:\n')
    print(df.shape[0], '\t subjects in input file')

    missing_cols = set(input_vars) - set(df.columns)
    if len(missing_cols) > 0:        
        print('missing input columns:', len(missing_cols), 'columns:', missing_cols)
        
    print('----------')
    
n_ldl_missing = df_final.ldlc.isna().sum()
df_final = df_final.assign(ldlc = lambda x: np.where(x.ldlc.isna(), calc_ldlc(x), x.ldlc))

cohort_excl = exclude(df_final)

cohort = (cohort_excl
          .reset_index()
          .rename_axis('cohort_idx')
          .filter(items = model_vars)
          
)

if args.print_info:
    print(cohort_excl.shape[0]-cohort.dropna().shape[0], '\t removed: missing variables')
    print(n_ldl_missing, '\t LDL-c values imputed')
    print('----------')
    print(cohort_excl.shape[0], '\t subjects remain')
    
    missing_cols = set(model_vars)-set(cohort.columns)
    if len(missing_cols) > 0:
        print('missing', len(missing_cols), 'columns:', missing_cols)
        
    print('--------------------\n\n')

cohort.dropna().to_csv(os.path.join(args.output_dir, ''.join((args.study, '.csv'))), index = True)