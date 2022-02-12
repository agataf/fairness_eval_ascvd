import pandas as pd 
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
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
     
def exclude(df):    
    #excl_race = df.racegrp.isna()
    excl_race = df.grp.isna()
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
    extr_all = (extr_sysbp | extr_totchol | extr_hdlc)
    
    
    if args.print_info:
        print(sum(excl_race), '\t removed: missing race variable')
        print(sum(excl_age), '\t removed: age')
        print(sum(excl_prevcond), '\t removed: previous conditions')
        print(sum(excl_statin), '\t removed: on statin')
        print(sum(extr_all), '\t removed: extreme values')
        print(sum(excl_missing), '\t removed: missing variables')
        
    print(df.shape)
    df = df[~(extr_all | excl_all)]
    print(df.shape)
    return df

def calc_ldlc(df):
    new_ldlc = df.totchol - (df.hdlc + df.trigly/5)
    return new_ldlc

cohort_frames = []
for el in ['mesa', 'fhs_os', 'chs', 'aric', 'jhs', 'cardia']:
    cohort_frames.append(pd.read_csv(os.path.join(args.input_dir, '.'.join((el, 'csv')))))

df = pd.concat(cohort_frames)


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
                   grp         = lambda x: (1-x.race_black)*1 + (x.gender_male)*2 + 1,
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
          .dropna()
         )

if args.print_info:
    print(cohort_excl.shape[0]-cohort.shape[0], '\t removed: missing variables')
    print(n_ldl_missing, '\t LDL-c values imputed')
    print('----------')
    print(cohort.shape[0], '\t subjects remain')
    print(cohort.censored_10yr.sum(), '\t censored')
    print(cohort.ascvd_10yr.sum(), '\t events')
    print(cohort.head())
    print(cohort.filter(['grp', 'censored_10yr', 'ascvd_10yr']).groupby(['grp']).agg({'ascvd_10yr': ['sum', 'mean'],
                                                                    'censored_10yr': ['sum', 'mean']}))
    print(cohort.filter(['censored_10yr', 'ascvd_10yr']).agg({'ascvd_10yr': ['sum', 'mean'],
                                                                    'censored_10yr': ['sum', 'mean']}))

    missing_cols = set(model_vars)-set(cohort.columns)
    if len(missing_cols) > 0:
        print('missing', len(missing_cols), 'columns:', missing_cols)
        
    print('--------------------\n\n')

cohort.dropna().to_csv(os.path.join(args.output_dir, 'all.csv'), index = True)