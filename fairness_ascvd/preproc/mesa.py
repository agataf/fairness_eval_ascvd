import pandas as pd 
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()

dirname_output = args.output_dir
os.makedirs(dirname_output, exist_ok=True)

exam1_path = os.path.join(args.input_dir, 'MESA/exam1/datasets')
exam1_file = 'MESAe1FinalLabel01022018.dta'

events_path = os.path.join(args.input_dir, 'MESA/events/datasets')
events_file = 'MESAEvThru2017_20191004.dta'


def read_file(filename, filepath, index='idno'):
    df = (pd
          .read_stata(os.path.join(filepath, filename), convert_categoricals=False)
          .assign(idno = lambda x: x.idno.astype(int))
          .set_index(index, drop=True)
         )
    
    return df

def cond_yrsto(time, cond):
    return [left/365.25 if right == 1 else np.nan for (left,right) in zip(time, cond)]

df = read_file(exam1_file, exam1_path)

df = (df
      .filter(items = ['idno', 'age1c', 'race1c', 'gender1', 'bmi1c',
                        'cig1c', 'sbp1c', 'dbp1c', 'glucos1c', 'dm031c',
                        'htnmed1c', 'htn1c', 'ldl1', 'hdl1', 'chol1', 'trig1',
                        'creatin1c', 'lipid1c', 'ascvd1c', 'bpmed1', 'sttn1c',
                        'agatum1c', 'agatpm1c'])
      .rename(columns = {'glucos1c': 'glucose',
                         'hdl1': 'hdlc',
                         'ldl1': 'ldlc',
                         'chol1': 'totchol',
                         'age1c': 'age',
                         'sbp1c': 'sysbp',
                         'dbp1c': 'diabp',
                         'bmi1c': 'bmi',
                         'lipid1c': 'cholmed',
                         'bpmed1': 'hyptmdsr',
                         'agatum1c': 'cac',
                         'agatpm1c': 'cac_phantomadjusted',
                         'trig1': 'trigly',
                         'afib1c': 'prevafib'})
      .assign(gender = df.gender1.map({1: 'M', 0: 'F'}),
              racegrp = df.race1c.map({3: 'B', 1: 'W'}),
              cursmoke = df.cig1c.map({2: 1, 1: 0, 0:0}),
              diabt126 = df.dm031c.map({0:0, 1:0, 2: 1, 3: 1}))
      .filter(items = ['glucose', 'hdlc', 'ldlc', 'totchol', 'age',
                        'sysbp', 'diabp', 'bmi', 'cholmed', 'hyptmdsr',
                        'cac', 'cac_phantomadjusted', 'gender', 'racegrp',
                        'cursmoke', 'diabt126', 'trigly'])
      .assign(prevmi   = 0, # from MESA exclusion criteria
              prevang  = 0,
              prevstrk = 0,
              prevchf  = 0,
              prevafib = 0,
              prevproc = 0)
     )

events = read_file(events_file, events_path)

# TODO: check logic
events = (events
          .assign(chddeath = lambda x: (x.dth == 1) & (x.dthtype == 1),
                       strkdeath = lambda x: (x.dth == 1) & (x.dthtype == 2),
                       timetoang = lambda x: cond_yrsto(x.angtt, x.ang),
                       timetomi = lambda x: cond_yrsto(x.mitt, x.mi),
                       timetostrk = lambda x: cond_yrsto(x.strktt, x.strk),
                       timetochddeath = lambda x: cond_yrsto(x.dthtt, x.chddeath),
                       timetostrkdeath = lambda x: cond_yrsto(x.dthtt, x.strkdeath),
                       timetodth = lambda x: x.dthtt/365.25,
                       lastfu = lambda x: x.fuptt/365.25)
          .rename(columns = {'dth': 'death'})
          .filter(items = ['chddeath', 'strkdeath', 'timetoang', 'timetomi', 
                       'timetostrk', 'timetochddeath', 'timetostrkdeath',
                       'timetodth', 'lastfu', 'strk', 'death', 'mi'])
         )


df_final = (df
            .merge(events, how='outer', on='idno')
            .assign(study = 'MESA')
            .reset_index()
            .rename(columns={'idno': 'cohort_pid'})
            .filter(['timetomi', 'timetostrk', 'timetochddeath', 'timetostrkdeath', 'timetodth',
                             'lastexam', 'lastfu', 
                             'mi', 'strk', 'chddeath', 'strkdeath', 'death',
                             'prevmi', 'prevstrk', 'prevproc', 'prevchf', 'prevcvd',
                             'prevap', 'prevang', 'prevchd', 'prevafib',
                             'cohort_pid', 'racegrp', 'gender', 'age', 'study',
                             'cursmoke', 'hyptmdsr', 'cholmed', 'diabt126',  'totchol',
                             'ldlc', 'trigly', 'hdlc', 'sysbp' , 'diabp', 'bmi'])
           )

df_final.to_csv(os.path.join(dirname_output, 'mesa.csv'), index = False)