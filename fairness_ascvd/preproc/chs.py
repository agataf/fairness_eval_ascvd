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

study_path = 'CHS'
study_path = os.path.join(args.input_dir, study_path)

files_dict = {'basebothfinal.csv': {'dirname': os.path.join(study_path, 'BASELINE'), 
                                'dict': ['diuret06', 'avzmdia', 'ccbir06', 'alpha06', 'amount',
                                         'afib', 'avzmsys', 'sttn06', 'mlpd06', 'smoke', 
                                         'htnmed06', 'miblmod', 'vaso06', 'ccbt06', 'hctzk06',
                                         'betad06', 'anblmod', 'alphad06', 'hctz06', 'trig44',
                                         'aced06', 'stblmod', 'ldladj', 'chblmod', 'race01',
                                         'corart', 'hdl44', 'cre44', 'bpssur', 'glu44', 'ccb06',
                                         'beta06', 'choladj', 'ecglvh', 'diabada', 'vasod06', 
                                         'age2', 'loop06', 'gend01', 'lipid06', 'ccbsr06',
                                         'ace06', 'idno', 'bp57', 'bmi', 'angbase']
                                   },
              'events.csv': {'dirname': os.path.join(study_path, 'EVENTS'),
                            'dict' : ['idno','evtype', 'increc', 'mi', 'str', 'fatal',
                                      'chddeath', 'defprob', 'ttoevent', 'censtime']}
             }

time_files = ['YEAR3/yr3final.csv', 
              'YEAR4/yr4final.csv', 
              'YEAR5/yr5oldfinal.csv',
              'YEAR6/yr6final.csv', 
              'YEAR7/yr7final.csv',
              'YEAR8/yr8final.csv',
              'YEAR9/yr9final.csv', 
              'YEAR10/yr10final.csv',
              'YEAR11/yr11final.csv', 
              'YEAR12/year12annualfinal.csv',
              'YEAR13/year13annualfinal.csv', 
              'YEAR14/year14annualfinal.csv',
              'YEAR15/year15annualfinal.csv', 
              'YEAR16/year16annualfinal.csv',
              'YEAR17/year17annualfinal.csv']
time_vars = ['idno', 'stdytime']

def read_file(filename, files_dict):
    df = pd.read_csv(os.path.join(files_dict[filename]['dirname'], filename), engine='python')
    df = (df
          .rename(columns = lambda x: x.lower())
          .filter(files_dict[filename]['dict'])
          .set_index('idno')
          .sort_index()
         )
    return df

def get_lastexam(time_files, time_vars):
    times = None
    for i, filename in enumerate(time_files):
        df = pd.read_csv(os.path.join(study_path, filename), engine='python')
        df.columns = df.columns.str.lower()
        df = (df
              .filter(time_vars)
              .set_index('idno')
              .assign(stdytime=lambda x: x.stdytime/365.25)
              .rename(columns={'stdytime': 'stdytime' + str(i+3)})
             )
        if times is not None:
            times = times.join(df)
        else:
            times = df

    times = times.assign(lastexam = lambda x: x.max(axis=1))
    return times.lastexam

def get_gender(df):
    gender = df.gend01.map({0: 'F', 1: 'M'})
    return gender

def get_racegrp(df):
    racegrp = df.race01.map({1: 'W', 2: 'B', 3: 'A', 4: '0', 5: '0'})
    return racegrp

def get_hyptmdml(df):
    hyptmdml = np.where((df.htnmed06 == 1) | (df.beta06 == 1) | (df.betad06 == 1) |
                    (df.ccb06 == 1) | (df.ace06 == 1) | (df.aced06 == 1) |
                    (df.vaso06 == 1) | (df.vasod06 == 1) | (df.diuret06 == 1) |
                    (df.loop06 == 1) | (df.hctz06 == 1) | (df.hctzk06 == 1) |
                    (df.ccbir06 == 1) | (df.ccbsr06 == 1) | (df.alpha06 == 1) |
                    (df.alphad06 == 1) | (df.ccbt06 == 1), 1, np.nan)

    hyptmdml = np.where((df.htnmed06 == 0) & ((df.beta06 == 0) | (df.betad06 == 0) |
                        (df.ccb06 == 0) | (df.ace06 == 0) | (df.aced06 == 0) |
                        (df.vaso06 == 0) | (df.vasod06 == 0) | (df.diuret06 == 0) |
                        (df.loop06 == 0) | (df.hctz06 == 0) | (df.hctzk06 == 0) |
                        (df.ccbir06 == 0) | (df.ccbsr06 == 0) | (df.alpha06 == 0) |
                        (df.alphad06 == 0) | (df.ccbt06 == 0)), 0, hyptmdml)
    return hyptmdml

def get_cholmed(df):
    cholmed1 = np.where((df.sttn06 == 1) | (df.lipid06 == 1) | (df.mlpd06 == 1), 1, np.nan)
    cholmed1 = np.where((df.sttn06 == 0) & (df.lipid06 == 0) & (df.mlpd06 == 0), 0, cholmed1) 
    return cholmed1

def get_age(df):
    return (df.age2*2)+63.5

def get_prevchd_org(df):
    prevchd = np.where((df.miblmod == 1) | (df.bpssur == 1) | (df.corart == 1), 1, 0)
    return prevchd

def get_prevproc_org(df):
    prevproc = np.where((df.bpssur == 1) | (df.corart == 1), 1, np.nan)
    prevproc = np.where((df.bpssur == 0) & (df.corart == 0), 0, prevproc)
    return prevproc

def get_diabt126(df):
    return df.diabada.map({1: 0, 2: 0, 3: 1, 4: 1})

def get_cursmoke(df):
    return df.smoke.map({1: 0, 2: 0, 3: 1})

def get_hyptmdsr(df):
    hyptmdsr = np.where((df.bp57 == 1) & (df.hyptmdml == 1), 1, np.nan)
    hyptmdsr = np.where((df.bp57 == 0) | (df.hyptmdml == 0), 0, hyptmdsr)
    return hyptmdsr

def get_mi(df):
    mi = np.where(((df.evtype == 1) | (df.evtype== 10)) & (df.increc== 1), 1, 0)
    return mi

def get_strk(df):
    strk = np.where((df.evtype == 3) & (df.increc== 1), 1, 0)
    return strk

def get_chddeath(df):
    chddeath = np.where(((df.evtype == 11) | (df.evtype == 4) | (df.evtype == 7) | (df.evtype == 8)) 
                        & (df.fatal == 1) & (df.increc != 0), 1, 0)
    return chddeath

def get_strkdeath(df):
    strkdeath = np.where((df.evtype == 11) & (df.fatal == 1) & (df.defprob != 0), 1, 0)
    return strkdeath

def get_dead(df):
    dead = np.where((df.evtype == 9) | (df.chddeath == 1) | (df.strkdeath == 0), 1, 0)
    return dead

def get_yrsto(timeto, event):
    yrsto = np.where(event==1, timeto/365.25, np.nan)
    return yrsto

exam = read_file('basebothfinal.csv', files_dict)


exam = exam.assign(cohort = 'ORG',
               gender = lambda x: get_gender(x),
               racegrp = lambda x: get_racegrp(x),
               cursmoke = lambda x: get_cursmoke(x),
               age = lambda x: get_age(x),
               diabt126 = lambda x: get_diabt126(x),
               cholmed = lambda x: get_cholmed(x),
               hyptmdml = lambda x: get_hyptmdml(x),
               prevchd = lambda x: get_prevchd_org(x),
               prevproc = lambda x: get_prevproc_org(x),
               exam = 1
              )

# TODO - don't have hyptmdsr, only hyptmdml
# hyptmdsr = np.where((df.bp57 == 1) & (hyptmdml == 1), 1, np.nan)
# hyptmdsr = np.where((df.bp57 == 0) | (hyptmdml == 0), 0, hyptmdsr)
exam = exam.rename(columns={'amount': 'avgsmoke',
                           'choladj': 'totchol',
                           'trig44': 'trigly',
                           'hdl44': 'hdlc',
                           'cre44': 'sercreat',
                           'ldladj': 'ldlc',
                           'glu44': 'glucose',
                           'ecglvh': 'lvh',
                           'avzmsys': 'sysbp',
                           'avzmdia': 'diabp', 
                           'miblmod': 'prevmi',
                           'chblmod': 'prevchf',
                           'stblmod': 'prevstrk',
                           'anblmod': 'prevap',
                           'afib': 'prevafib',
                           'htnmed06': 'hyptmdsr',
                           'angbase': 'prevang'})

exam = exam.filter(['prevafib', 'sysbp', 'diabp', 'smoke', 'prevmi', 'prevap', 'trigly', 'prevstrk',
            'ldlc', 'prevchf', 'prevang', 'hdlc', 'glucose', 'bmi', 'totchol', 'diabetes', 'hyptmdsr',
            'cohort', 'gender', 'racegrp', 'cursmoke', 'age', 'diabt126', 
            'cholmed', 'hyptmdml', 'prevchd', 'prevproc', 'exam'])

events = read_file('events.csv', files_dict)
events = events.assign(mi = lambda x: get_mi(x),
                       strk = lambda x: get_strk(x),
                       chddeath = lambda x: get_chddeath(x),
                       strkdeath = lambda x: get_strkdeath(x),
                       death = lambda x: get_dead(x),
                       timetomi = lambda x: get_yrsto(x.ttoevent, x.mi),
                       timetostrk = lambda x: get_yrsto(x.ttoevent, x.strk),
                       timetochddeath = lambda x: get_yrsto(x.ttoevent, x.chddeath),
                       timetostrkdeath = lambda x: get_yrsto(x.ttoevent, x.strkdeath),
                      )

events = events.rename(columns={'censtime': 'max_time'})

maxes = (events
         .filter(items = ['mi', 'strk', 'chddeath', 'strkdeath', 'death'])
         .groupby('idno')
         .agg('max'))

mins = (events
        .filter(items = ['timetomi', 'timetostrk', 'timetochddeath', 'timetostrkdeath'])
        .groupby('idno')
        .agg('min'))

events = maxes.merge(mins, on='idno')

lastexam = get_lastexam(time_files, time_vars)

df_final = (exam
            .merge(events, how='outer', on='idno')
            .merge(lastexam, how='outer', on='idno')
            .assign(study = 'CHS')
            .reset_index()
            .rename(columns={'idno': 'cohort_pid'})
            .filter(items = ['timetomi', 'timetostrk', 'timetochddeath', 'timetostrkdeath', 'timetodth',
                             'lastexam', 'lastfu', 
                             'mi', 'strk', 'chddeath', 'strkdeath', 'death',
                             'prevmi', 'prevstrk', 'prevproc', 'prevchf', 'prevcvd',
                             'prevap', 'prevang', 'prevchd', 'prevafib',
                             'cohort_pid', 'racegrp', 'gender', 'age', 'study',
                             'cursmoke', 'hyptmdsr', 'cholmed', 'diabt126',  'totchol',
                             'ldlc', 'trigly', 'hdlc', 'sysbp' , 'diabp', 'bmi']
                   )
           )

df_final.to_csv(os.path.join(dirname_output, 'chs.csv'), index = False)
