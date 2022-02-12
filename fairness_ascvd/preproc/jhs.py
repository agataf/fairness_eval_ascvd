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
study_path = 'jackson_heart_study'
study_path = os.path.join(args.input_dir, study_path)

files_dict = {'analysis.csv': {'dirname': os.path.join(study_path,'JHS_2020_08_03'), 
                                'dict': ['subjid', 'VisitDate', 'ARIC', 'age', 'male',
                                         'BMI', 'currentSmoker', 'sbp', 'dbp', 'FPG',
                                         'Diabetes', 'BPmeds', 'HTN', 'ldl', 'hdl', 
                                         'totchol', 'trigs', 'eGFRckdepi', 'statinMeds',
                                         'strokeHx', 'MIHx', 'CardiacProcHx', 'CarotidAngioHx',
                                         'CHDHx', 'CVDHx', 'Afib', 'CAC', 'visit']},
             'v2_1076_events.sas7bdat': {'dirname': os.path.join(study_path, 'JHS_2020_08_03'),
                                         'dict': []},
             'afulong_death.xlsx': {'dirname': os.path.join(study_path, 'JHS_2021_02_21'), 
                                    'dict': []}}
def read_file(filename, files_dict, index='subjid'):
    file_extension = filename.split('.')[-1]
    path = os.path.join(files_dict[filename]['dirname'], filename)
    if file_extension == 'csv':
        df = pd.read_csv(path, engine='python')
    elif file_extension == 'dta':
        df = pd.read_stata(path)
    elif file_extension == 'sas7bdat':
        df = pd.read_sas(path)
        df[index] = df[index].str.decode(encoding = 'utf-8')
    elif file_extension == 'xlsx':
        df = pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError('filename must be of type csv, dta or sas7bdat')
    
    if len(files_dict[filename]['dict']) > 0:
        df = df.filter(files_dict[filename]['dict'])
        
    df = df.assign(idno = lambda x: x[index].str.slice(1,7).astype('str'))
    df = df.set_index('idno')
    df.sort_index(inplace=False)
    
    return df

def days_to_from(col_to, col_from):
    return (col_to - col_from).dt.days/365.25

def get_examdate(df):
    return pd.to_datetime(df.VisitDate)

def get_laststudy(df):
    dates = (df
             .sort_values(by=['examdate'])
             .filter(items=['examdate', 'visit'])
             .pivot(columns='visit', values='examdate')
             .rename(columns = lambda x: ''.join(('examdate', str(x))))
            )
    dates = dates.assign(maxdate = lambda x: x.max(axis=1),
                         laststudy = lambda x: (x.maxdate-x.examdate1).dt.days/365.25)

    return dates.laststudy

def get_cursmoke(df):
    # in the original setup, missing values were removed, 
    # here I assume no info about current smoking means no smoking
    return df.currentSmoker.fillna(0) 
    

exam = read_file('analysis.csv', files_dict)

exam = exam.assign(examdate = lambda x: get_examdate(x),
                   laststudy = lambda x: get_laststudy(x),
                   cursmoke = lambda x: get_cursmoke(x),
                   prevchf = 0,
                   racegrp = 'B', 
                   gender = lambda x: x.male.map({1: 'M', 0: 'F'}))

exam = exam.rename(columns={'FPG'          : 'glucose',
                            'BMI'          : 'bmi',
                            'hdl'          : 'hdlc',
                            'ldl'          : 'ldlc',
                            'trigs'        : 'trigly',
                            'MIHx'         : 'prevmi',
                            'strokeHx'     : 'prevstrk',
                            'CardiacProcHx': 'prevproc',
                            'CHDHx'        : 'prevchd',
                            'CVDHx'        : 'prevcvd',
                            'Afib'         : 'prevafib',
                            'Diabetes'     : 'diabt126',
                            'sbp'          : 'sysbp',
                            'dbp'          : 'diabp',
                            'statinMeds'   : 'cholmed',
                            'BPmeds'       : 'hyptmdsr'})


exam = (exam
        # filtering just visit 1
        .query('(ARIC != 1) & (visit == 1)')
        .filter(items=['examdate', 'laststudy', 'cursmoke', 'prevchf', 'hdlc',
                       'ldlc', 'totchol', 'trigly', 'prevmi', 'prevstrk', 'prevproc',
                       'prevchd', 'prevcvd', 'prevafib', 'diabt126', 'sysbp',
                       'diabp', 'cholmed', 'hyptmdsr', 'racegrp', 'gender', 'age', 'bmi'])
       )

events = read_file('v2_1076_events.sas7bdat', files_dict)

# no information on stroke death
df = (events
      .rename(columns = {'DATEMI': 'midate',
                             'ED17DP': 'strkdate',
                             'ENDDATE': 'chddeathdate',
                             'IN17DP'  : 'strk',
                             'MI17'    : 'mi',
                             'FATCHD17': 'chddeath'})
      .merge(exam, how='outer', on='idno')
      .assign(timetomi = lambda x: days_to_from(x.midate, x.examdate),
              timetostrk = lambda x: days_to_from(x.strkdate, x.examdate),
              timetochddeath = lambda x: days_to_from(x.chddeathdate, x.examdate))

         )

deaths = read_file('afulong_death.xlsx', files_dict, index='subjid')

deaths = (deaths
            .assign(date            = lambda x: pd.to_datetime(x.date),
                    lastcontactdate = lambda x: pd.to_datetime(x.lastcontactdate))
            .filter(items=['lastcontactdate', 'death'])
            .groupby('idno').agg('max')
           )

df_final =  (deaths
             .merge(df, how='outer', on='idno')
             .assign(lastfu = lambda x: days_to_from(x.lastcontactdate, x.examdate))
             .assign(study = 'JHS')
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


df_final.to_csv(os.path.join(dirname_output, 'jhs.csv'), index = False)