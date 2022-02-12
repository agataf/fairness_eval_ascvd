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

study_path = 'FHS/Datasets/CSV/'
study_path = os.path.join(args.input_dir, study_path)

files_dict = {'vr_wkthru_ex09_1_1001d.csv' : {'dirname': study_path, 
                                             'dict': ['age1', 'bmi1', 'calc_ldl1', 'currsmk1', 'dbp1',
                                                      'sbp1', 'hdl1', 'tc1',  'trig1', 'liprx1',
                                                      'dmrx1', 'hrx1', 'sex', 'idtype']},
              'vr_diab_ex09_1_1002d.csv'   : {'dirname': study_path,
                                             'dict': ['hx_diab1']},
              'e_exam_ex08_1_0005d.csv'    : {'dirname': study_path, 
                                             'dict': ['h702', 'h7025']},
              'e_exam_ex03_7_0426d.csv'    : {'dirname': study_path,
                                             'dict': ['h700', 'h701', 'h702', 'h703', 'h704', 'h705', 'h706', 'h7025']},
              'vr_survcvd_2017_a_1194d.csv': {'dirname': study_path, 
                                             'dict': []},
              'vr_svstk_2017_a_1196d.csv'  : {'dirname': study_path, 
                                              'dict': []},
              'vr_afcum_2018_a_1169d.csv'  : {'dirname': study_path, 
                                             'dict': ['rhythm', 'ecgdate']},
              'vr_survdth_2017_a_1192d.csv': {'dirname': study_path, 
                                             'dict': []},
              'vr_soe_2018_a_1311d.csv'    : {'dirname': study_path, 
                                             'dict': []},
              'race_1.csv': {'dirname': study_path, 
                                             'dict': []}
             }


def read_file(filename, files_dict, index='subjid'):
    df = pd.read_csv(os.path.join(files_dict[filename]['dirname'], filename), engine='python')
    df = (df
          .rename(columns = lambda x: x.lower())
          .set_index(index)
          .sort_index()
         )
    if len(files_dict[filename]['dict']) > 0:
        df = df.filter(files_dict[filename]['dict'])

    return df

def get_gender(df):
    gender = df.sex.map({1: 'M', 2: 'F'})
    return gender

def get_lastfu(df):
    return df.filter(['lastcon', 'lastsoe']).max(axis=1)/365.25
    
def get_prevchf(df):
    prevchf = np.where((df.event>40) & (df.date<=0), 1, 0)
    return prevchf

def get_prevmi(df):
    prevmi = np.where(((df.event>= 1) & (df.event <= 5))
                      & (df.date<=0), 1, 0)
    return prevmi

def get_mi(df):
    mi = np.where((df.event>= 1) & (df.event <= 5), 1, 0)
    return mi

def get_chddeath(df):
    chddeath = np.where((df.event>= 21) & (df.event <= 24), 1, 0)
    return chddeath

def get_strkdeath(df):
    strkdeath = np.where((df.event==25), 1, 0)
    return strkdeath

def get_timetomi(df):
    timetomi = np.where((df.mi==1), df.date/365.25, np.nan)
    return timetomi

def get_timetochddeath(df):
    timetochddeath = np.where((df.chddeath==1), df.date/365.25, np.nan)
    return timetochddeath

def get_timetostrkdeath(df):
    timetostrkdeath = np.where((df.strkdeath==1), df.date/365.25, np.nan)
    return timetostrkdeath

def get_prevstrk(df):
    return np.where(df.strokedate < 0, 1, 0)

def get_strk(df):
    return np.where(df.stroke == 1, 1, 0)

def get_timetostrk(df):
    return np.where(df.stroke == 1, df.strokedate/365.25, np.nan)

def get_prevafib(df):
    return np.where((df.rhythm==1) & (df.ecgdate<0), 1, 0)
             
def get_racegrp(df):
    racegrp = np.where((df.h702 == 1) | (df.race == 'W'), 'W', np.nan)
    racegrp = np.where((df.h703 == 1) | (df.race == 'B'), 'B', racegrp)
    # collapsing Asian, Native Hawaiian and Pacific Islander into a single category
    racegrp = np.where((df.h704 == 1) | (df.h705 == 1) | (df.race == 'A'), 'A', racegrp)
    # collapsing non-white, American Indian or Alaska Native (h706) and Other into a single category
    racegrp = np.where((df.h706 == 1) | (df.race == 'O') | (df.h7025 == 1), 'O', racegrp)
    return racegrp

def get_ethnicity(df):
    ethnicity = np.where((df.h700 == 1) | (df.ethnicity == 'Hisp'), 'H', np.nan)
    ethnicity = np.where((df.h701 == 1) | (df.ethnicity == 'NHisp'), 'NH', ethnicity)
    return ethnicity

exams = read_file('vr_wkthru_ex09_1_1001d.csv', files_dict, index='pid')
diabs = read_file('vr_diab_ex09_1_1002d.csv', files_dict, index='pid')

exam = (exams
        .merge(diabs, how='left', on='pid')
        .rename(columns={'age1': 'age', 
                             'bmi1': 'bmi',
                             'calc_ldl1': 'ldlc',
                             'currsmk1': 'cursmoke',
                             'dbp1': 'diabp',
                             'sbp1': 'sysbp',
                             'hdl1': 'hdlc',
                             'tc1': 'totchol',
                             'trig1': 'trigly',
                             'liprx1': 'cholmed',
                             'dmrx1': 'diabmed',
                             'hrx1': 'hyptmdsr', 
                             'hx_diab1': 'diabt126'
                    })
        .assign(gender = lambda x: get_gender(x))
        .filter(['gender', 'age', 'bmi', 'ldlc', 'cursmoke', 
              'diabp', 'sysbp', 'hdlc',  'totchol', 'diabt126',
              'trigly',  'cholmed', 'diabmed', 'hyptmdsr', 'idtype'])
    )

race_ex1 = read_file('race_1.csv', files_dict, index='pid')
race_ex3 = read_file('e_exam_ex03_7_0426d.csv', files_dict, index='pid')
race_ex8 = read_file('e_exam_ex08_1_0005d.csv', files_dict, index='pid')

races = (race_ex1
 .merge(race_ex3, how='outer', on='pid')
 .merge(race_ex8, how='outer', on='pid')
 .assign(h702 = lambda x: x.filter(
                                    ['h702_x', 'h702_y']
                                                ).max(axis=1))
 .drop(columns=['h702_x', 'h702_y'])
 .assign(racegrp = lambda x: get_racegrp(x),
        ethnicity = lambda x: get_ethnicity(x))
 .assign(racegrp = lambda x: x.racegrp.replace('nan', 'W'))
 .filter(['racegrp', 'ethnicity'])
)

events = read_file('vr_survcvd_2017_a_1194d.csv', files_dict, index='pid')
stroke = read_file('vr_svstk_2017_a_1196d.csv', files_dict, index='pid')
death = read_file('vr_survdth_2017_a_1192d.csv', files_dict, index='pid')

surv_events = (events
          .merge(stroke.filter(['stroke', 'strokedate']), how='left', on='pid')
          .merge(death, how='left', on='pid')
             )

afib = (read_file('vr_afcum_2018_a_1169d.csv', files_dict, index='pid')
        .assign(prevafib = lambda x: get_prevafib(x))
        .filter(['prevafib'])
        .groupby('pid')
        .agg('max')
       )

surv_events = (surv_events
 .assign(lastexam = lambda x: x.lastatt/365.25,
             timetodth = lambda x: x.datedth/365.25,
             lastfu = lambda x: get_lastfu(x),
             strk = lambda x: get_strk(x),
             timetostrk = lambda x: get_timetostrk(x),
             prevstrk = lambda x: get_prevstrk(x)
             )
 .filter(['lastexam', 'timetodth', 'lastfu', 'strk', 'timetostrk', 'prevstrk', 'chddeath'])
)

soe_events = read_file('vr_soe_2018_a_1311d.csv', files_dict, index='pid')
soe_events = (soe_events
 .assign(prevchf = lambda x: get_prevchf(x),
             prevmi = lambda x: get_prevmi(x),
             mi = lambda x: get_mi(x),
             chddeath = lambda x: get_chddeath(x),
             strkdeath = lambda x: get_strkdeath(x),
             timetomi = lambda x: get_timetomi(x),
             timetochddeath = lambda x: get_timetochddeath(x),
             timetostrkdeath = lambda x: get_timetostrkdeath(x))
 )

maxes = (soe_events
 .filter(items=['prevchf', 'prevmi', 'mi', 'chddeath', 'strkdeath'])
 .groupby('pid')
 .agg('max')
)

mins = (soe_events
 .filter(items=['timetomi', 'timetochddeath', 'timetostrkdeath'])
 .groupby('pid')
 .agg('min')
)

soe_events = (maxes
              .merge(mins, on='pid'))

# it seems we only have events data for about half of all the people

df_final = (soe_events
            .merge(surv_events, how='outer', on='pid')
            .assign(chddeath = lambda x: x.filter(
                                    ['chddeath_x', 'chddeath_y']
                                                ).max(axis=1))
            .drop(columns=['chddeath_x', 'chddeath_y'])
            .merge(afib, how='outer', on='pid')
            .merge(races, how='outer', on='pid')
            .merge(exam, how='outer', on='pid')
            .query('idtype == 1')
            .assign(study = 'FHS_OS')
            .reset_index()
            .rename(columns={'pid': 'cohort_pid'})
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

df_final.to_csv(os.path.join(dirname_output, 'fhs_os.csv'), index = False)