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

study_path = 'ARIC/Main_Study'
study_path = os.path.join(args.input_dir, study_path)

dirname_v1 = os.path.join(study_path, 'v1/csv')
dirname_v5 = os.path.join(study_path, 'v5/csv')
dirname_stroke = os.path.join(study_path, 'cohort_Stroke/csv')
dirname_chd = os.path.join(study_path, 'cohort_CHD/csv')
dirname_incident = os.path.join(study_path, 'cohort_Incident/csv')

files_dict = {'derive13.csv': {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'CENTERID', 'CHOLMDCODE01', 'CHOLMDCODE02',
                                  'CURSMK01', 'DIABTS02', 'DIABTS03', 'GENDER',
                                  'HDL01', 'HYPTMD01', 'RACEGRP', 'CIGT01', 'CIGTYR01',
                                  'GLUCOS01', 'CLVH01', 'V1AGE01', 'ENROLL_YR', 'PREVMI05',
                                  'PRVCHD05', 'PREVHF01', 'FORSMK01', 'FAST1202',
                                  'FAST0802', 'LDL02', 'BMI01', 'WSTHPR01', 'TGLEFH01',
                                  'MOMHISTORYSTR', 'DADHISTORYSTR', 'MOMHISTORYCHD',
                                  'DADHISTORYCHD', 'DADHISTORYDIA', 'MOMHISTORYDIA',
                                  'RANGNA01']},
              'sbpa02.csv':   {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'SBPA21', 'SBPA22']},
              'lipa.csv':     {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'LIPA01', 'LIPA02', 'LIPA08']},
              'hom.csv':      {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'HOM10D', 'HOM10E', 'HOM29', 'HOM31', 'HOM32', 
                                  'HOM35','HOM10A', 'HOM54', 'HOM12', 'HOM13', 'HOM14',
                                  'HOM15E', 'HOM15D', 'HOM16D', 'HOM16E', 'HOM18D', 
                                  'HOM18E', 'HOM19D', 'HOM19E', 'HOM20', 'HOM21', 'HOM22',
                                  'HOM23E', 'HOM23D', 'HOM24D', 'HOM24E', 'HOM26D', 
                                  'HOM26E', 'HOM27D', 'HOM27E', 'HOM55', 'HOM56']},
              'msra.csv':     {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'MSRA01', 'MSRA02', 'MSRA08F']},
              'anta.csv':     {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'ANTA01', 'ANTA07A', 'ANTA04']},
              'stroke01.csv': {'dirname': dirname_v1, 
                               'dict':['ID_C', 'TIA01', 'STROKE01', 'STIA01']},
              'phea.csv':     {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'PHEA06', 'PHEA07A', 'PHEA08', 'PHEA09A']},
              'atrfib11.csv': {'dirname': dirname_v1, 
                               'dict': ['ID_C', 'AF']},
              'status51.csv': {'dirname': dirname_v5, 
                               'dict': ['ID_C', 'KNWNDEADBYVISIT21', 'KNWNDEADBYVISIT31',
                                  'KNWNDEADBYVISIT41', 'KNWNDEADBYVISIT51', 'RESPOND22',
                                  'RESPOND32', 'RESPOND42', 'RESPOND52',
                                  'LASTFUINTERVIEW_DATE51_DAYS', 'STATUSDATE21_DAYS',
                                  'STATUSDATE31_DAYS', 'STATUSDATE41_DAYS', 'STATUSDATE51_DAYS']},
              'cderps16.csv': {'dirname': dirname_stroke, 
                               'dict': ['ID_C', 'FINALDX', 'EVENTYPE']},
              'cevtps16.csv': {'dirname': dirname_chd, 
                               'dict': ['ID_C', 'CMIDX']},
              'incps16.csv':  {'dirname': dirname_incident, 
                               'dict': ['ID_C', 'MI16', 'INDP16', 'FATCHD16', 
                                        'FUINC16', 'FT16DP', 'FUMI16']}}

binary_dict = {'0' : 0, '1' : 1, 'T' : np.nan}

def read_file(filename, files_dict):
    df = pd.read_csv(os.path.join(files_dict[filename]['dirname'], filename), engine='python')
    df = df.filter(files_dict[filename]['dict'])
    df = df.set_index('ID_C')
    df.columns = df.columns.str.lower()
    df.sort_index(inplace=True)
    return df

def get_prevstrk(df):
    prevstrk = np.where((df.stroke01 == 'Y') | (df.hom10d == 'Y'), 1, np.nan)
    prevstrk = np.where((df.stroke01 != 'Y') | (df.hom10d == 'N'), 0, prevstrk)
    return to_numeric(prevstrk)

def get_prevproc(df):
    prevproc = np.where((df.phea06 == 'N'), 0, np.nan)
    prevproc = np.where((df.phea06 == 'Y') & ((df.phea07a== 'N') & (df.phea09a== 'N')), 0, prevproc)
    prevproc = np.where((df.phea06 == 'Y') & ((df.phea07a== 'Y') | (df.phea09a== 'Y')), 1, prevproc)
    return to_numeric(prevproc)

def get_diabmed(df):
    diabmed = np.where((df.msra08f == 'N') | (df.msra02 == 'T'), 0, np.nan)
    diabmed = np.where((df.msra08f == 'Y'), 1, diabmed)
    return diabmed

def get_diabtsr(df):
    diabtsr = np.where((df.hom10e == 'N'), 0, np.nan)
    diabtsr = np.where((df.hom10e == 'Y'), 1, diabtsr)
    return diabtsr

def get_diabhx(df):
    diabhx = np.where((df.glucose >= 200) | ((df.glucose >= 126) & (df.glucose >= 126)) | 
                  (df.diabtsr == 1) | (df.diabmed == 1) | (df.hom10e == 'Y'), 
                  1, np.nan)
    diabhx = np.where((df.glucose > 0) & (df.glucose < 126) & 
                       (df.diabtsr ==0) & ((df.diabmed == 0) | (df.diabmed == np.nan)), 
                      0, diabhx)
    return diabhx

def get_diabt126(df):
    # replaced fast08 >= 1 with == '1' - check if there is an error somewhere earlier
    diabt126 = np.where((df.glucose >= 200) | ((df.glucose >= 126) & (df.fast08 == '1')) | 
                       (df.diabmed == 1), 
                        1, np.nan)
    diabt126 = np.where((df.glucose > 0) & ((df.glucose < 126) & (df.diabmed != 1)),
                        0, diabt126)
    diabt126 = np.where((df.glucose >= 126) & (df.glucose < 200) & (df.fast08 != '1') &
                        (df.diabmed == np.nan),
                        np.nan, diabt126)
    return diabt126

def get_defstroke(df):
    defstroke = np.where((df.finaldx == 'A') | (df.finaldx == 'B') | (df.finaldx == 'C') | (df.finaldx == 'D'), 1, 0)
    return defstroke

def get_strkdeath(df):
    strkdeath = np.where(df.defstroke & ((df.eventype == 'I') | (df.eventype == 'O')), 1, 0)
    return strkdeath

def get_defmi(df):
    defmi = np.where(df.cmidx == 'DEFMI', 1, 0)
    return defmi

def get_mi(df):
    mi = np.where((df.mi16==1) & (df.defmi==1), 1, 0)
    return mi

def get_stroke(df):
    stroke = np.where(((df.indp16==1) & (df.defstroke==1)) & (df.strkdeath==0), 1, 0)
    return stroke

def get_timetomi(df):
    timetomi = np.where(df.mi==1, df.fuinc16/365.25, np.nan)
    return timetomi

def get_timetostrk(df):
    timetostrk = np.where(df.strk==1, df.ft16dp/365.25, np.nan)
    return timetostrk

def get_timetochddeath(df):
    timetochddeath = np.where(df.chddeath==1, df.fuinc16/365.25, np.nan)
    return timetochddeath

def get_timetostrkdeath(df):
    timetostrkdeath= np.where(df.strkdeath==1, df.ft16dp/365.25, np.nan)
    return timetostrkdeath

# TODO: is it possible there's a later death time?
def get_timetodth(df):
    timetodth = np.where(df.knwndeadbyvisit51==1, df.statusdate51_days/365.25, np.nan)
    return timetodth
    
def lastfu(df):
    lastfu = (df
              .filter(items=['lastfuinterview_date51_days', 
                             'statusdate51_days'])
              .max(axis=1)
             )/365.25
    return lastfu


def dead(df):
    return df.knwndeadbyvisit51

def to_numeric(col):
    col = pd.to_numeric(col, errors='coerce')
    return col


cohort=None
for filename in ['derive13.csv', 'sbpa02.csv', 'lipa.csv', 'hom.csv', 'msra.csv',
                 'anta.csv', 'stroke01.csv', 'phea.csv', 'atrfib11.csv', 'status51.csv']:
    df = read_file(filename, files_dict)
    if cohort is not None:
        cohort = cohort.join(df)
    else:
        cohort = df
        
cohort.columns = cohort.columns.str.lower()

cohort = cohort.rename(columns={'sbpa21': 'sysbp',
                               'sbpa22': 'diabp',
                               'v1age01': 'age',
                               'hyptmd01': 'hyptmdsr',
                               'bmi01': 'bmi',    
                               'fast0802': 'fast08',
                               'fast1202': 'fast12',
                               'ldl02': 'ldlc',
                               'tglefh01': 'tglefh',
                               'glucos01': 'glucose',
                               'hdl01': 'hdlc',
                               'lipa01': 'totchol',
                               'lipa02': 'trigly',
                               'lipa08': 'apolpa',
                               'prevmi05': 'prevmi',
                               'prvchd05': 'prevchd',
                               'prevhf01': 'prevchf',
                               'diabts03': 'study_dm', 
                               'bmi01' : 'bmi'})

aric_v1 = (cohort
          .assign(hyptmdsr = lambda x: to_numeric(x.hyptmdsr.map(binary_dict)),
                       cholmed = lambda x: x.cholmdcode01.map(binary_dict),
                       cursmoke = lambda x: x.cursmk01.map(binary_dict),
                       prevap = lambda x: x.rangna01.map({1:1, 4:0}).values,
                       prevafib = lambda x: np.where((x.af == 1), 1, 0),
                       prevstrk = lambda x: get_prevstrk(x),
                       prevproc = lambda x: get_prevproc(x),
                       diabmed = lambda x: get_diabmed(x),
                       diabtsr = lambda x: get_diabtsr(x),
                       diabhx = lambda x: get_diabhx(x),
                       diabt126 = lambda x: get_diabt126(x),
                       exam = 1,
                       lastfu = lambda x: lastfu(x),
                       death = lambda x: dead(x),
                       prevmi = lambda x: to_numeric(x.prevmi),
                       prevchf = lambda x: to_numeric(x.prevchf),
                       sysbp = lambda x: to_numeric(x.sysbp),
                       timetodth = lambda x: get_timetodth(x))
           .filter(items = ['glucose', 'hdlc', 'ldlc', 'totchol', 'age',
                                 'sysbp', 'diabp', 'bmi', 'cholmed', 'hyptmdsr',
                                 'gender', 'racegrp', 'cursmoke', 'study_dm', 'trigly',
                                 'prevmi', 'prevap', 'prevafib', 'prevstrk', 'prevproc',
                                 'prevchf', 'diabt126', 'lastfu', 'death',
                                 'timetodth'
                                ]
                       )
          )
             
    
cderps16 = (read_file('cderps16.csv', files_dict)
            .assign(defstroke = lambda x: get_defstroke(x), 
                    strkdeath = lambda x: get_strkdeath(x))
            .groupby('ID_C')
            .max()
            .filter(items=['defstroke', 'strkdeath'])
            
           )

cevtps16 = (read_file('cevtps16.csv', files_dict)
            .assign(defmi = lambda x: get_defmi(x))
            .groupby('ID_C')
            .max()
            .filter(items=['defmi'])
           )

events = (read_file('incps16.csv', files_dict)
            .merge(cderps16, how='outer', on='ID_C')
            .merge(cevtps16, how='outer', on='ID_C')
            .rename(columns={'fatchd16': 'chddeath'})
            .assign(mi = lambda x: get_mi(x),
                    strk = lambda x: get_stroke(x),
                    timetomi = lambda x: get_timetomi(x),
                    timetostrk = lambda x: get_timetostrk(x),
                    timetochddeath = lambda x: get_timetochddeath(x),
                    timetostrkdeath = lambda x: get_timetostrkdeath(x)
                   )
           )

df_final = (aric_v1
            .merge(events, how='outer', on='ID_C')
            .assign(study = 'ARIC')
            .reset_index()
            .rename(columns={'ID_C': 'cohort_pid'})
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

df_final.to_csv(os.path.join(dirname_output, 'aric.csv'), index = False)
