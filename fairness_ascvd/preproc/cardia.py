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

study_path = 'CARDIA'
study_path = os.path.join(args.input_dir, study_path)

dirname_yr0 = os.path.join(study_path,'Y00/DATA/csv')
dirname_yr2 = os.path.join(study_path,'Y02/DATA/csv')
dirname_yr10 = os.path.join(study_path,'Y10/DATA/csv')
dirname_yr15 =  os.path.join(study_path,'Y15/DATA/csv')
dirname_yr20 =  os.path.join(study_path,'Y20/DATA/csv')
dirname_events =  os.path.join(study_path,'DATA/csv')

yr0_file_dict = {
    'aaf08v2':['pid', 'a08heart'],
    'aaf09gen': ['pid', 'a09hrtak', 'a09angin', 'a09chf'],
}

yr10_file_dict = {'eaf08': ['pid', 'e08ancth', 'e08tia', 'e08hrtak']}

yr15_file_dict = {'faf05': ['pid', 'f05fstmn'],
                  'faf08': ['pid', 'f08hbp', 'f08hbnow', 'f08diab', 'f08chnow', 
                            'f08preg', 'f08hrtak', 'f08tia',
                            'f08angin', 'f08othht'],
                  'falip': ['pid', 'fl1chol', 'fl1hdl', 'fl1ntrig', 'fl1ldl'],
                  'faf02': ['pid', 'f02avgsy', 'f02avgdi', 'f02pulse'],
                  'faf10': ['pid', 'f10cigs', 'f10tobac'],
                  'faf09tob': ['pid', 'f09smknw', 'f09cgtdy'],
                  'faref': ['pid', 'race', 'sex', 'ex6_age', 'f02date', 'f03ed'],
                  'faglu': ['pid', 'fl7glu'],
                  'faf20': ['pid', 'f20bmi', 'f20wgt', 'f20wst1'],
                  'fains': ['pid', 'fl7ins']}

exams_var_dict = {'iaref': ['pid', 'x9status', 'b02date', 'c02date', 'd02date',
                          'e02date', 'f02date', 'g02date', 'h02date', 'i02date'],
                'haref': ['pid', 'x8status'],
                'garef': ['pid', 'x7status'],
                'faref': ['pid', 'x6status'],
                'earef': ['pid', 'x5status'],
                'daref': ['pid', 'x4status'],
                'caref': ['pid', 'x3status'],
                'baref': ['pid', 'x2status'], 
                'aaref': ['pid', 'a02date']}

exams_path_dict = {'iaref':  os.path.join(study_path,'Y30/DATA/csv'),
                'haref':  os.path.join(study_path,'Y25/DATA/csv'),
                'garef':  os.path.join(study_path,'Y20/DATA/csv'),
                'faref':  os.path.join(study_path,'Y15/DATA/csv'),
                'earef':  os.path.join(study_path,'Y10/DATA/csv'),
                'daref':  os.path.join(study_path,'Y07/DATA/csv'),
                'caref':  os.path.join(study_path,'Y05/DATA/csv'),
                'baref':  os.path.join(study_path,'Y02/DATA/csv'), 
                'aaref':  os.path.join(study_path,'Y00/DATA/csv')}

fu_var_dict = {'aaflwup1': ['pid', 'fm006rsp', 'fm012rsp', 'fm018rsp'],
               'baflwup1': ['pid', 'fm030rsp', 'fm036rsp', 'fm042rsp',
                            'fm048rsp', 'fm054rsp'],
               'caflwp1b': ['pid', 'fm066rsp', 'fm072rsp', 'fm078rsp'],
               'daflwup': ['pid', 'fm090rsp', 'fm096rsp', 'fm102rsp',
                           'fm108rsp', 'fm114rsp'],
               'gaflwup1': ['pid', 'fm252rsp', 'fm264rsp', 'fm276rsp',
                            'fm288rsp']
              }
               
fu_path_dict = {'aaflwup1':  os.path.join(study_path,'Y00/DATA/csv'),
                'baflwup1':  os.path.join(study_path,'Y02/DATA/csv'),
                'caflwp1b':  os.path.join(study_path,'Y05/DATA/csv'),
                'daflwup':  os.path.join(study_path,'Y07/DATA/csv'),
                'gaflwup1':  os.path.join(study_path,'Y25/DATA/csv')
               }


def read_file(filename, path, files_dict, index):
    df = pd.read_csv(os.path.join(path, '.'.join((filename, 'csv'))), engine='python')
    df = (df
          .rename(columns = lambda x: x.lower())
          .filter(files_dict[filename])
          .set_index(index)
          .sort_index()
         )

    return df

def read_file_events(filename, path, index):
    df = pd.read_csv(os.path.join(path, '.'.join((filename, 'csv'))), engine='python')
    df = (df
          .rename(columns = lambda x: x.lower())
          .set_index(index)
          .sort_index()
         )

    return df

def get_smokstat(df):
    smokstat = np.where((df.f10cigs==1), 3, np.nan)
    smokstat = np.where((df.f10cigs==2) & (df.f09smknw == 1), 2, smokstat)
    smokstat = np.where((df.f10cigs==2) & (df.f09smknw == 2), 1, smokstat)
    return smokstat

def get_cursmoke(df):
    cursmoke = np.where((df.smokstat==1), 1, np.nan)
    cursmoke = np.where(((df.smokstat==2) | (df.smokstat==3)), 0, cursmoke)
    return cursmoke



def get_hyptmdsr(df):
    hyptmdsr = np.where((df.f08hbnow == 1) | ((df.f08hbnow == 2) & (df.f08hbp == 1)), 0, np.nan)
    hyptmdsr = np.where((df.f08hbnow == 1) | ((df.f08hbnow == 2) & (df.f08hbp == 2)), 1, hyptmdsr)
    #hyptmdsr = np.where((df.f08bpmed == 2) & (df.f09mdnow == 1), 0, hyptmdsr)
    hyptmdsr = np.where((df.f08hbnow == 8), np.nan, hyptmdsr)
    return hyptmdsr

def get_cholmed(df):
    cholmed = df.f08chnow.map({1:0, 2:1})
    return cholmed

def get_diabt126(df):
    fast08 = np.where((df.f05fstmn/60 < 8), 0, 1)
    fast12 = np.where((df.f05fstmn/60 < 12), 0, 1)
    glucose = df.fl7glu*0.94359772+6.979619035
    diabt126 = np.where((glucose >=200) | ((glucose>=126) & (fast08>=1)), 1, np.nan)
    diabt126 = np.where((glucose > 0) & (glucose < 126), 0, diabt126)
    return diabt126
    
def get_diabhx(df):
    diabhx = np.where((df.f08diab==2) & ((df.f09dibst>=3) | (df.f09dibst) <= 4), 1, np.nan)
    diabhx = np.where((df.f08diab==1) | ((df.f08diab==2) & (df.f09dibst) == 5), 0, diabhx)
    return diabhx

def get_lastexam(df):
    lastexam = df.a02date/12
    lastexam = np.where(df.x2status=="E", df.b02date/12, np.nan)
    lastexam = np.where(df.x3status=="E", df.c02date/12, lastexam)
    lastexam = np.where(df.x4status=="E", df.d02date/12, lastexam)
    lastexam = np.where(df.x5status=="E", df.e02date/12, lastexam)
    lastexam = np.where(df.x6status=="E", df.f02date/12, lastexam)
    lastexam = np.where(df.x7status=="E", df.g02date/12, lastexam)
    lastexam = np.where(df.x8status=="E", df.h02date/12, lastexam)
    lastexam = np.where(df.x9status=="E", df.i02date/12, lastexam)
    
    return lastexam
    
def get_prevchd(df):
    prevchd = np.where((df.a09hrtak==1) | (df.a08heart==1), 0, np.nan)
    prevchd = np.where(df.a09hrtak==2, 1, prevchd)
    prevchd = np.where(df.a08heart==8, np.nan, prevchd)
    return prevchd


def get_prevap(df):
    prevap = np.where((df.a09angin==1) | (df.a08heart==1), 0, np.nan)
    prevap = np.where(df.a09angin==2, 1, prevap)
    prevap = np.where(df.a08heart==8, np.nan, prevap)
    return prevap

def get_prevchf(df):
    return df.a09chf.map({8:1}).fillna(0)
    
# exam at year 15

yr15_frames = [read_file(filename, dirname_yr15, yr15_file_dict, index='pid') for filename in yr15_file_dict.keys()]
yr15_df = (pd
          .concat(yr15_frames, axis=1)
          .assign(smokstat = lambda x: get_smokstat(x), 
                  cursmoke = lambda x: get_cursmoke(x), 
                  hyptmdsr = lambda x: get_hyptmdsr(x),
                  cholmed = lambda x: get_cholmed(x), 
                  diabt126 = lambda x: get_diabt126(x),
                  heartrate = lambda x: x.f02pulse*2,
                  exam15time = lambda x: x.f02date/12,
                  prevmi15 = lambda x: np.where(x.f08hrtak==1, 1, 0),
                  prevap15 = lambda x: np.where(x.f08angin==2, 1, 0),
                  prevchd15 = lambda x: np.where(x.f08othht==2, 1, 0),
                  gender = lambda x: x.sex.map({1: 'M', 2: 'F'}),
                  racegrp = lambda x: x.race.map({4: 'B', 5: 'W'}),
                  exam = 15)
           .rename(columns = {'f20bmi': 'bmi',
                             'ex6_age': 'age',
                             'f02avgsy': 'sysbp',
                             'f02avgdi': 'diabp',
                             'fl1chol': 'totchol',
                             'fl1ntrig': 'trigly',
                             'fl1hdl': 'hdlc',
                             'fl1ldl': 'ldlc'
                             })
           .filter(['smokstat', 'cursmoke', 'hyptmdsr', 'cholmed', 'diabt126',
                    'heartrate', 'exam', 'bmi', 'age', 'racegrp',
                    'gender', 'exam15time', 'sysbp', 'diabp',
                    'totchol', 'hdlc', 'ldlc', 'trigly', 'prevmi15', 'prevap15',
                   'prevchd15'])
         )

# dates of exams

frames_exams = [read_file(filename, exams_path_dict[filename],
                          exams_var_dict, index='pid')
                for filename in exams_var_dict.keys()]

exams = (pd
         .concat(frames_exams, axis=1)
         .assign(lastexam = lambda x: get_lastexam(x))#,
                #exam15time = lambda x: x.f02date/12)
         #.filter(['lastexam', 'exam15time'])
         .filter(['lastexam'])
        )

# dates of follow-ups

frames_fu = [read_file(filename, fu_path_dict[filename], 
                       fu_var_dict, index='pid')
             for filename in fu_var_dict.keys()]

fu_df = (pd
         .concat(frames_fu, axis=1)
         .filter(items=['fm006rsp', 'fm012rsp', 'fm018rsp', 'fm030rsp',
                        'fm036rsp', 'fm042rsp', 'fm048rsp', 'fm054rsp',
                        'fm066rsp', 'fm072rsp', 'fm078rsp', 'fm090rsp',
                        'fm096rsp', 'fm102rsp', 'fm108rsp', 'fm114rsp', 
                        'fm330rsp', 'fm336rsp', 'fm342rsp', 'fm348rsp'])
         .assign(lastfu = lambda x: x.max(axis=1)/12)
         .filter(['lastfu'])
)

# dates of events

# some people have both strk and mi at the same time
events = read_file_events('outcomes2016', 
                          os.path.join(study_path,'OUTCOMES/DATA/csv'), 
                          index='pid')
events = (events
           .rename(columns = {'strokea': 'strk',
                              'dead': 'death'})
           .assign(strkdeath = lambda x: np.where(x.strokeafnf & ~x.strk, 1, 0),
                   chddeath = lambda x: np.where(x.chdafnf & ~x.chda, 1, 0,),
                   timetomi = lambda x: np.where(x.mi, x.miatt/365.25, np.nan),
                   timetostrk = lambda x: np.where(x.strk, x.strokeaatt/365.25, np.nan),
                   timetostrkdeath = lambda x: np.where(x.strkdeath, x.strokeafnfatt/365.25, np.nan),
                   timetochddeath = lambda x: np.where(x.chddeath, x.chdafnfatt/365.25, np.nan),
                   timetodth = lambda x: np.where(x.death, x.deathatt/365.25, np.nan),
                   timetochf = lambda x: np.where(x.chf, x.chfatt/365.25, np.nan),
                   timetochd = lambda x: np.where(x.chda, x.chdaatt/365.25, np.nan),
                   timetoafib = lambda x: np.where(x.afib, x.afibatt/365.25, np.nan)
                  )
           .filter(['strk', 'strkdeath', 'mi', 'chddeath', 'timetostrk', 'timetostrkdeath',
                    'timetomi', 'timetodth', 'timetochddeath', 'death', 'timetochf',
                    'timetochd', 'timetoafib'])
         )

df_all = (yr15_df
          .merge(exams, how='outer', on='pid')
          .merge(events, how='outer', on='pid')
          .merge(fu_df, how='outer', on='pid')
         )

# update time to start at year 15

df_all_times = (df_all
                  .filter(['lastexam', 'exam15time', 'timetostrk', 
                          'timetostrkdeath', 'timetomi', 'timetodth',
                          'timetochddeath', 'timetochf', 'timetochd',
                          'timetoafib', 'lastfu'])
                 .subtract(df_all.exam15time, axis=0)
                 .assign(prevstrkEv = lambda x: np.where(x.timetostrk<=0, 1, 0),
                     prevmiEv = lambda x: np.where(x.timetomi<=0, 1, 0),
                     prevchfEv = lambda x: np.where(x.timetochf<=0, 1, 0),
                     prevchdEv = lambda x: np.where(x.timetochd<=0, 1, 0),
                     prevafibEv = lambda x: np.where(x.timetoafib<=0, 1, 0))
                 )

df_all.update(df_all_times)

# find prevalent diseases - corresponding to exclusion criteria

yr0_list = ['aaf08v2', 'aaf09gen']
yr0_frames = [read_file(filename, dirname_yr0, yr0_file_dict, index='pid') for filename in yr0_list]

yr0_prevs = (pd
          .concat(yr0_frames, axis=1)
          .assign(prevchd1 = lambda x: get_prevchd(x),
                 prevap1 = lambda x: get_prevap(x),
                 prevchf1 = lambda x: get_prevchf(x))
          .filter(['prevchd1', 'prevap1', 'prevchf1'])

         )  

yr10_prevs = read_file('eaf08', dirname_yr10, yr10_file_dict, index='pid')
yr10_prevs = (yr10_prevs
           .assign(prevproc10 = lambda x: x.e08ancth.map({2:1}).fillna(0),
                   prevmi10 = lambda x: x.e08hrtak.map({1:1}).fillna(0),
                   prevstrk10 = lambda x: x.e08tia.fillna(0))
           .filter(['prevproc10', 'prevmi10', 'prevstrk10'])
          )

yr15_prevs = df_all_times.filter(['prevmi15', 'prevap15', 'prevchd15', 'prevstrkEv', 'prevmiEv', 'prevchfEv', 'prevchdEv',
       'prevafibEv'])

all_prevs = (yr0_prevs
             .merge(yr10_prevs, how='outer', on='pid')
             .merge(yr15_prevs, how='outer', on='pid')
             .assign(prevchd = lambda x: x.filter(['prevchd1', 'prevchd15', 'prevchdEv']).max(axis=1),
                    prevap = lambda x: x.filter(['prevap1', 'prevap15']).max(axis=1),
                    prevchf = lambda x: x.filter(['prevchf1', 'prevchfEv']).max(axis=1),
                    prevproc = lambda x: x.prevproc10,
                    prevmi = lambda x: x.filter(['prevmi10', 'prevmi15', 'prevmiEv']).max(axis=1),
                     prevstrk = lambda x: x.filter(['prevstrk10', 'prevstrkEv']).max(axis=1),
                     prevafib = lambda x: x.prevafibEv
                    )
             .filter(['prevchd', 'prevap', 'prevchf', 'prevproc', 'prevmi', 'prevstrk', 'prevafib'])
)

df_final = (df_all
            .merge(all_prevs, how='outer', on='pid')
            .assign(study = 'CARDIA')
            .reset_index()
            .rename(columns={'pid': 'cohort_pid'})
            .filter(['timetomi', 'timetostrk', 'timetochddeath', 'timetostrkdeath', 'timetodth',
                             'lastexam', 'lastfu', 
                             'mi', 'strk', 'chddeath', 'strkdeath', 'death',
                             'prevmi', 'prevstrk', 'prevproc', 'prevchf', 'prevcvd',
                             'prevap', 'prevang', 'prevchd', 'prevafib',
                             'cohort_pid', 'racegrp', 'gender', 'age', 'study',
                             'cursmoke', 'hyptmdsr', 'cholmed', 'diabt126',  'totchol',
                             'ldlc', 'trigly', 'hdlc', 'sysbp' , 'diabp', 'bmi'])
)

df_final.to_csv(os.path.join(dirname_output, 'cardia.csv'), index = False)