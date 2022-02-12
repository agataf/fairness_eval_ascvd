import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--preds_path', type=str, help='path where predictions of any of the experiments are stored')
parser.add_argument('--cohort_path', type=str, help='path where input cohorts are stored')
parser.add_argument('--result_path', type=str, help='path where inference results will be stored')

args = parser.parse_args()

df = pd.read_csv(args.cohort_path)

df = df.assign(grp = np.where((df.grp==1) & (df.race_black==0),
                              2,
                              np.where((df.grp==3) & (df.race_black==0),
                                        4,
                                        df.grp)
                             )
              )

id_to_study_dict = df.filter(['person_id', 'study']).set_index('person_id').to_dict()['study']
person_to_grp_dict = df.filter(['person_id', 'grp']).set_index('person_id').to_dict()['grp']

preds = (pd
         .read_csv(args.preds_path)
         .assign(study = lambda x: [id_to_study_dict.get(el) for el in x.person_id],
                 grp = lambda x: [person_to_grp_dict.get(el) for el in x.group])
        )

def get_means(df, aggregate_by):
    return (df
            .groupby(aggregate_by)
            .mean()
            .filter(['age', 'ascvd_10yr', 'censored_10yr', 'event_time_10yr'])
            .reset_index()
           )

def get_counts(df, aggregate_by):
    return (df
            .groupby(aggregate_by)
            .count()
            .filter(['person_id'])
            .rename(columns={'person_id': 'counts'})
            .reset_index()
         )

def get_adjusted_prevalence(df, aggregate_by):
    weighted_counts = (df
                       .groupby(aggregate_by + ['labels'])
                       .agg({'person_id': 'count', 'weights': 'mean'})
                       .rename(columns={'person_id': 'counts', 'weights': 'mean_ipcw'})
                       .assign(weighted_counts = lambda x: x.counts*x.mean_ipcw)
                       .filter(['weighted_counts'])
                       .reset_index()
                      )

    weighted_prev = (weighted_counts
                     .assign(labels = lambda x: x.labels.astype(str))
                     .pivot(index=aggregate_by, columns='labels', values='weighted_counts')
                     .assign(prev_weighted = lambda x: x['1']/(x['0']+x['1']))
                     .reset_index()
                     .rename_axis('', axis='columns')
                     .filter(aggregate_by + ['prev_weighted'])
                     #.round(3)
                    )
    return weighted_prev

    
    
means = get_means(df, aggregate_by=['study', 'grp']) 
means_group = get_means(df, aggregate_by=['grp']).assign(study='all')
means_all = get_means(df.assign(grp='all', study='all'), aggregate_by=['grp', 'study'])

means = pd.concat([means, means_group, means_all]).reset_index(drop=True)

counts = get_counts(df, aggregate_by=['study', 'grp']) 
counts_group = get_counts(df, aggregate_by=['grp']).assign(study='all')
counts_all = get_counts(df.assign(grp='all', study='all'), aggregate_by=['grp', 'study'])

counts = pd.concat([counts, counts_group, counts_all]).reset_index(drop=True)

adj_prev = get_adjusted_prevalence(preds, aggregate_by=['group', 'study']).fillna(0)
adj_prev_group = get_adjusted_prevalence(preds, aggregate_by=['group']).fillna(0).assign(study='all')
adj_prev_all = get_adjusted_prevalence(preds.assign(group='all', study='all'), aggregate_by=['group', 'study']).fillna(0)

adj_prev = pd.concat([adj_prev, adj_prev_group, adj_prev_all]).reset_index(drop=True)

os.makedirs(args.result_path, exist_ok=True)

(means
 .merge(counts, on=['study', 'grp'])
 .merge(adj_prev.rename(columns={'group': 'grp'}), on=['study', 'grp'])
 .filter(['study', 'grp', 'counts', 'age', 'prev_weighted', 'censored_10yr'])
 .rename(columns={'grp': 'group'})
 .assign(group = lambda x: pd.Categorical(x.group.map({1: 'Black women', 2: 'non-Black women',
                                                       3: 'Black men', 4: 'non-Black men', 'all': 'All'}), 
                                          categories = ['Black women', 'non-Black women', 'Black men', 'non-Black men', 'All'],
                                          ordered=True),
        counts = lambda x: x.counts.round(0),
         prev_weighted = lambda x: [str(el)+'%' for el in (x.prev_weighted*100).round(2)],
         censored_10yr = lambda x: [str(el)+'%' for el in (x.censored_10yr*100).round(2)]
        )
 .groupby(['group','study']).max()
 .dropna()
 .round(2)
).to_csv(os.path.join(args.result_path, 'Table1.csv'))