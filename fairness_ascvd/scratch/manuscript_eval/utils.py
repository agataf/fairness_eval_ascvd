import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import math

def censoring_weights(df, model_type = 'KM'):

    if model_type == 'KM':
        censoring_model = KaplanMeierFitter()
    else:
        raise ValueError("censoring_model not defined")
    
    censoring_model.fit(df.query('is_train==1').event_time, 1.0*~df.query('is_train==1').event_indicator)
    
    weights = 1 / censoring_model.survival_function_at_times(df.event_time_10yr.values - 1e-5)
    weights_dict = dict(zip(df.index.values, weights.values))
    return weights_dict

def get_censoring(df, by_group=True, model_type='KM'):
    
    if by_group:
        weight_dict = {}
        for group in [1, 2, 3, 4]:
            group_df = df.query('grp==@group')
            group_weights_dict = censoring_weights(group_df, model_type)
            weight_dict.update(group_weights_dict)

    else:
        weight_dict = censoring_weights(cohort, censoring_model_type)

    weights = pd.Series(weight_dict, name='weights') 
    return weights

def add_ranges(df, threshold1 = 0.075, threshold2 = 0.2):
    
    range1 = (df.pred_probs < threshold1).astype(int)
    range2 = ((df.pred_probs >= threshold1) & (df.pred_probs < threshold2)).astype(int)
    range3 = ((df.pred_probs >= threshold2)).astype(int)
    rang = 1*range2 + 2*range3
        
    return rang

def treat_relative_risk(df):
    ldlc_reductions_by_treatment = {0: 1, 1: 0.7, 2: 0.5}
    relative_risk_statin = 0.75

    absolute_ldlc_reduction = df.ldlc*(1-df.treat.map(ldlc_reductions_by_treatment))

    return [math.pow(relative_risk_statin, el/38.7) for el in absolute_ldlc_reduction]

def log_reg(x):
    return 1/(1+np.exp(-1*x))

def run_pce_model(df):
    coefs = {1: [17.114, 0, 0.94, 0, -18.920, 4.475, 29.291, 
                 -6.432, 27.820, -6.087, 0.691, 0, 0.874],
             2: [-29.799, 4.884, 13.54, -3.114, -13.578, 3.149,
                 2.019, 0, 1.957, 0, 7.574, -1.665, 0.661],
             3: [2.469, 0, 0.302, 0, -0.307, 0, 1.916, 
                 0, 1.809, 0, 0.549, 0, 0.645],
             4: [12.344, 0, 11.853, -2.664, -7.990, 1.769, 
                 1.797, 0, 1.7864, 0, 7.837, -1.795, 0.658]
            }
    mean_risk = {1: 86.61, 2:-29.18, 3:19.54, 4:61.18}
    baseline_survival = {1:0.9533, 2:0.9665, 3:0.8954, 4:0.9144}

    data_df = (pd.DataFrame({'log(age)': np.log(df.age),
                             'log(age)^2': np.log(df.age)**2,
                             'log(totchol)': np.log(df.totchol),
                             'log(age)*log(totchol)': np.log(df.age)*np.log(df.totchol),
                             'log(hdlc)': np.log(df.hdlc),
                             'log(age)*log(hdlc)': np.log(df.age)*np.log(df.hdlc),
                             'rxbp*log(sysbp)': df.rxbp*np.log(df.sysbp),
                             'rxbp*log(age)*log(sysbp)': df.rxbp*np.log(df.age)*np.log(df.sysbp),
                             '(1-rxbp)*log(sysbp)': (1-df.rxbp)*np.log(df.sysbp),
                             '(1-rxbp)*log(age)*log(sysbp)': (1-df.rxbp)*np.log(df.age)*np.log(df.sysbp),
                             'cursmoke': df.cursmoke,
                             'log(age)*cursmoke': df.cursmoke*np.log(df.age),
                             'diabt126': df.diabt126
                       }
                      )
         )

    risks = []
    for group in [1,2,3,4]:
        relative_risk = (data_df
                         .iloc[np.where(df.grp==group)]
                         .multiply(coefs[group])
                         .sum(axis=1)
                         .sub(mean_risk[group])
                         .transform(np.exp)
                        )
        risk = 1 - pow(baseline_survival[group], relative_risk)
        risks.append(risk)

    risks = pd.concat(risks).sort_index()
    risks.name='pred_probs'
    
    return risks

def run_revised_pce_model(df):
    coefs = {'women': [0.106501, 0.432440, 0.000056, 0.017666, 0.731678,
                       0.943970, 1.009790, 0.151318, -0.008580, -0.003647,
                       0.006208, 0.152968, -0.000153, 0.115232, -0.092231, 
                       0.070498, -0.000173, -0.000094, -12.823110],
             'men': [0.064200, 0.482835, -0.000061, 0.038950, 2.055533,
                     0.842209, 0.895589, 0.193307, 0, -0.014207,
                     0.011609, -0.119460, 0.000025, -0.077214, -0.226771,
                     -0.117749, 0.004190, -0.000199, -11.679980]}

    groups_dict = {1: 'women', 2: 'women', 3: 'men', 4: 'men'}

    data_df = (pd.DataFrame({'sex': df.grp.map(groups_dict),
                             'age': df.age,
                             'black': df.race_black,
                             'sysbp^2': df.sysbp**2,
                             'sysbp': df.sysbp,
                             'rxbp': df.rxbp,
                             'diabt': df.diabt126,
                             'cursmoke': df.cursmoke,
                             'totchol/hdlc': df.totchol/df.hdlc,
                             'age_if_black': df.age*df.race_black,#only women
                             'sysbp_if_rxbp': df.sysbp*df.rxbp,
                             'sysbp_if_black': df.sysbp*df.race_black,
                             'black_and_rxbp': df.rxbp*df.race_black, 
                             'age*sysbp': df.age*df.sysbp, 
                             'black_and_diabt': df.diabt126*df.race_black, 
                             'black_and_cursmoke': df.cursmoke*df.race_black,
                             'totchol/hdlc_if_black': df.totchol/df.hdlc*df.race_black,
                             'sysbp_if_black_and_rxbp': df.sysbp*df.rxbp*df.race_black,
                             'age*sysbp_if_black': df.sysbp*df.age*df.race_black}
                           )
               .assign(intercept=1)
         )

    risks = []
    for sex in ['women','men']:
        risk = (data_df
                .query("sex==@sex")
                .drop(columns='sex')
                .multiply(coefs[sex])
                .sum(axis=1)
                .apply(lambda x:log_reg(x))
                        )
        risks.append(risk)

    risks = pd.concat(risks).sort_index()
    risks.name='pred_probs'
    
    return risks