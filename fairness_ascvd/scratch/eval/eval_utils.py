import pandas as pd
#import os
import numpy as np

from prediction_utils.pytorch_utils.metrics import (
    CalibrationEvaluator
)

grp_label_dict = {1: 'Black women', 2: 'White women', 3: 'Black men', 4: 'White men'} 

def get_calib_probs(model, x, transform=None):
    
    if transform=='log':
        model_input = np.log(x)
    else:
        model_input = x
        
    calibration_density = model.predict_proba(model_input.reshape(-1, 1))[:, -1]
                    
    df = pd.DataFrame({'pred_probs': x,
                       'model_input': model_input,
                       'calibration_density': calibration_density})  
    return df
    
def get_calib_model(preds_test, transform=None):
    
    evaluator = CalibrationEvaluator()
    _, model = evaluator.get_calibration_density_df(preds_test.labels,
                                                     preds_test.pred_probs,
                                                     preds_test.weights,
                                                     transform = transform)

    return model

def plot_calibration(df, zoom=False, output_path=None, row_var='lambda_reg'):   
    
    df = df.assign(group = lambda x: x.group.map(grp_label_dict))
    g = sns.relplot(data = df,
                    x = 'pred_probs',
                    y = 'calibration_density',
                    kind = 'line',
                    hue = 'group',
                    row = row_var,
                    legend = True,
                    ci = 'sd',
                    err_style = 'bars',
                    aspect = 1.2)
    axes = g.axes.flatten()
    for ax in axes:
        ax.axline(xy1 = (0,0), slope = 1, color = "b", dashes = (5, 2), label = "Perfectly calibrated")
        ax.axvline(x = 0.075, linestyle = '--', color = 'grey')
        ax.axvline(x = 0.2,   linestyle = '--', color = 'grey')
        ax.axhline(y = 0.075, linestyle = '--', color = 'grey')
        ax.axhline(y = 0.2,   linestyle = '--', color = 'grey')

        if zoom:
            ax.set(xlim = (0, 0.25), ylim = (0, 0.25))
        else:
            ax.set(xlim = (0, 1), ylim = (0, 1))

        ax.set_xlabel("predicted risk")
        ax.set_ylabel("fraction of positives")
    
    if output_path is not None:
        g.savefig(output_path)