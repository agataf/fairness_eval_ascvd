import pandas as pd
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--experiment_path", type=str, help="path to directory where experiment results are stored")

args = parser.parse_args()

# identify names of config files for a given experiment
experiment_configs = pd.read_csv(os.path.join(args.experiment_path, 'config', 'config.csv'))
config_ids = experiment_configs.id.values

outputs = []

for config_id in config_ids:
    for fold_id in range(1,11):
        
        model_path = os.path.join(args.experiment_path,
                                  'performance',
                                  '.'.join((str(config_id), 'yaml')),
                                  str(fold_id)
                                 )
        
        config_path = os.path.join(args.experiment_path,
                                   'config', 
                                   '.'.join((str(config_id), 'yaml'))
                                  )
        
        config = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        lambda_reg = config.get('lambda_group_regularization')
        
        if lambda_reg is None:
            model_id = 0
        else:
            model_id = lambda_reg

        output_df = (
            pd.read_parquet(
                os.path.join(model_path,
                             'output_df.parquet'
                            )
            )
            .assign(fold_id    = fold_id,
                    config_id  = config_id,
                    model_id   = model_id,
                    model_type = args.model_type)
        )
        
        outputs.append(output_df)


preds = pd.concat(outputs)

# Save aggregated results
aggregate_path = os.path.join(args.experiment_path, 'performance', 'all')
os.makedirs(aggregate_path, exist_ok = True)
preds.to_csv(os.path.join(aggregate_path, 'predictions.csv'), index=False)
