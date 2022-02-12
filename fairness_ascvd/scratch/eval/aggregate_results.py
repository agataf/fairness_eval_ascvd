import pandas as pd
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--model_type", type=str, required=True)
parser.add_argument("--cohort_path", type=str, help="path where input cohorts are stored", required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts/cohort/all_cohorts.csv')
parser.add_argument("--base_path", type=str, required=False,
                   default='/labs/shahlab/projects/agataf/data/cohorts/pooled_cohorts')

args = parser.parse_args()

experiment_configs = pd.read_csv(os.path.join(args.base_path, 'experiments', args.experiment_name,
                                 'config', 'config.csv'))
config_ids = experiment_configs.id.values

aggregate_path = os.path.join(args.base_path, 'experiments', args.experiment_name, 'performance', 'all')
os.makedirs(aggregate_path, exist_ok = True)

standard_eval = []
fair_eval = []
outputs=[]
for config_id in config_ids:
    for fold_id in range(1,11):
        RESULT_PATH = os.path.join(args.base_path, 'experiments', args.experiment_name, 'performance',
                                           '.'.join((str(config_id), 'yaml')), str(fold_id))
        LOGGING_PATH = os.path.join(RESULT_PATH, 'training_log.log')
        
        CONFIG_PATH = os.path.join(args.base_path, 'experiments', args.experiment_name, 'config',
                                                   '.'.join((str(config_id), 'yaml')))


        config = yaml.load(open(CONFIG_PATH), Loader=yaml.SafeLoader)
        lambda_reg = config.get('lambda_group_regularization')
        if lambda_reg is None:
            model_id = 0
        else:
            model_id = lambda_reg

        result_df_group_standard_eval = (pd.read_parquet(
            os.path.join(RESULT_PATH,
                         'result_df_group_standard_eval.parquet'
                        ))
                                         .assign(fold_id=fold_id,
                                                 config_id=config_id,
                                                 model_id=model_id,
                                                model_type=args.model_type))
        standard_eval.append(result_df_group_standard_eval)
        
        result_df_group_fair_ova = (pd.read_parquet(
            os.path.join(RESULT_PATH,
                         'result_df_group_fair_ova.parquet'
                        ))
                                         .assign(fold_id=fold_id,
                                                 config_id=config_id,
                                                 model_id=model_id,
                                                model_type=args.model_type))
        fair_eval.append(result_df_group_fair_ova)
        
        output_df = (
            pd.read_parquet(
                os.path.join(RESULT_PATH,
                             'output_df.parquet'
                            )
            )
            .assign(fold_id    = fold_id,
                    config_id  = config_id,
                    model_id   = model_id,
                    model_type = args.model_type)
        )
        
        outputs.append(output_df)

df = pd.concat(standard_eval)
df_fair = pd.concat(fair_eval)
preds = pd.concat(outputs)

df.to_csv(os.path.join(aggregate_path, 'standard_evaluation.csv'), index=False)
df_fair.to_csv(os.path.join(aggregate_path, 'fairness_evaluation.csv'), index=False)
preds.to_csv(os.path.join(aggregate_path, 'predictions.csv'), index=False)
