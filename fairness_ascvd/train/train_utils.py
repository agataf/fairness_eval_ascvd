import pandas as pd
import os
import sys
import numpy as np
import torch
import logging
import math

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
# from fairness_ascvd.prediction_utils.pytorch_utils.datasets import ArrayDataset
# from fairness_ascvd.prediction_utils.pytorch_utils.models import TorchModel
# from fairness_ascvd.prediction_utils.pytorch_utils.robustness import GroupDROModel

# from fairness_ascvd.prediction_utils.util import yaml_write

# from fairness_ascvd.prediction_utils.pytorch_utils.layers import LinearLayer
# from fairness_ascvd.prediction_utils.pytorch_utils.lagrangian import MultiLagrangianThresholdRateModel
# from fairness_ascvd.prediction_utils.pytorch_utils.group_fairness import EqualThresholdRateModel

from prediction_utils.pytorch_utils.datasets import ArrayDataset
from prediction_utils.pytorch_utils.models import TorchModel
from prediction_utils.pytorch_utils.robustness import GroupDROModel

from prediction_utils.util import yaml_write

from prediction_utils.pytorch_utils.layers import LinearLayer
from prediction_utils.pytorch_utils.lagrangian import MultiLagrangianThresholdRateModel
from prediction_utils.pytorch_utils.group_fairness import EqualThresholdRateModel


class EqualThresholdRateModelLinear(EqualThresholdRateModel):
    """
    Override default to use logistic regression
    """
    def init_model(self):
        return LinearLayer(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )

class MultiLagrangianThresholdModelLinear(MultiLagrangianThresholdRateModel):
    """
    Override default to use logistic regression
    """
    def init_model(self):
        return LinearLayer(
            self.config_dict["input_dim"], self.config_dict["output_dim"]
        )


class Dataset:


    def __init__(self, df, feature_columns, deg=1, val_fold_id = '1', test_fold_id = 'test', eval_fold_id = 'eval', batch_size=128):

        if deg > 1:
            # TODO: consider changing how interactions are constructed
            poly_fitter = PolynomialFeatures(degree=deg, include_bias=False)
            self.features = poly_fitter.fit_transform(df[feature_columns].to_numpy())
            self.feature_names = poly_fitter.get_feature_names_out(feature_columns)
            #self.feature_names = poly_fitter.get_feature_names(feature_columns)
            assert len(self.feature_names) == self.features.shape[1]
        else:
            self.features = df[feature_columns].to_numpy()
            self.feature_names = feature_columns
        
        self.scaler = StandardScaler()
        
        self.df_dict_uncensored = {
            'train': df.query('(fold_id != @val_fold_id) & (fold_id != @test_fold_id) & (censored_10yr != 1.0)'),
            'val': df.query('(fold_id == @val_fold_id) & (censored_10yr != 1.0)'),
            'test': df.query('fold_id == @test_fold_id & (censored_10yr != 1.0)'),
            'eval': df.query('fold_id == @eval_fold_id & (censored_10yr != 1.0)')
        }
    
        self.weight_dict_uncensored = {
            key: value['weights'].values
            for key, value in self.df_dict_uncensored.items()
        }
        
        self.group_dict_uncensored = {
            key: value['grp'].values
            for key, value in self.df_dict_uncensored.items()
        }
        
        self.row_id_dict_uncensored = {
            key: value.index
            for key, value in self.df_dict_uncensored.items()
        }

        self.features_dict_uncensored = {
            key: self.features[value.index]
            for key, value in self.df_dict_uncensored.items()
        }

        self.labels_dict_uncensored = {
            key: value['ascvd_10yr'].values
            for key, value in self.df_dict_uncensored.items()
        }

        self.df_dict_all = {
            'train': df.query('(fold_id != @val_fold_id) & (fold_id != @test_fold_id)'),
            'val': df.query('(fold_id == @val_fold_id)'),
            'test': df.query('fold_id == @test_fold_id'),
            'eval': df.query('fold_id == @eval_fold_id')
        }

        self.features_dict_all = {
            # todo: used to be value.person_id.values instead of value.index, 
            # but needed to change it for the case of splitting. make sure nothing's broken.
            key: self.features[value.index]
            for key, value in self.df_dict_all.items()
        }

        self.event_time_dict_all = {
            key: value['event_time'].values
            for key, value in self.df_dict_all.items()
        }

        self.event_indicator_dict_all = {
            key: value['event_indicator'].values
            for key, value in self.df_dict_all.items()
        }
        
        # Preparing data for feeding to torch models
        
        self.scaler.fit(self.features_dict_uncensored['train'])
        
        self.features_dict_uncensored_scaled = {
            key: self.scaler.transform(value)
            for key, value in self.features_dict_uncensored.items()
        }
        
        self.dataset_dict = {
            key: ArrayDataset(
                {
                    'features': torch.FloatTensor(self.features_dict_uncensored_scaled[key]), 
                    'labels': torch.LongTensor(self.labels_dict_uncensored[key]),
                    'row_id': torch.LongTensor(self.row_id_dict_uncensored[key]),
                    'group': torch.LongTensor(self.group_dict_uncensored[key]),
                    'weights': torch.FloatTensor(self.weight_dict_uncensored[key])
                }
            )
            for key in self.features_dict_uncensored.keys()
        }
        
        self.loaders_dict = {
            key: DataLoader(value, batch_size=batch_size, shuffle=True)
            for key, value in self.dataset_dict.items()
        }

        self.loaders_dict_predict = {
            key: DataLoader(value, batch_size=batch_size, shuffle=False)
            for key, value in self.dataset_dict.items()
        }
        
def logger_setup(config_dict, args):
    
    logger = logging.getLogger(__name__)
    
    if config_dict.get('logging_path') is not None:
        logging.basicConfig(
            filename=config_dict.get('logging_path'),
            level='DEBUG' if args['print_debug'] else 'INFO',
            format="%(message)s",
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level='DEBUG' if args['print_debug'] else 'INFO',
            format="%(message)s",
        )
    return logger

def model_setup(config_dict, logger, args):
    assert(config_dict['group_objective_type'] in ["standard", "lagrangian", "dro", "regularized"]), print('group_objective_type must be one of ["standard", "lagrangian", "dro", "regularized"]') 
    if config_dict['group_objective_type'] == 'standard':
        model_class = TorchModel
    elif config_dict['group_objective_type'] == 'lagrangian':
        model_class = MultiLagrangianThresholdModelLinear
    elif config_dict['group_objective_type'] == 'dro':
        model_class = GroupDROModel
    elif config_dict['group_objective_type'] == 'regularized':
        model_class = EqualThresholdRateModelLinear

    # initialize model with params
    model = model_class(**config_dict)

    # log
    logger.info("Model config dict: {}".format(model.config_dict) )
    # Write the resulting config
    yaml_write(config_dict, os.path.join(args.get('result_path'), "config.yaml"))

    return model, logger


def get_dict_subset(dictionary, keys):
    return dict([(key,dictionary[key]) for key in keys])