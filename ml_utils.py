
import os
import gc
import json
import logging
from typing import Dict, Any, Union, Tuple

import numpy as np

import xarray as xr

import pandas as pd

import torch
import torch.nn as nn

from ai4water.models import Learner
from ai4water.utils import prepare_data as to_supervised


logger = logging.getLogger(__name__)


class GlobalLSTM1(nn.Module):
    """
    LSTM receives both static and dynamic inputs at each time step
    """
    def __init__(
            self,
            input_size,
            hidden_size:int = 256,
            num_layers:int = 1,
            dropout:float = 0.4,
            output_size:int = 1
    ):
        
        super(GlobalLSTM1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        dynamic = x[0]
        static = x[1].unsqueeze(1).repeat(1, dynamic.shape[1], 1)

        x = torch.concat((dynamic, static), dim=2)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, dtype=x.dtype).to(x.device)

        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True, dtype=x.dtype).to(x.device)
  
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.dropout(out)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


MODELS = {
    "GlobalLSTM1": GlobalLSTM1,
}


# we don't want to write after each batch, that will be very long
class LearnerForInference(Learner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.eval()

    def eval(self, batch_x, batch_y):        
        """Calls the model with x and y data and returns trues and preds"""
        assert len(batch_x) == 2, f"batch_x should have 2 elements but got {len(batch_x)}"

        batch_x = [batch_x[0].float(), batch_x[1].float()]  # todo, do we need it every time?

        if self.use_cuda:
            batch_x = [batch_x[0].cuda(), batch_x[1].cuda()]
            batch_y = batch_y.cuda()

        pred_y = self.model(batch_x)

        del batch_x

        return batch_y, pred_y



def check_nans(
        input_x: np.ndarray,
        input_y: np.ndarray,
        label_y: np.ndarray,
        index: np.ndarray,
        verbosity: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checks whether anns are present or not and checks shapes of arrays
    being prepared."""

    assert input_x.shape[0] == input_y.shape[0] == label_y.shape[0], """
    shapes are not same"""

    nan_idx_y = np.isnan(label_y).any(axis=(1, 2))
    if nan_idx_y.sum() > 0:
        non_nan_idx_y = np.invert(nan_idx_y)
        if verbosity > 0:
            logger.info(f'After removing {nan_idx_y.sum()} examples with nan labels, new examples are {non_nan_idx_y.sum()}')

        label_y = label_y[non_nan_idx_y]
        input_x = input_x[non_nan_idx_y]
        input_y = input_y[non_nan_idx_y]
        index = index[non_nan_idx_y]

    nan_idx_x = np.isnan(input_x).any(axis=(1, 2))
    if nan_idx_x.sum() > 0:
        non_nan_idx_x = np.invert(nan_idx_x)
        if verbosity > 0:
            logger.info(f'After removing {nan_idx_x.sum()} examples with nan inputs, new examples are {non_nan_idx_x.sum()}')

        label_y = label_y[non_nan_idx_x]
        input_x = input_x[non_nan_idx_x]
        input_y = input_y[non_nan_idx_x]
        index = index[non_nan_idx_x]

    return input_x, input_y, label_y, index


def stn_sample_generator(
        dyn_data:pd.DataFrame,
        static_df:pd.DataFrame,
        stn:str,
        lookback:int = 365,
        subsample_x:bool = False,
        x_samples_to_keep:int = 1000,
        gamma_for_subsampling_x:float = 5.0,        
        verbosity:int = 0
):
    """
    makes x,y for one whole station at a time. 
    This is useful when we want to feed the data of one station at a time."""
    
    static_data = static_df.loc[stn].copy()
    # fill the nan with 0s
    #static_data.fillna(0.0, inplace=True)
    if static_data.isna().sum()>0:
        raise ValueError(f"static data has nan values for {static_data[static_data.isna()]}")

    input_steps = 1
    forecast_step = 0
    # save index of dataframe to keep track of the samples
    step = (lookback * input_steps) + forecast_step - input_steps
    index = dyn_data.index[step:]
    index = [stn+'_'+str(idx) for idx in index]
    ini_len = len(dyn_data)

    x, _, y = to_supervised(dyn_data, lookback, dyn_data.shape[1]-1, 1, input_steps=input_steps, forecast_step=forecast_step)

    x, _, y, index = check_nans(x, _, y, index=np.array(index), verbosity=verbosity-3)

    y = y.reshape(-1, 1)

    if verbosity>3: logger.info(f"    {stn}, samples: {ini_len} -> {x.shape[0]}")
    for i in range(len(x)):
        yield (x[i].astype('float32'), static_data.values.astype('float32')), y[i].astype('float32'), index[i]


def stn_batch_generator(generator, batch_size):
    # Function to batch the generator into fixed sizes

    x, x_s, y, indices = [], [], [], []
    for (x_, x_s_), y_, index_ in generator:

        x.append(x_)
        x_s.append(x_s_)
        y.append(y_)
        indices.append(index_)

        if len(x) == batch_size:
            #print("Yield a batch")
            x = torch.Tensor(np.array(x))
            x_s = torch.Tensor(np.array(x_s))
            y = torch.Tensor(np.array(y))
            indices = np.array(indices)            
            yield (x, x_s), y, indices
            x, x_s, y, indices = [], [], [], []  # Reset batch for next collection

    #print("Yield the last batch if it's smaller than batch_size")
    if len(x)>0:
        x = torch.Tensor(np.array(x))
        x_s = torch.Tensor(np.array(x_s))
        y = torch.Tensor(np.array(y))
        indices = np.array(indices)            
        yield (x, x_s), y, indices


def apply_scalers(
        dyn_data, 
        static_data, 
        #target,
        scalers:Dict[str, Any],
        verbosity:int = 0
        ):
    if all([key in scalers for key in ['x_dyn_mean', 'x_dyn_std']]):
        # if dyn_data is dictionary, transformation is done in place
        if isinstance(dyn_data, dict):
            for stn, dyn_da in dyn_data.items():
                dyn_data[stn] = (dyn_da - scalers['x_dyn_mean']) / scalers['x_dyn_std']
        else:
            dyn_data = (dyn_data - scalers['x_dyn_mean']) / scalers['x_dyn_std']

        if verbosity>2:
            logger.info(f"Standardized dyanmic feautres")

    if all([key in scalers for key in ['x_st_mean', 'x_st_std']]):
        static_data = (static_data - scalers['x_st_mean']) / scalers['x_st_std']

    # if all([key in scalers for key in ['y_mean', 'y_std']]):
    #     if isinstance(target, dict):
    #         for stn, target_da in target.items():
    #             target[stn] = (target_da - scalers['y_mean']) / scalers['y_std']
    #     else:
    #         target = (target - scalers['y_mean']) / scalers['y_std']

        if verbosity>2:
            logger.info(f"Standardized target features")

    return dyn_data, static_data#, target


class Inference(object):
    """

    inference = Inference(
        path='results/Arcticnet__49052/',
        best_weights='results/Arcticnet__49052/best_model.pt',
        batch_size = 2048,
    )
    predictions = inference(X_dynamic, X_static)
    """
    def __init__(
            self,
            path: Union[str, os.PathLike],
            best_weights: Union[str, os.PathLike],
            config:dict = None,
            batch_size:int = 1024,
            verbosity:int = None
    ):
        """
        path : 
            path where the outputs will be saved
        """        
        self.path = path
        self.best_weights = best_weights
        self.batch_size = batch_size

        if verbosity>2:
            logger.info(f"Using {best_weights} for inference")

        if config is None:
            config_path = os.path.join(path, 'config.json')

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found in {path}. Please make sure to provide the correct path for 'out_dir' where the config.json file is located.")

            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            assert isinstance(config, dict), "config should be a dictionary"
            assert all(key in config for key in ['model', 'model_config', 'data_config']), "config should contain keys: model, model_config, data_config"
            self.config = config.copy()

        if verbosity:
            self.verbosity = verbosity
        else:
            self.verbosity = self.config['verbosity']
        
        self.data_config = self.config['data_config']

        model = MODELS[self.config['model']](
            input_size=len(self.data_config['dynamic_features']) + len(self.data_config['static_features']),
            **self.config['model_config']
            )

        # Pick device for inference
        if torch.cuda.is_available():
            use_cuda = True
            device = "cuda"
        else:
            use_cuda = False
            device = "cpu"
        if self.verbosity>2:
            logger.info(f"Using device: {device} for inference")

        # Load the saved state_dict
        state_dict = torch.load(best_weights, map_location=device)
        if isinstance(state_dict, dict) and 'model_state' in list(state_dict.keys()):
            state_dict = state_dict['model_state']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        if self.verbosity>2: logger.info(f"   Sucessfully updated weights of model from {os.path.basename(best_weights)}")

        self.learner = LearnerForInference(
            model=model,
            use_cuda=use_cuda,
            verbosity=self.verbosity,
            path=path,
            )

    def load_scalers(self):
        scalers_file = os.path.join(self.path, 'scalers.json')
        if not os.path.exists(scalers_file):
            raise FileNotFoundError(f"Scalers file not found in {self.path}. Please make sure to provide the correct path for 'out_dir' where the scalers.json file is located.")
        with open(scalers_file, 'r') as f:
            scalers = json.load(f)

        scalers['x_st_mean'] = pd.Series(scalers['x_st_mean'])
        scalers['x_st_std'] = pd.Series(scalers['x_st_std'])

        x_dyn_mean = pd.Series(scalers['x_dyn_mean'])
        # consert pd.Series to xarray DataArray
        scalers['x_dyn_mean'] = xr.DataArray(x_dyn_mean.values, 
                                             dims=['dynamic_features'], 
                                             coords={'dynamic_features': x_dyn_mean.index})
        x_dyn_std = pd.Series(scalers['x_dyn_std'])
        scalers['x_dyn_std'] = xr.DataArray(x_dyn_std.values, 
                                            dims=['dynamic_features'], 
                                            coords={'dynamic_features': x_dyn_std.index})
        
        scalers['y_mean'] = np.float32(scalers['y_mean'])
        scalers['y_std'] = np.float32(scalers['y_std'])
        return scalers

    def __call__(
            self, 
            X_dynamic:pd.DataFrame,
            X_static:pd.DataFrame,
            ):

        scalers = self.load_scalers()

        assert len(X_dynamic.columns) == len(self.data_config['dynamic_features'])
        for col in X_dynamic.columns:
            assert col in self.data_config['dynamic_features'], f"{col} is not in the list of dynamic features in config"

        # convert dynamic into DataArray with dimensions time and dynamic_features
        dynamic = xr.DataArray(X_dynamic.values, 
                               dims=['time', 'dynamic_features'], 
                               coords={'time': X_dynamic.index, 'dynamic_features': X_dynamic.columns})

        assert isinstance(X_static, pd.DataFrame)
        X_static = X_static[self.data_config['static_features']]
        assert len(X_static.columns) == len(self.data_config['static_features'])
        for col in X_static.columns:
            assert col in self.data_config['static_features'], f"{col} is not in the list of static features in config"

        dyn_X, static_df = apply_scalers(dynamic, 
                                          X_static, 
                                          scalers=scalers, 
                                          verbosity=self.verbosity-1)

        if self.verbosity>3:
            logger.info(f" Making Inference on Data")

        return self.make_inference(dyn_X, static_df, scalers)

    def make_inference(
            self,
            dyn_ds:xr.Dataset,
            static_df: pd.DataFrame,
            scalers:dict,
            )->pd.DataFrame:
        """
        makes inference on dictionary whose keys are station names and values are 
        xarray datasets.
        
        It returns a DataFrame whose index is names of stations and column names are 
        names of performance metrics.
        """

        dyn_x = dyn_ds.to_pandas()
        dyn_y = pd.Series(np.full(dyn_x.shape[0], 0), index=dyn_x.index, name='target')  # dummy y, not used in inference but needed for generator
        dyn_data = pd.concat([dyn_x, dyn_y], axis=1)

        kwargs = dict(
            dyn_data=dyn_data,
            static_df=static_df,
            stn='-9999',
            lookback=self.data_config['lookback'],
            verbosity=self.verbosity,
            subsample_x = True if self.data_config.get('timestep', 'D')=='H' else False,
        )

        gc.collect()
        if self.learner.use_cuda:
            torch.cuda.empty_cache()

        xd_stn, y, indices, pred_stn = [], [], [], []

        gen = stn_sample_generator(**kwargs)

        # iterate over the generator and get x, y and indices, and save them in a list to make a batch == eval_batch_size

        for batch_idx, (x_, y_, indices_) in enumerate(stn_batch_generator(gen, self.batch_size)):

            batch_xd, _ = x_

            pred_y_ = self.learner.predict(list(x_), y_)

            xd_stn.append(batch_xd)
            y.append(y_)
            indices.append(indices_)
            pred_stn.append(pred_y_)

        # check if it is a new station
        df = pd.DataFrame()
        if len(y) >0:

            if self.verbosity>3:
                logger.info(f"  Inference Done after {batch_idx+1} batches! Processing results")

            # index is StnID_1969-12-13 00:00:00 -> 1969-12-13 00:00:00
            index = pd.to_datetime(np.array([s.rpartition("_")[2] for sub in indices for s in sub]))

            obs_stn = np.concatenate([val.cpu().numpy().reshape(-1, ) for val in y])

            pred_stn = np.concatenate([val.reshape(-1, ) for val in pred_stn])

            # put back the data to original scale
            obs_stn = (obs_stn * scalers['y_std']) + scalers['y_mean']
            pred_stn = (pred_stn * scalers['y_std']) + scalers['y_mean']

            df = pd.DataFrame({'true': obs_stn, 'pred': pred_stn}, index=index)  

        return df


def run_lstm(forcing_data, lstm_inference, lstm_static, warmup_days=365):
    """Run LSTM inference and return qsim series (after warmup)."""
    dyn_features = lstm_inference.data_config['dynamic_features']  # ['pet', 'pcp', 'tmax', 'tmin', 'temp']
    X_dynamic = forcing_data[dyn_features]  # reorder columns to match training order
    result_df = lstm_inference(X_dynamic, lstm_static)  # returns DataFrame with 'true' and 'pred'

    qsim = result_df['pred']
    # Streamflow is physically non-negative; the LSTM head is unconstrained and
    # can emit small negative values, so clip them to 0 before any flow stats /
    # dry-spell thresholds are computed (applied to baseline and future alike).
    qsim = qsim.clip(lower=0.0)
    # The LSTM lookback already consumes initial timesteps, but align with HBV warmup
    # Trim to match HBV output time range
    warmup_end = forcing_data.index[warmup_days]
    qsim = qsim.loc[qsim.index >= warmup_end]
    return qsim
