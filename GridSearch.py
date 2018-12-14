# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:11:14 2018

@author: bfe-kli
"""

'''
import start
'''
import abc  # abstract base class
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path
import sys
sys.path.append('C:\\Users\\leoliu\\Anaconda3\\envs\\python3.6\\lib\\site-packages\\tensorflow')
sys.path.append(os.path.realpath(os.path.dirname(os.path.realpath(__file__))))
#print(sys.path)
import datetime as dt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import math
import keras
from keras.models import Sequential, load_model
from keras.models import Model as Functional_model
from keras.layers import Dense, Dropout, Convolution1D, Flatten, Input, merge, Activation
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras import layers
from keras import optimizers
from keras.wrappers import scikit_learn
from sklearn.model_selection import GridSearchCV
import utils.utils as utils
import config as config
from keras.layers import LSTM
from ann_abstr import Ann_model
from metrics import mean_neg_log_loss_parametric, mean_neg_log_loss_discrete
from lstm import Prob_LSTM
import matplotlib.pyplot as plt
'''
import end
'''
prefix = 'lstm_'
model_identifier='DHW_3'
dataset_identifier='DHW_3'
forecast_horizon_mins = 5
# forecast_horizon 1 day or 5 mins
granularity_s = 300
#forecast for every 5 mins
look_back_mins = 60
sliding_window_width = int(dt.timedelta(minutes=look_back_mins).total_seconds() / granularity_s)
nb_input_neurons = sliding_window_width
working_directory=config.working_directory
trained_models_folder=config.trained_models_folder
preprocessed_datasets_folder=config.preprocessed_datasets_folder

forecast_type='watthours'

use_cal_vars=False

pdf_sample_points_min=0.0
pdf_sample_points_max=1.0
pdf_resolution=50.0

loss_func = mean_neg_log_loss_discrete
pdf_granularity = (pdf_sample_points_max - pdf_sample_points_min) / pdf_resolution
pdf_sample_points = np.linspace(pdf_sample_points_min, pdf_sample_points_max, pdf_resolution)

train_cv_test_split=(0.8, 0.1, 0.1)
cleanse=False

activation='sigmoid'

keras.backend.clear_session()
'''
batch_size = 8
epochs = 1
#dropout = 0.5
learning_rate = 0.01
nb_hidden_layers = 3
nb_hidden_neurons = 50
'''

def generate_dataset_filename(dataset_identifier):
    """
	Generate the dataset's filename corresponding to the model's granularity
	"""
    return utils.generate_dataset_filename(dataset_identifier, granularity_s)

def open_dataset_file(filename, preprocessed_datasets_folder=config.preprocessed_datasets_folder):
    """
	"""
    return utils.open_dataset_file(filename, working_directory, preprocessed_datasets_folder)

def scale(values, scaling_factor):
    """
	Scales the values by the model's scaling_factor
	"""
    return values / (scaling_factor * 1.0)

def generate_input_data(lagged_vals, t0, scaling_factor):
    """
    Prepare (normalize) input data for forecasting
    """
    X = scale(lagged_vals, scaling_factor)
    Xtemp = X.reshape((X.shape[0], X.shape[1], 1))
    X = Xtemp[:-sliding_window_width][:][:]

    if use_cal_vars:

        minutes = t0.minute
        # Normalized values
        minutes = minutes / 60.0
        hours = t0.hour
        hours = hours / 24.0
        day = t0.weekday
        day = day / 7.0
        month = t0.month
        month = month / 12.0
    
        minsaux = np.zeros(Xtemp.shape)
        hoursaux = np.zeros(Xtemp.shape)
        daysaux = np.zeros(Xtemp.shape)
        monthsaux = np.zeros(Xtemp.shape)
    
        for i_sample in range(len(t0)-1):
            for i_timestamp in range(lagged_vals.shape[1]):
                i_timestamp_total = i_timestamp + i_sample
                if i_timestamp_total > len(t0)-1:
                    minsaux[i_sample][i_timestamp][0] = 0
                    hoursaux[i_sample][i_timestamp][0] = 0
                    daysaux[i_sample][i_timestamp][0] = 0
                    monthsaux[i_sample][i_timestamp][0] = 0
                else:
                    minsaux[i_sample][i_timestamp][0] = minutes[i_timestamp_total]
                    hoursaux[i_sample][i_timestamp][0] = (hours[i_timestamp_total])
                    daysaux[i_sample][i_timestamp][0] = (day[i_timestamp_total])
                    monthsaux[i_sample][i_timestamp][0] = (month[i_timestamp_total])
    
        minutes = minsaux[:-sliding_window_width][:][:]
        hours = hoursaux[:-sliding_window_width][:][:]
        day = daysaux[:-sliding_window_width][:][:]
        month = monthsaux[:-sliding_window_width][:][:]
    
        if activation == 'tanh':
            minutes = minutes * 2.0 - 1  # scale to [-1,1]
            hours = hours * 2.0 - 1
            day = day * 2.0 - 1
            month = month * 2.0 - 1
    
        X = np.concatenate((X, minutes, hours, day, month), axis=2)
        
    return X

def generate_output_data(ground_truth):
        """
        Generates an output of type [0,0,...,Pt,...,0,0] to be compared against the pdf output from the model
        """
        nb_sample_points = len(pdf_sample_points)
        batch_size = len(ground_truth)
        y = np.zeros((batch_size, nb_sample_points))
        
        pdf_sample_points_grid = pdf_sample_points.reshape((1, nb_sample_points))
        pdf_sample_points_grid = np.repeat(pdf_sample_points_grid, batch_size, axis=0)

        ground_truth_grid = ground_truth.reshape((batch_size, 1))
        ground_truth_grid = np.repeat(ground_truth_grid, nb_sample_points, axis=1)

        rows_idx = np.arange(0, batch_size)
        cols_idx = np.argmin(np.abs(pdf_sample_points_grid - ground_truth_grid), axis=1)
        y[rows_idx, cols_idx] = 1.0
        y = y[:-sliding_window_width]

        return y

def generate_training_data_lstm(dataset, train_cv_test_split=train_cv_test_split, cleanse=False):
    """
	From Pandas DataFrame (timestep, watthour) generate all valid training examples and split respectivly; X and y
	are scaled to model scale.
	"""
    # load raw data, df for dataframe
    ds = dataset.copy(deep=True)
    ds.watthour = np.nan_to_num(ds.watthour.values)
    # 5mins load (288 loads for each day)

    scaling_factor =np.max(np.array(ds.watthour))

    nb_forecast_steps = int(dt.timedelta(minutes=forecast_horizon_mins).total_seconds() / granularity_s)

    nb_examples = len(np.array(ds.watthour)) - nb_forecast_steps - sliding_window_width

    #history_offset = sliding_window_width + nb_forecast_steps - 1

    #lagged_vals = np.array(list_5mins_load[-history_offset:])

    s = np.array(ds.watthour).itemsize

    lagged_vals = as_strided(np.array(ds.watthour), shape=(nb_examples, sliding_window_width), strides=(s, s))

    if sliding_window_width != 0:
            t0 = ds.index[sliding_window_width - 1:-nb_forecast_steps - 1]
    else:
            t0 = ds.index[:-nb_forecast_steps]

    if forecast_type == 'watthours':
        s = np.array(ds.watthour).itemsize
        watthour_intervals = as_strided(np.array(ds.watthour)[sliding_window_width:], shape=(nb_examples, nb_forecast_steps), strides=(s, s))
                # find the latest column of lagged value
                # print "fc_hor", watthour_intervals[2]
                # print "last_elem_of_sum", watthour_intervals[:,-1]
                # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].watthour.values
        mask = np.unique(np.where(watthour_intervals != 0.0)[0])  # np.where returns indices for nonzero values as [xi][yi]; take only unique row indices
        #?????找出watthour_intervals中不是0的数的行数
        ground_truth = np.sum(watthour_intervals,axis=-1)  # integrate watthour over forecast horizon to get total energy in Wh
        #?????    
        # print y.shape
    elif forecast_type == 'watts':
        ground_truth = ds.watthour.values[sliding_window_width + nb_forecast_steps - 1:-1]
                #mask = np.array(np.where(y != 0.0)).reshape((-1))
            # print 'gt', ground_truth
            # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].watthour.values
    else:
        print('Unsupported forecast type. Please define forecast type as either \'watts\' or \'watthours\'.')

            # print(">>ground truth", ground_truth)
            # ground truth real values
    ground_truth = ground_truth.reshape(-1, 1)

    ground_truth = (ground_truth/scaling_factor)
    #????????为什么这里ground_truth没有大于1的？----因为前面*granularity_s/3600,另外这里跟源代码不一样

    # generate_input_data
    X = generate_input_data(lagged_vals, t0, scaling_factor)

    # y is an vector with normalized energy consumption within the time interval
    y = generate_output_data(ground_truth)
    
    # print(">>ground truth", ground_truth)
    # print(">>output", y)

    if cleanse:
        ground_truth = ground_truth[mask]
        y = y[mask]  # cleansing the data leads to extreme performance drop
        X = X[mask, :]
        t0 = t0[mask]

    val_idx = int(len(y) * train_cv_test_split[0])
    test_idx = int(len(y) * (train_cv_test_split[0] + train_cv_test_split[1]))

    y_train = y[0:val_idx]
    X_train = X[0:val_idx, :]
    ground_truth_train = ground_truth[0:val_idx]
    t0_train = t0[0:val_idx]

    y_val = y[val_idx:test_idx]
    t0_val = t0[val_idx:test_idx]
    ground_truth_val = ground_truth[val_idx:test_idx]
    X_val = X[val_idx:test_idx, :]

    y_test = y[test_idx:]
    t0_test = t0[test_idx:]
    ground_truth_test = ground_truth[test_idx:]
    X_test = X[test_idx:, :]

    return (X_train, y_train, ground_truth_train, t0_train), \
           (X_val, y_val, ground_truth_val, t0_val), \
           (X_test, y_test, ground_truth_test, t0_test), \
           scaling_factor

def generate_model(dropout_rate, learning_rate, nb_hidden_layers, nb_hidden_neurons):
    """
    Generate the neural network model
    Define the model's architecture and the implemented functions
    """
    hidden_neurons = []
    for i in range(nb_hidden_layers):
        hidden_neurons.append(nb_hidden_neurons)
    
    # Size of input layer
    # -------------------
    # LSTMs expect a 3-dim input of the form [samples, timesteps, features]
    if use_cal_vars:
        input_layer = Input(shape=(nb_input_neurons, 5))
    else:
        input_layer = Input(shape=(nb_input_neurons, 1))
    # input_layer = Input(shape=(1, self.nb_input_neurons)) # TODO Dimension???!!

    # Number of hidden layers
    nb_layers = np.array(hidden_neurons).shape[0]
    if nb_layers > 1:
        x = LSTM(hidden_neurons[0], return_sequences=True)(input_layer)
        x = Dropout(dropout_rate)(x) # dropout layer to prevent overfitting
    else:
        x = LSTM(hidden_neurons[0])(input_layer)
        x = Dropout(dropout_rate)(x)
    iter_temp = 1
    for hn in hidden_neurons[1:]:
        if iter_temp == len(hidden_neurons) - 1:
            x = LSTM(hn)(x)
        else:
            # if some hidden layers have to be added return sequence
            x = LSTM(hn, return_sequences=True)(x)
        iter_temp = iter_temp + 1
        x = Dropout(dropout_rate)(x)

    # Output layer is a pdf function with all power "bins", see theory
    pdf = Dense(len(pdf_sample_points), activation='softmax')(x)  # previous layers (x) are stacked
    model = Functional_model(input=input_layer, output=pdf) # LSTM model definition
   
    model.compile(loss=loss_func, optimizer='adam', metrics=['accuracy'])
    
    return model




filename = generate_dataset_filename(dataset_identifier=dataset_identifier)

dataset = open_dataset_file(filename, preprocessed_datasets_folder)

(X_train, y_train, ground_truth_train, t0_train), \
           (X_val, y_val, ground_truth_val, t0_val), \
           (X_test, y_test, ground_truth_test, t0_test), \
           scaling_factor = generate_training_data_lstm(dataset, train_cv_test_split=train_cv_test_split, cleanse=False)

# 设置种子，为了可复现（这个无关紧要）
seed = 7
np.random.seed(seed)

model = scikit_learn.KerasClassifier(build_fn=generate_model, verbose=1, epochs = 100)

batch_size = [64, 128]
#epochs = [50, 100, 200]
dropout_rate = [0.5, 0.7]
learning_rate = [0.01, 0.1]
nb_hidden_layers = [1, 2]
nb_hidden_neurons = [40, 50]

param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, learning_rate=learning_rate, nb_hidden_layers=nb_hidden_layers, nb_hidden_neurons=nb_hidden_neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, scoring='neg_log_loss', cv=3)
'''
grid = GridSearchCV(estimator=model, 
                    param_grid={'batch_size':[8, 16, 32, 64, 128, 256],
                                'epochs':[50, 100, 200],
                                'dropout_rate':[0.3, 0.5, 0.7],
                                'learning_rate':[0.001, 0.01, 0.1, 1],
                                'nb_hidden_layers':[1, 2, 3],
                                'nb_hidden_neurons':[30, 40, 50]}, 
                    scoring='neg_log_loss',
                    n_jobs=1)
'''
grid_result = grid.fit(X_train, y_train)

print('Best: {} using {}'.format(grid_result.best_score_, grid_result.best_params_)) 
means = grid_result.cv_results_['mean_test_score'] 
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params'] 
for mean, std, param in zip(means, stds, params): 
    print("%f (%f) with: %r" % (mean, std, param))
