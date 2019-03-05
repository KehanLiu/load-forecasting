import abc  # abstract base class
import os.path, sys
from numpy.lib.stride_tricks import as_strided
import datetime as dt
import numpy as np
import pandas as pd
import math
from keras.models import Sequential, load_model
from keras.models import Model as Functional_model
from keras.layers import Dense, Dropout, Convolution1D, Flatten, Input, merge, Activation
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping
import tensorflow as tf
from keras import backend as K

import utils
from metrics import mean_neg_log_loss_parametric, mean_neg_log_loss_discrete
import config
from ann_abstr import Ann_model

# tf.python.control_flow_ops = tf  # bug fix for tensorflow dropout





class Discrete_feedforward(Ann_model):
    """
	Implementation of Softmax Distribution Network
	"""
    prefix = 'disc_ff_'

    def __init__(self,
                 model_identifier,
                 granularity_s,
                 forecast_horizon_mins,
                 look_back_mins,
                 hidden_neurons,
                 working_directory=config.working_directory,
                 trained_models_folder=config.trained_models_folder,
                 dropout=0,
                 scaling_factor=5000.0,
                 forecast_type='watthours',
                 learning_rate=0.1,
                 use_cal_vars=False,
                 activation='sigmoid',
                 pdf_sample_points_min=0.0,
                 pdf_sample_points_max=7000.0,
                 pdf_resolution=200.0):
        """ init general attributes required for Ann_model """
        self.model_identifier = model_identifier
        self.granularity_s = granularity_s
        self.forecast_horizon_mins = forecast_horizon_mins
        self.look_back_mins = look_back_mins
        self.hidden_neurons = hidden_neurons
        self.working_directory = working_directory
        self.trained_models_folder = trained_models_folder
        self.dropout = dropout
        self.scaling_factor = scaling_factor
        self.forecast_type = forecast_type
        self.learning_rate = learning_rate
        self.use_cal_vars = use_cal_vars
        self.activation = activation

        self.loss_func = mean_neg_log_loss_discrete  # 'categorical_crossentropy' #

        """ init model-specific attributes """
        self.pdf_sample_points_min = pdf_sample_points_min
        self.pdf_sample_points_max = pdf_sample_points_max
        self.pdf_resolution = pdf_resolution
        self.pdf_granularity = (pdf_sample_points_max - pdf_sample_points_min) / pdf_resolution
        self.pdf_sample_points = np.linspace(pdf_sample_points_min, pdf_sample_points_max, pdf_resolution)

        self.init_model()

    """	implement abstract methods """

    def generate_model(self):
        """
		"""

        if self.use_cal_vars:
            self.nb_input_neurons += 4  # minute, hour of day, day of week, month

        # print "Size of input layer:", self.nb_input_neurons
        input_layer = Input(shape=(self.nb_input_neurons,))

        x = Dense(self.hidden_neurons[0])(input_layer)
        x = Dropout(self.dropout)(x)

        for hn in self.hidden_neurons[1:]:
            x = Dense(hn, activation=self.activation)(x)
            x = Dropout(self.dropout)(x)

        pdf = Dense(len(self.pdf_sample_points), activation='softmax')(x)

        model = Functional_model(input=input_layer, output=pdf)

        return model

    def write_training_log(self, history):
        """
		"""
        with open(os.path.join(self.working_directory, 'training_log.csv'), 'a') as log_file:
            log_file.write(self.model_name + ',' + \
                           str(self.model_identifier) + ',' + \
                           str(self.granularity_s) + ',' + \
                           str(self.forecast_horizon_mins) + ',' + \
                           str(self.look_back_mins) + ',' + \
                           str(self.hidden_neurons) + ',' + \
                           str(self.forecast_type) + ',' + \
                           str(history.history['loss'][-1]) + ',' + \
                           str(history.history['val_loss'][-1]) + '\n')

    def generate_input_data(self, lagged_vals, t0):
        """
		"""
        X = self.scale(lagged_vals)

        if self.use_cal_vars:
            minutes = t0.minute
            minutes = minutes.reshape((len(t0), 1)) / 60.0  # scale to [0,1]
            hours = t0.hour
            hours = hours.reshape((len(t0), 1)) / 24.0
            day = t0.weekday
            day = day.reshape((len(t0), 1)) / 7.0
            month = t0.month
            month = month.reshape((len(t0), 1)) / 12.0
            if self.activation == 'tanh':
                minutes = minutes * 2.0 - 1  # scale to [-1,1]
                hours = hours * 2.0 - 1
                day = day * 2.0 - 1
                month = month * 2.0 - 1
            X = np.hstack((X, minutes, hours, day, month))
        return X

    def generate_output_data(self, ground_truth):
        """
		"""
        nb_sample_points = len(self.pdf_sample_points)
        batch_size = len(ground_truth)
        y = np.zeros((batch_size, nb_sample_points))

        pdf_sample_points_grid = self.pdf_sample_points.reshape((1, nb_sample_points))
        pdf_sample_points_grid = np.repeat(pdf_sample_points_grid, batch_size, axis=0)

        ground_truth_grid = ground_truth.reshape((batch_size, 1))
        ground_truth_grid = np.repeat(ground_truth_grid, nb_sample_points, axis=1)

        rows_idx = np.arange(0, batch_size)
        cols_idx = np.argmin(np.abs(pdf_sample_points_grid - ground_truth_grid), axis=1)
        y[rows_idx, cols_idx] = 1.0

        return y

    def predict_on_preprocessed_input(self, X):
        """
		"""
        if X.shape[1] != self.nb_input_neurons:
            print(dt.datetime.now().strftime('%x %X') + ' Dim 1 of X does not match number of input neuros.')
            return

        y_pred = self.model.predict(X) / self.pdf_granularity

        batch_size = y_pred.shape[0]
        nb_sample_points = len(self.pdf_sample_points)

        pdf_sp_grid = np.repeat(self.pdf_sample_points.reshape((1, nb_sample_points)), batch_size, axis=0)
        return (pdf_sp_grid, y_pred)

    def generate_model_name(self):
        """
		"""
        name = self.prefix + str(self.model_identifier) + \
               '_' + str(self.forecast_type) + \
               '_granu' + str(self.granularity_s) + \
               '_hor' + str(self.forecast_horizon_mins) + \
               '_lb' + str(self.look_back_mins) + \
               '_drop' + str(self.dropout) + \
               '_pdflen' + str(len(self.pdf_sample_points)) + \
               '_scale' + str(self.scaling_factor) + '_' + self.activation
        if self.use_cal_vars:
            name += '_cal'

        name += '_lay'
        for hn in self.hidden_neurons:
            name = name + str(hn) + '-'
        name = name[:-1]
        return name[:249]  # limit length of name for ntfs file system

