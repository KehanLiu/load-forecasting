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
from ann_forecast.ann_abstr import Ann_model

tf.python.control_flow_ops = tf  # bug fix for tensorflow dropout





class Mdn_feedforward(Ann_model):
    """
	Mixture density network implementation
	"""
    prefix = 'mdn_ff_'

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
                 learning_rate=0.0001,
                 use_cal_vars=False,
                 activation='sigmoid',
                 nb_kernels=5):
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

        self.loss_func = mean_neg_log_loss_parametric

        """init model-specific attributes"""
        self.nb_kernels = nb_kernels

        self.init_model()

    """implement abstract methods"""

    def generate_model(self):
        """
		Builds the network specified by the architecture
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

        mixing_coeffs = Dense(self.nb_kernels, activation='softmax')(x)
        centroids = Dense(self.nb_kernels, activation='linear')(x)
        std_devs = Dense(self.nb_kernels, activation='softplus')(x)

        output_layer = merge([mixing_coeffs, centroids, std_devs], mode='concat', concat_axis=1)

        model = Functional_model(input=input_layer, output=output_layer)

        return model

    def write_training_log(self, history):
        """
		With history as keras training history object
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
		Generates the model input matrices from the lagged values and the timestamps
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
		Generates the outputs y.
		"""
        return self.scale(ground_truth)



    def predict_on_preprocessed_input(self, X):
        """
		"""

        # if X.shape[1] != self.nb_input_neurons:
        #     print(dt.datetime.now().strftime('%x %X') + ' Dim 1 of X does not match number of input neuros.')
        #     return

        y_pred = self.model.predict(X)

        nb_kernels = int(y_pred.shape[1] / 3)

        coeffs = y_pred[:, 0: nb_kernels]
        centroids = y_pred[:, nb_kernels: 2 * nb_kernels]
        std_devs = y_pred[:, 2 * nb_kernels: 3 * nb_kernels]

        return self.reverse_scale_kernels(coeffs, std_devs, centroids)

    def generate_model_name(self):
        """
		"""
        name = self.prefix + str(self.model_identifier) + \
               '_' + str(self.forecast_type) + \
               '_granu' + str(self.granularity_s) + \
               '_hor' + str(self.forecast_horizon_mins) + \
               '_lb' + str(self.look_back_mins) + \
               '_drop' + str(self.dropout) + \
               '_ker' + str(self.nb_kernels) + \
               '_scale' + str(self.scaling_factor) + '_' + self.activation
        if self.use_cal_vars:
            name += '_cal'

        name += '_lay'
        for hn in self.hidden_neurons:
            name = name + str(hn) + '-'
        name = name[:-1]
        return name[:249]  # limit length of name for ntfs file system

    """ implement model-specific methods """

    def reverse_scale_kernels(self, coeffs, std_devs, centroids):
        """
		Reverses scale of standard deviations and centroids. Coeffs stay untouched as they need to sum up to 1.
		"""
        return (coeffs, self.reverse_scale(std_devs), self.reverse_scale(centroids))