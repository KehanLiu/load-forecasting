import abc  # abstract base class
import os.path
import sys
import datetime as dt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import math
from keras.models import Sequential, load_model
from keras.models import Model as Functional_model
from keras.layers import Dense, Dropout, Convolution1D, Flatten, Input, merge, Activation
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras import metrics
import utils.utils as utils
import config as config
from keras.layers import LSTM
from ann_abstr import Ann_model
from metrics import mean_neg_log_loss_parametric, mean_neg_log_loss_discrete



class Prob_LSTM(Ann_model):
    """
    LSTM neural network definition
    Implementation with Softmax Distribution Network
    """
    prefix = 'lstm_'
    def __init__(self,
                 model_identifier,
                 granularity_s,
                 forecast_horizon_mins,
                 look_back_mins,
                 hidden_neurons,
                 working_directory=config.working_directory,
                 trained_models_folder=config.trained_models_folder,
                 dropout=0,
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
        Generate the neural network model
        Define the model's architecture and the implemented functions
        """
        # Size of input layer
        # -------------------
        # LSTMs expect a 3-dim input of the form [samples, timesteps, features]
        if self.use_cal_vars:
            input_layer = Input(shape=(self.nb_input_neurons, 5))
        else:
            input_layer = Input(shape=(self.nb_input_neurons, 1))
        # input_layer = Input(shape=(1, self.nb_input_neurons)) # TODO Dimension???!!

        # Number of hidden layers
        nb_layers = np.array(self.hidden_neurons).shape[0]
        if nb_layers > 1:
            x = LSTM(self.hidden_neurons[0], return_sequences=True)(input_layer)
            x = Dropout(self.dropout)(x) # dropout layer to prevent overfitting
        else:
            x = LSTM(self.hidden_neurons[0])(input_layer)
            x = Dropout(self.dropout)(x)
        iter_temp = 1
        for hn in self.hidden_neurons[1:]:
            if iter_temp == len(self.hidden_neurons) - 1:
                x = LSTM(hn)(x)
            else:
                # if some hidden layers have to be added return sequence
                x = LSTM(hn, return_sequences=True)(x)
            iter_temp = iter_temp + 1
            x = Dropout(self.dropout)(x)

        # Output layer is a pdf function with all power "bins", see theory
        pdf = Dense(len(self.pdf_sample_points), activation='softmax')(x)  # previous layers (x) are stacked
        model = Functional_model(input=input_layer, output=pdf) # LSTM model definition
        return model

    def write_training_log(self, history):
        """
        Write file with training's results
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
        Prepare (normalize) input data for forecasting
        """
        X = self.scale(lagged_vals)

        X = X.reshape((X.shape[0], X.shape[1], 1))

        if self.use_cal_vars:

            minutes = t0.minute
            # Normalized values
            minutes = minutes / 60.0
            hours = t0.hour
            hours = hours / 24.0
            day = t0.weekday
            day = day / 7.0
            month = t0.month
            month = month / 12.0

            minsaux = np.zeros(X.shape)
            hoursaux = np.zeros(X.shape)
            daysaux = np.zeros(X.shape)
            monthsaux = np.zeros(X.shape)

            for i_sample in range(len(t0)):
                for i_timestamp in range(lagged_vals.shape[1]):
                    minsaux[i_sample][i_timestamp][0] = minutes[i_sample]
                    hoursaux[i_sample][i_timestamp][0] = (hours[i_sample])
                    daysaux[i_sample][i_timestamp][0] = (day[i_sample])
                    monthsaux[i_sample][i_timestamp][0] = (month[i_sample])

            minutes = minsaux
            hours = hoursaux
            day = daysaux
            month = monthsaux

            if self.activation == 'tanh':
                minutes = minutes * 2.0 - 1  # scale to [-1,1]
                hours = hours * 2.0 - 1
                day = day * 2.0 - 1
                month = month * 2.0 - 1

            X = np.concatenate((X, minutes, hours, day, month), axis=2)

        return X

    def generate_output_data(self, ground_truth):
        """
        Generates an output of type [0,0,...,Pt,...,0,0] to be compared against the pdf output from the model
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
        Employ model to predict X values
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
        Generate the model's file name with its main characteristics
        """
        name = self.prefix + str(self.model_identifier) + \
               '_' + str(self.forecast_type) + \
               '_granu' + str(self.granularity_s) + \
               '_hor' + str(self.forecast_horizon_mins) + \
               '_lb' + str(self.look_back_mins) + \
               '_drop' + str(self.dropout) + \
               '_pdflen' + str(len(self.pdf_sample_points)) + \
               '_' + self.activation
        if self.use_cal_vars:
            name += '_cal'

        name += '_lay'
        for hn in self.hidden_neurons:
            name = name + str(hn) + '-'
        name = name[:-1]
        return name[:249]  # limit length of name for ntfs file system