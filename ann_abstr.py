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

#tf.python.control_flow_ops = tf  # bug fix for tensorflow dropout


class Ann_model(object):
    """
	Abstract base class for artificial neural network models that declares the methods to be implemented and implements shared methods. Do not instantiate. 
	"""
    __metaclass__ = abc.ABCMeta
    """
	TODO
	declare abstract attributes
	"""

    @abc.abstractmethod
    def generate_model(self):
        raise NotImplementedError('Please implement generate_model() to use Ann_model')

    @abc.abstractmethod
    def write_training_log(self, history):
        raise NotImplementedError('Please implement write_training_log() to use Ann_model')

    @abc.abstractmethod
    def predict_on_preprocessed_input(self, X):
        raise NotImplementedError('Please implement predict_on_preprocessed_input() to use Ann_model')

    @abc.abstractmethod
    def generate_model_name(self):
        raise NotImplementedError('Please implement generate_model_name() to use Ann_model')

    @abc.abstractmethod
    def generate_input_data(self, lagged_vals, t0):
        raise NotImplementedError('Please implement generate_input_data() to use Ann_model')

    @abc.abstractmethod
    def generate_output_data(self, ground_truth):
        raise NotImplementedError('Please implement generate_output_data() to use Ann_model')

    def init_model(self):
        """
		Initializes and compiles model
		"""
        self.sliding_window_width = int(dt.timedelta(minutes=self.look_back_mins).total_seconds() / self.granularity_s)

        self.nb_input_neurons = self.sliding_window_width


        self.model_name = self.generate_model_name()
        # self.model_directory = os.path.join(self.working_directory, self.trained_models_folder, self.model_name + '/')

        # print(dt.datetime.now().strftime('%x %X') + ' Creating model...')
        self.model = self.generate_model()
        self.init_weights()

        # if not os.path.exists(self.model_directory):
        #     os.makedirs(self.model_directory)

        # print(dt.datetime.now().strftime('%x %X') + ' Compiling model...')
        optimizer = SGD(lr=self.learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        self.model.compile(loss=self.loss_func, optimizer='adam')

    # from keras.utils.visualize_util import plot
    # plot(self.model, to_file=os.path.join(self.model_directory,'architecture.png'))


    def init_weights(self):
        """
		Tries to restore previously saved model weights.
		"""
        try:
            self.model.load_weights(os.path.join(self.model_directory, self.model_name + '.h5'))
            print(dt.datetime.now().strftime('%x %X') + ' Model ' + self.model_name + ' successfully restored.')
        #except Exception, e:
        except Exception as e:
            print(dt.datetime.now().strftime(
                '%x %X') + ' No saved model found for ' + self.model_name + ' in ' + self.working_directory + '. Initializing new model.')


    def train_on_data_file(self,
                           dataset_identifier,
                           preprocessed_datasets_folder=config.preprocessed_datasets_folder,
                           batch_size=1000,
                           nb_epoch=10,
                           verbose=0):
        """
		Runs training using a preprocessed dataset file.
		Dataset_identifier e.g. 'ukdale_1', the prefix from config.py without the granularity_s suffix
		"""
        filename = self.generate_dataset_filename(dataset_identifier)

        dataset = self.open_dataset_file(filename, preprocessed_datasets_folder)

        if dataset.empty:
            return

        X_train, y_train, _t0_train, X_val, y_val, _t0_val, _X_test, _y_test, _t0_test = self.generate_training_data(
            dataset)

        self.train(X_train, y_train, X_val, y_val, batch_size, nb_epoch, verbose)


    def train(self,
              X_train,
              y_train,
              validation_split=0.0,
              X_val=None,
              y_val=None,
              batch_size=2000,
              nb_epoch=10,
              verbose=0,
              patience=50):
        """
		Train the model where X are model inputs (nb_examples, nb_inputs) and y are outputs (nb_examples,1)
		"""
        # csv_logger = CSVLogger(os.path.join(self.model_directory, 'log.csv'))
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=patience,
                                       verbose=verbose)

        if X_val is not None:
            history = self.model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                                     validation_split=validation_split, validation_data=(X_val, y_val),
                                     callbacks=[csv_logger, early_stopping], verbose=verbose)
        else:
            history = self.model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size,
                                     validation_split=validation_split,
                                     verbose=verbose)

        # self.write_training_log(history)

        print('Model has been trained successfully.')
        # self.model.save_weights(self.model_directory + self.model_name + '.h5')


    def predict_all(self, dataset):
        """
		Predict for all valid time steps in dataset; dataset as Pandas DataFrame (timestamp, power)
		"""
        ds = dataset.copy(deep=True)

        ds.power = np.nan_to_num(ds.power.values)

        nb_predictions = len(ds.power.values) - self.sliding_window_width

        s = ds.power.values.itemsize
        lagged_vals = as_strided(ds.power.values, shape=(nb_predictions, self.sliding_window_width), strides=(s, s))

        t0 = ds.index[self.sliding_window_width - 1:-1]

        X = self.generate_input_data(lagged_vals, t0)

        return self.predict_on_preprocessed_input(X), t0, t0 + pd.Timedelta(minutes=self.forecast_horizon_mins)

    def predict_next(self, dataset):
        """
		Predict only for the next timestep; dataset as Pandas DataFrame (timestamp, power)
		"""
        ds = dataset.copy(deep=True)

        ds.power = np.nan_to_num(ds.power.values)

        lagged_vals = ds.power.values[-self.sliding_window_width:].reshape((1, self.sliding_window_width))

        t0 = ds.index[-1]

        X = self.generate_input_data(lagged_vals, t0)

        return self.predict_on_preprocessed_input(X), t0, t0 + pd.Timedelta(minutes=self.forecast_horizon_mins)

    def generate_dataset_filename(self, dataset_identifier):
        """
		Generate the dataset's filename corresponding to the model's granularity
		"""
        return utils.generate_dataset_filename(dataset_identifier, self.granularity_s)

    def open_dataset_file(self, filename, preprocessed_datasets_folder=config.preprocessed_datasets_folder):
        """
		"""
        return utils.open_dataset_file(filename, self.working_directory, preprocessed_datasets_folder)

    def scale(self, values):
        """
		Scales the values by the model's scaling_factor
		"""
        return values / (self.scaling_factor * 1.0)

    def reverse_scale(self, values):
        """
		"""
        return values * (self.scaling_factor * 1.0)


    def preprocess_input_data(self, dataset):
        ds = dataset.copy(deep=True)
        ds.power = np.nan_to_num(ds.power.values)
        nb_forecast_steps = int(dt.timedelta(minutes=self.forecast_horizon_mins).total_seconds() / self.granularity_s)

        # Cut-off unnecessary data points
        history_offset = self.sliding_window_width + nb_forecast_steps - 1

        lagged_vals = ds.power.values[-history_offset:]
        s = ds.power.values.itemsize
        lagged_vals = as_strided(lagged_vals, shape=(nb_forecast_steps, self.sliding_window_width), strides=(s, s))
        t0 = ds.index[-nb_forecast_steps:]
        X = self.generate_input_data(lagged_vals, t0)

        return X


    def preprocess_input_data_recursive(self, dataset):
        ds = dataset.copy(deep=True)
        ds.power = np.nan_to_num(ds.power.values)
        nb_forecast_steps = 1 # <-

        nb_examples = len(ds.power.values) - self.sliding_window_width
        s = ds.power.values.itemsize
        lagged_vals = as_strided(ds.power.values, shape=(nb_examples, self.sliding_window_width), strides=(s, s))
        t0 = ds.index[self.sliding_window_width:]
        ground_truth = ds.power.values[self.sliding_window_width + nb_forecast_steps - 1:]

        X = self.generate_input_data(lagged_vals, t0)
        y = self.generate_output_data(ground_truth)

        return X, y, ground_truth, t0




    def generate_training_data(self, dataset, train_cv_test_split=(0.8, 0.1, 0.1), cleanse=False):
        """
		From Pandas DataFrame (timestep, power) generate all valid training examples and split respectivly; X and y are scaled to model scale.
		"""
        ds = dataset.copy(deep=True)

        ds.power = np.nan_to_num(ds.power.values)

        nb_forecast_steps = int(dt.timedelta(minutes=self.forecast_horizon_mins).total_seconds() / self.granularity_s)

        nb_examples = len(ds.power.values) - nb_forecast_steps - self.sliding_window_width

        s = ds.power.values.itemsize

        self.scaling_factor = max(ds.power.values)

        lagged_vals = as_strided(ds.power.values, shape=(nb_examples, self.sliding_window_width), strides=(s, s))

        # print 'series', ds.power.values[2:2+(self.sliding_window_width+nb_forecast_steps)]
        # print 'history', X[2]
        # print X.shape
        # p_t0 = ds.power.values[self.sliding_window_width-1:-nb_forecast_steps-1]
        if self.sliding_window_width != 0:
            t0 = ds.index[self.sliding_window_width - 1:-nb_forecast_steps - 1]
        else:
            t0 = ds.index[:-nb_forecast_steps]
        # print ds.loc[t0[2]]

        # print "p_t0", p_t0[2]

        if self.forecast_type == 'watthours':
            s = ds.power.values.itemsize
            power_intervals = as_strided(ds.power.values[self.sliding_window_width:],
                                         shape=(nb_examples, nb_forecast_steps), strides=(s, s))
            # print "fc_hor", power_intervals[2]
            # print "last_elem_of_sum", power_intervals[:,-1]
            # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].power.values
            mask = np.unique(np.where(power_intervals != 0.0)[
                                 0])  # np.where returns indices for nonzero values as [xi][yi]; take only unique row
			#  indices
            ground_truth = np.sum(power_intervals,
                                  axis=-1) * self.granularity_s / 3600  # integrate power over forecast horizon to get total energy in Wh
        # print y.shape
        elif self.forecast_type == 'watts':
            ground_truth = ds.power.values[self.sliding_window_width + nb_forecast_steps - 1:-1]
            # mask = np.array(np.where(y != 0.0)).reshape((-1))

        # print 'gt', ground_truth
        # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].power.values
        else:
            print('Unsupported forecast type. Please define forecast type as either \'watts\' or \'watthours\'.')
            return

        X = self.generate_input_data(lagged_vals, t0)
        y = self.generate_output_data(ground_truth)

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
               (X_test, y_test, ground_truth_test, t0_test)

    def generate_training_data_lstm(self, dataset, train_cv_test_split=(0.8, 0.1, 0.1), cleanse=False):
        """
		From Pandas DataFrame (timestep, power) generate all valid training examples and split respectivly; X and y
		are scaled to model scale.
		"""
        ds = dataset.copy(deep=True)

        ds.power = np.nan_to_num(ds.power.values)

        # Scale dataset
        # data set is scaled by dividing all loads by the maximum load form the data set
        self.scaling_factor = np.max(ds.power)
        # ds.power = ds.power / self.scaling_factor

        nb_forecast_steps = int(dt.timedelta(minutes=self.forecast_horizon_mins).total_seconds() / self.granularity_s)
        nb_examples = len(ds.power.values) - nb_forecast_steps - self.sliding_window_width
        s = ds.power.values.itemsize
        lagged_vals = as_strided(ds.power.values, shape=(nb_examples, self.sliding_window_width), strides=(s, s))

        if self.sliding_window_width != 0:
            t0 = ds.index[self.sliding_window_width - 1:-nb_forecast_steps - 1]
        else:
            t0 = ds.index[:-nb_forecast_steps]

        if self.forecast_type == 'watthours':
            s = ds.power.values.itemsize
            power_intervals = as_strided(ds.power.values[self.sliding_window_width:],
                                         shape=(nb_examples, nb_forecast_steps), strides=(s, s))
            # print "fc_hor", power_intervals[2]
            # print "last_elem_of_sum", power_intervals[:,-1]
            # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].power.values
            mask = np.unique(np.where(power_intervals != 0.0)[
                                 0])  # np.where returns indices for nonzero values as [xi][yi]; take only unique row
            #  indices
            ground_truth = np.sum(power_intervals,
                                  axis=-1) * self.granularity_s / 3600  # integrate power over forecast horizon to
            # get total energy in Wh
        # print y.shape
        elif self.forecast_type == 'watts':
            ground_truth = ds.power.values[self.sliding_window_width + nb_forecast_steps - 1:-1]
            #mask = np.array(np.where(y != 0.0)).reshape((-1))
        # print 'gt', ground_truth
        # print "P_t1", ds.loc[t0+pd.Timedelta(minutes=self.forecast_horizon_mins)].power.values
        else:
            print('Unsupported forecast type. Please define forecast type as either \'watts\' or \'watthours\'.')
            return

        # print(">>ground truth", ground_truth)
        # ground truth real values
        ground_truth = ground_truth.reshape(-1, 1)
        ground_truth = (self.scaling_factor * ground_truth)

        X = self.generate_input_data(lagged_vals, t0)

        # y is an vector with normalized energy consumption within the time interval
        y = self.generate_output_data(ground_truth)

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
               (X_test, y_test, ground_truth_test, t0_test)


    def visualize_input_weights(self):
        """
		Show weights of first layer as heat map
		"""
        import matplotlib.pyplot as plt
        data = self.model.layers[1].get_weights()[0]
        print (data.shape)
        img = plt.imshow(np.transpose(data), interpolation='nearest')
        plt.show()

