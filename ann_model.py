from ann_forecast.mdn import Mdn_feedforward
from ann_forecast.sdn import Discrete_feedforward
from ann_forecast.lstm import Prob_LSTM
from sklearn.exceptions import NotFittedError
import numpy as np
import datetime as dt
import pandas as pd



class ann_model(object):


	# NOTE:  // TODO: 3 Options for forecasting
	#// This class implements variant 2
	#
	# 1) forecasting horizon = model granularity
	#   => 1 model, [horizon] additional lagged values
	# 2) varying model granularity up to forecasting horizon
	#   => [horizon]/[granularity] models
		# 3) data granularity = model granularity, recursive predictions
	#   => 1 model, predictions are fed into model as new features for subsequent timesteps


	def __init__(self):


		self.model_list = []
		self.model_type = None
		self.granularity = None
		self.custom_granularity = False
		self.forecast_steps = None
		self.nb_lagged_vals = None
		self.use_cal_vars = None


# ----------------------------------------------------------------------------------------------------------------------


	def fit_mdn(self, raw_data, forecast_horizon_min, granularity_min='default',
								look_back_mins=60,
								hidden_neurons=[40,40,40],
								use_cal_vars=True,
								dropout = 0.2,
								kernels=5,
								epochs=20):

		# Safe model type
		self.model_type='mdn'

		# Empty previous model list (if existing)
		self.model_list = []

		self.use_cal_vars = use_cal_vars

		# Derive granularity in minutes from data or resample if specified
		if granularity_min == 'default':
			self.granularity = (raw_data.index[1] - raw_data.index[0]).seconds / 60
		else:
			self.custom_granularity = True
			self.granularity = granularity_min
			raw_data = raw_data.resample(str(self.granularity) + 'Min').interpolate(method="linear")

		# Determine number of lagged values:
		self.nb_lagged_vals = look_back_mins / self.granularity

		# Forecast horizon in granularity steps
		self.forecast_steps = forecast_horizon_min / self.granularity

		# Fit a model for each forecasting step up to the forecasting horizon
		for i in range(self.forecast_steps):

			print("Fit model " + str(i+1) + "/" + str(self.forecast_steps))

			for_hor_iter = (i + 1) * self.granularity


			model = Mdn_feedforward(model_identifier = 'some-arbitrary-id',
											forecast_type='watts',
											granularity_s=self.granularity*60,
											forecast_horizon_mins=for_hor_iter,
											look_back_mins=look_back_mins,
											hidden_neurons=hidden_neurons,
											use_cal_vars=use_cal_vars,
											dropout = dropout,
											nb_kernels=kernels)

			(X_train, y_train, _gt_train, _t0_train), _, _ = model.generate_training_data(
				raw_data, train_cv_test_split=(1, 0, 0), cleanse=False)
			model.train(X_train, y_train, nb_epoch=epochs, verbose=1)

			self.model_list.append(model)


# ----------------------------------------------------------------------------------------------------------------------


	def fit_sdn(self, raw_data, forecast_horizon_min, granularity_min='default',
									look_back_mins=60,
									hidden_neurons=[40, 40, 40],
									use_cal_vars=True,
									dropout=0.2,
									epochs=20,
									pdf_sample_points_min = 0.0,
									pdf_sample_points_max = 7000.0,
									pdf_resolution = 200.0):

		# Safe model type
		self.model_type = 'sdn'

		# Empty previous model list (if existing)
		self.model_list = []

		self.use_cal_vars = use_cal_vars

		# Derive granularity in minutes from data or resample if specified
		if granularity_min == 'default':
				self.granularity = (raw_data.index[1] - raw_data.index[0]).seconds / 60
		else:
			self.custom_granularity = True
			self.granularity = granularity_min
			raw_data = raw_data.resample(str(self.granularity) + 'Min').interpolate(method="linear")

		# Determine number of lagged values:
		self.nb_lagged_vals = look_back_mins / self.granularity

		# Forecast horizon in granularity steps
		self.forecast_steps = forecast_horizon_min / self.granularity

		# Fit a model for each forecasting step up to the forecasting horizon
		for i in range(self.forecast_steps):
			print("Fit model " + str(i + 1) + "/" + str(self.forecast_steps))
			for_hor_iter = (i + 1) * self.granularity

			model = Discrete_feedforward(model_identifier='some-arbitrary-id',
											forecast_type='watts',
											granularity_s=self.granularity * 60,
											forecast_horizon_mins=for_hor_iter,
											look_back_mins=look_back_mins,
											hidden_neurons=hidden_neurons,
											use_cal_vars=use_cal_vars,
											dropout=dropout,
											pdf_sample_points_min=pdf_sample_points_min,
											pdf_sample_points_max=pdf_sample_points_max,
											pdf_resolution=pdf_resolution)

			(X_train, y_train, _gt_train, _t0_train), _, _ = model.generate_training_data(
																raw_data, train_cv_test_split=(1, 0, 0), cleanse=False)
			model.train(X_train, y_train, nb_epoch=epochs, verbose=1)

			self.model_list.append(model)





	def fit_lstm(self, raw_data, forecast_horizon_min, granularity_min='default',
                 look_back_mins = 60,
                 hidden_neurons=[40, 40, 40],
                 dropout=0.2,
                 epochs=20,
                 use_cal_vars=True,
                 pdf_sample_points_min=0.0,
                 pdf_sample_points_max=7000.0,
                 pdf_resolution=200.0):

		# Safe model type
		self.model_type = 'lstm'

		# Empty previous model list (if existing)
		self.model_list = []

		self.use_cal_vars = use_cal_vars

		# Derive granularity in minutes from data or resample if specified
		if granularity_min == 'default':
			self.granularity = (raw_data.index[1] - raw_data.index[0]).seconds / 60
		else:
			self.custom_granularity = True
			self.granularity = granularity_min
			raw_data = raw_data.resample(str(self.granularity) + 'Min').interpolate(method="linear")

		# Determine number of lagged values:
		self.nb_lagged_vals = look_back_mins / self.granularity

		# Forecast horizon in granularity steps
		self.forecast_steps = forecast_horizon_min / self.granularity

		# Fit a model for each forecasting step up to the forecasting horizon
		for i in range(self.forecast_steps):
			print("Fit model " + str(i + 1) + "/" + str(self.forecast_steps))
			for_hor_iter = (i + 1) * self.granularity

			model = Prob_LSTM(model_identifier='some-arbitrary-id',
										forecast_type='watts',
										granularity_s=self.granularity * 60,
										forecast_horizon_mins=for_hor_iter,
										look_back_mins=look_back_mins,
										hidden_neurons=hidden_neurons,
										use_cal_vars=use_cal_vars,
										dropout=dropout,
										pdf_sample_points_min=pdf_sample_points_min,
										pdf_sample_points_max=pdf_sample_points_max,
										pdf_resolution=pdf_resolution)

			(X_train, y_train, _gt_train, _t0_train), _, _ = model.generate_training_data_lstm(
				raw_data, train_cv_test_split=(1, 0, 0), cleanse=False)
			model.train(X_train, y_train, nb_epoch=epochs, verbose=1)

			self.model_list.append(model)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


	def predict(self, recent_data):

		# Check if fitted model is available
		if not self.model_list:
			raise NotFittedError("NotFittedError: The model instance is not fitted yet. Call 'fit' with appropriate "
			 "arguments before using this method." % {'name': 'SVR'})

		# Resample if required
		if self.custom_granularity:
			recent_data = recent_data.resample(str(self.granularity) + 'Min').interpolate(method="linear")

		# Extract features
		lagged_vals = recent_data.power.values[-self.nb_lagged_vals:]
		t = recent_data.index[-1]
		if self.use_cal_vars:
			cal_vars = np.array([t.minute / 60.0, t.hour / 24.0, t.dayofweek / 7.0, t.month / 12.0])
		expected_vals = []
		timestamps = []

		if self.model_type == 'mdn':

			# Estimate for each forecast step
			for model in self.model_list:
				prprc_in = model.scale(lagged_vals)
				if self.use_cal_vars:
					prprc_in = np.hstack((prprc_in, cal_vars))
				(coeffs, std_devs, centroids) = model.predict_on_preprocessed_input(prprc_in.reshape(1, len(prprc_in)))
				estimate = np.sum(np.multiply(coeffs, centroids), axis=1)
				expected_vals.append(estimate[0])
				t = t + dt.timedelta(minutes=self.granularity)
				timestamps.append(t)

		if self.model_type == 'sdn':

			# Estimate for each forecast step
			for model in self.model_list:
				prprc_in = model.scale(lagged_vals)
				if self.use_cal_vars:
					prprc_in = np.hstack((prprc_in, cal_vars))
				(pdf_sp, pdf_sv) = model.predict_on_preprocessed_input(prprc_in.reshape(1, len(prprc_in)))
				intervals = pdf_sp[:, 1:] - pdf_sp[:, :-1]
				sp = 0.5 * (pdf_sp[:, 1:] + pdf_sp[:, :-1])
				sv = 0.5 * (pdf_sv[:, 1:] + pdf_sv[:, :-1])
				estimate = np.sum(np.multiply(sp, np.multiply(sv, intervals)), axis=1)
				expected_vals.append(estimate[0])
				t = t + dt.timedelta(minutes=self.granularity)
				timestamps.append(t)


		if self.model_type == 'lstm':

			# Estimate for each forecast step
			for model in self.model_list:
				prprc_in = model.scale(lagged_vals)
				prprc_in = prprc_in.reshape((1, lagged_vals.size, 1))

				if self.use_cal_vars:
					cal_vars_reshaped = np.zeros((1, lagged_vals.size, 4))
					minutes = np.zeros(prprc_in.shape)
					hours = np.zeros(prprc_in.shape)
					days = np.zeros(prprc_in.shape)
					months = np.zeros(prprc_in.shape)
					for i in range(lagged_vals.size):
						minutes[0][i][0] = cal_vars[0]
						hours[0][i][0] = cal_vars[1]
						days[0][i][0] = cal_vars[2]
						months[0][i][0] = cal_vars[3]
					prprc_in = np.concatenate((prprc_in, minutes, hours, days, months), axis=2)

				# print(prprc_in)
				(pdf_sp, pdf_sv) = model.predict_on_preprocessed_input(prprc_in)
				intervals = pdf_sp[:, 1:] - pdf_sp[:, :-1]
				sp = 0.5 * (pdf_sp[:, 1:] + pdf_sp[:, :-1])
				sv = 0.5 * (pdf_sv[:, 1:] + pdf_sv[:, :-1])
				estimate = np.sum(np.multiply(sp, np.multiply(sv, intervals)), axis=1)
				expected_vals.append(estimate[0])
				t = t + dt.timedelta(minutes=self.granularity)
				timestamps.append(t)



		prediction = pd.DataFrame({'power': expected_vals}, timestamps)

		return prediction



	def get_model_type(self):
		return self.model_type