from keras import backend as K
import numpy as np

def mean_neg_log_loss_parametric(y_true, y_pred):
	"""
	Loss function for mixture density networks
	"""
	eps = 1e-30
	nb_kernels = int(K.int_shape(y_pred)[1]/3)
	coeffs = y_pred[:, 0: nb_kernels]
	centroids = y_pred[:, nb_kernels : 2*nb_kernels] 
	variances = y_pred[:, 2*nb_kernels : 3*nb_kernels] 
	exponentials = K.exp(- K.square(centroids - y_true) /2.0 / K.square(variances))
	neg_log_loss = -K.log( K.sum(coeffs / variances * exponentials, axis=1) +eps)
	return K.mean(neg_log_loss, axis=-1)

def mean_neg_log_loss_discrete(y_true, y_pred):
	"""
	Equals categorical cross entropy for one hot vector but by implementing it, eps can be set
	"""
	eps = 1e-30

	llh = K.clip( K.sum(y_true * y_pred, axis = 1), min_value=eps, max_value=20.0)

	return K.mean( -K.log(llh), axis = 0)


