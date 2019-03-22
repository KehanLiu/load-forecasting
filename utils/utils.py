import os.path
import datetime as dt
import pandas as pd

def generate_dataset_filename(model_identifier, granularity_s):
	"""
	"""
	name = str(model_identifier) + '_granu' + str(granularity_s) + '.csv'
	return name

def open_dataset_file(filename, working_dir, preprocessed_datasets_folder):
	training_data_url = os.path.join(working_dir, preprocessed_datasets_folder, filename)

	if not os.path.isfile(training_data_url):
		print(dt.datetime.now().strftime('%x %X') + \
			' Training data file ' + training_data_url + \
			' does not exist. Please generate preprocessed datasets first.')
		return pd.DataFrame()

	print(dt.datetime.now().strftime('%x %X') + ' Loading training data...')

	dataset = pd.read_csv(training_data_url)

	dataset = dataset.set_index('timestamp')
	dataset.index = pd.to_datetime(dataset.index, infer_datetime_format=True) # convert string to datetime format
	
	return dataset

def open_dataset_for_test_file(filename, working_dir, preprocessed_datasets_folder):
    testing_data_url = os.path.join(working_dir, preprocessed_datasets_folder, filename)
    
    dataset_for_test = pd.read_csv(testing_data_url)

    dataset_for_test = dataset_for_test.set_index('timestamp')
    dataset_for_test.index = pd.to_datetime(dataset_for_test.index, infer_datetime_format=True) # convert string to datetime format
	
    return dataset_for_test