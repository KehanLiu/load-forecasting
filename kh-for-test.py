from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

# define a function to convert flow rate to energy
def convertFlowrateToEnergy(vectorSeries):
    vector=[]
    for i in vectorSeries:
        vector.append(i  * 1000 * 4.186 * 50 * 1/3600000) #E=V*rou*c*deltaT, unit KWh
    return vector
    
# random seed
# np.random.seed(1234)
   
# load raw data, df for dataframe
df_raw = pd.read_csv(r'C:\Users\leoliu\Dropbox\learning_kh\master thesis\eon acs\my work\model\dataset\tool\DHW0001\loadprofile_14d_every5mins.csv', header=None)
# numpy array
df_raw_array = df_raw.values
# 5mins load (288 loads for each day)
list_5mins_load = [df_raw_array[i,1] for i in range(0, len(df_raw))]
#convert 5 mins flow rate to energy
vector_load = convertFlowrateToEnergy(list_5mins_load)
# the length of the sequnce for predicting the future value
sequence_length = 24

# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(vector_load, sequence_length)

# Divide all values by maximum value
matrix_load = np.array(matrix_load)
print (np.max(matrix_load))
max_value = np.max(matrix_load)
#a = matrix_load
matrix_load = matrix_load/max_value

# shift all data by mean
#shifted_value = matrix_load.mean()
#matrix_load -= shifted_value

print ("Data  shape: ", matrix_load.shape)


# split dataset: 90% for training set and 10% for testing
train_row = int(round(0.9 * matrix_load.shape[0]))
train_set = matrix_load[:train_row, :]

# shuffle the training set (but do not shuffle the test set),but why??????????
# np.random.shuffle(train_set)
# the training set
X_train = train_set[:, :-1]
# the last column is the true value to compute the mean-squared-error loss
y_train = train_set[:, -1] 
# the test set
X_test = matrix_load[train_row:, :-1]
y_test = matrix_load[train_row:, -1]

# the input to LSTM layer needs to have the shape of (number of samples, the dimension of each element)???But in the official document of Keras, the input should be in the shape of (samples, timesteps, input_dim)
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# build the model
model = Sequential()
# layer 1: LSTM 如果return_sequences=True：返回形如（samples，timesteps，output_dim）的3D张量否则，返回形如（samples，output_dim）的2D张量， 这里没填num_units的数量，该值表示隐藏层神经元的个数
model.add(LSTM(input_dim=23, output_dim=50, return_sequences=True))
model.add(Dropout(0.2))
# layer 2: LSTM, 我们这里使用了两个LSTM进行叠加，第二个LSTM第一个参数指的是输入的维度，这和第一个LSTM的输出维度并不一样，这也是LSTM比较“随意”的地方。最后一层采用了线性层。
model.add(LSTM(output_dim=100, return_sequences=False))
model.add(Dropout(0.2))
# layer 3: dense 全连接神经层
# linear activation: a(x) = x
model.add(Dense(output_dim=1, activation='linear'))
# compile the model
model.compile(loss="mse", optimizer="rmsprop")

# train the model

#batchsize越小，一个batch中的随机性越大，越不易收敛。然而batchsize越小，速度越快，权值更新越频繁；且具有随机性，对于非凸损失函数来讲，更便于寻找全局最优。从这个角度看，收敛更快，更容易达到全局最优。
#batchsize越大，越能够表征全体数据的特征，其确定的梯度下降方向越准确，（因此收敛越快），且迭代次数少，总体速度更快。然而大的batchsize相对来讲缺乏随机性，容易使梯度始终向单一方向下降，陷入局部最优；而且当batchsize增大到一定程度，再增大batchsize，一次batch产生的权值更新（即梯度下降方向）基本不变。因此理论上存在一个最合适的batchsize值，使得训练能够收敛最快或者收敛效果最好（全局最优点）。
#根据现有的调参经验，加入正则化项BN后，在内存容量允许的情况下，一般来说设置一个较大的batchsize值更好，通常从128开始调整。
#epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
#validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之后，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
model.fit(X_train, y_train, batch_size=32, nb_epoch=50, validation_split=0.1, verbose=1)

# evaluate the result
#verbose：含义同fit的同名参数，但只能取0或1
test_mse = model.evaluate(X_test, y_test, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))

# get the predicted values
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))

# plot the results
fig = plt.figure()
plt.plot(y_test)
plt.plot(predicted_values)
plt.xlabel('5mins')
plt.ylabel('DHW load (*1e5)')
plt.show()
fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')

# save the result into csv file
y_test = np.reshape(y_test, (len(y_test),1))
test_result_multiply_max_value = (np.hstack((predicted_values, y_test)) + shifted_value) * max_value
np.savetxt('output_load_forecasting_result_multiply_max_value.csv', test_result_multiply_max_value, fmt='%.2f',delimiter=',',header="#pres-data,#real-data")
#test_result_plus_no_shifted_value = np.hstack((predicted_values, y_test))
#np.savetxt('output_load_forecasting_result_plus_no_shifted_value.csv', test_result_plus_no_shifted_value, fmt='%.2f',delimiter=',',header="#real-data,#pred-data")