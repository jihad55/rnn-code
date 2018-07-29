# rnn-code
rnn code


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#get input and output
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#reshaping
X_train = np.reshape(X_train, (1257,1, 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()
#add lstm layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

#add output layer
regressor.add(Dense(units = 1))

#compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting rnn to training set
regressor.fit(X_train, y_train, batch_size = 32, epochs= 200)
