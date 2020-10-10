import os
import numpy as np
import keras
import random
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
# from tensorflow import set_seed

# Used to make experiments as reproduceable as possible
def reset_seed():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.set_random_seed(1)
    np.random.seed(1)
    random.seed(1)


def run_lstm(x_train, y_train, x_val, y_val, x_test, y_test):
    reset_seed()
    model = Sequential()
    model.add(LSTM(15, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(LSTM(10))
    # model.add(Dense(11, activation='softmax'))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val), verbose=1)