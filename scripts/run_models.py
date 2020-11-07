import os
import numpy as np
import keras
import random
from . import constants, metrics, log
from .preprocess import preprocess_cnn, preprocess_lstm
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, Input, concatenate
import tensorflow as tf
from tensorflow.random import set_seed


# Used to make experiments as reproduceable as possible
def reset_seed():
    os.environ['PYTHONHASHSEED'] = str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def reverse_ordinal_encoding(grades):
    return [4 + np.sum(grade) for grade in grades]


# Uses ordinal probabilities gotten from model predictions to determine predicted grade
def convert_ordinal_prob_to_grade(pred):
    converted = [[1 if prob >= 0.5 else 0 for prob in sample]
                 for sample in pred]
    return reverse_ordinal_encoding(converted)


def run(problems, hold_positions, model_type):
    if model_type == "CNN":
        return run_cnn(problems)
    elif model_type == "LSTM" or model_type == "LSTM_RANDOM":
        return run_lstm(problems, hold_positions, model_type)


def run_lstm(problems, hold_positions, model_type):
    log.log_output(model_type, "Begin preprocessing for " + model_type)
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_lstm(problems, hold_positions, random_beta=model_type=="LSTM_RANDOM")
    log.log_output(model_type, "Completed preprocessing for " + model_type)

    hyperparameters = constants.hyperparameters_lstm

    reset_seed()

    model = Sequential()
    model.add(LSTM(hyperparameters['nodes_2'], input_shape=(
        x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(LSTM(hyperparameters['nodes_1']))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    log.log_output(model_type, "Begin " + model_type + " Training")

    history = model.fit(x_train, y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], validation_data=(
        x_val, y_val), verbose=1)

    log.log_output(model_type, "Completed " + model_type + " Training")

    pred = convert_ordinal_prob_to_grade(model.predict(x_test))
    table, mse = metrics.ordinal_evaluation_report(y_test, pred, return_mse=True)

    log.log_output(model_type, "Scores for " + model_type + " on test dataset:\n\n" + table)

    return mse


def run_cnn(problems):
    log.log_output("CNN", "Begin preprocessing for CNN")
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_cnn(problems)
    log.log_output("CNN", "Completed preprocessing for CNN")

    reset_seed()

    inputs = Input(shape=(18, 11, 1))
    conv1 = Conv2D(filters=4, strides=1, kernel_size=(11, 7))(inputs)
    conv2 = Conv2D(filters=1, kernel_size=1)(inputs)
    flat1 = Flatten()(conv1)
    flat2 = Flatten()(conv2)
    combine = concatenate([flat1, flat2])
    dense1 = Dense(5, activation='sigmoid')(combine)
    dense2 = Dense(50, activation='sigmoid')(dense1)
    outputs = Dense(10, activation='sigmoid')(dense2)

    model = keras.Model(inputs, outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    log.log_output("CNN", "Begin CNN Training")

    history = model.fit(x_train, y_train, epochs=40, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1)

    log.log_output("CNN", "Completed CNN Training")

    pred = convert_ordinal_prob_to_grade(model.predict(x_test))
    table, mse = metrics.ordinal_evaluation_report(y_test, pred, return_mse=True)

    log.log_output("CNN", "Scores for CNN on test dataset:\n\n" + table)

    return mse