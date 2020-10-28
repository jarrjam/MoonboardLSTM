import os
import numpy as np
import keras
import random
from . import constants, metrics
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, Input, concatenate
import tensorflow as tf
from tensorflow.random import set_seed
import wandb
from wandb.keras import WandbCallback
from sklearn.metrics import f1_score, accuracy_score, classification_report, mean_squared_error, mean_absolute_error

if constants.wandb_mode == "LSTM":
    wandb.init(config=constants.hyperparameters_lstm,
               project="moonboard_rnn", group="LSTM_original", entity="jarrjam")
    config = wandb.config
elif constants.wandb_mode == "CNN":
    wandb.init(project="moonboard_rnn", group="CNN", entity="jarrjam")


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


def run_lstm(x_train, y_train, x_val, y_val, x_test, y_test):
    if constants.wandb_mode == "LSTM":
        hyperparameters = config
        callbacks = [WandbCallback()]
    else:
        hyperparameters = constants.hyperparameters_lstm
        callbacks = []

    reset_seed()
    model = Sequential()

    model.add(LSTM(hyperparameters['nodes_2'], input_shape=(
        x_train.shape[1], x_train.shape[2]), return_sequences=True))
    model.add(LSTM(hyperparameters['nodes_1']))

    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    history = model.fit(x_train, y_train, epochs=hyperparameters['epochs'], batch_size=hyperparameters['batch_size'], validation_data=(
        x_val, y_val), verbose=1, callbacks=callbacks)

    # if constants.wandb_mode == "LSTM":
    #     model.save(os.path.join(wandb.run.dir, "model.h5"))

    pred = convert_ordinal_prob_to_grade(model.predict(x_test))

    print("F1:", f1_score(np.array(y_test), pred, average='weighted'))
    print("Accuracy:", accuracy_score(np.array(y_test), pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("Macro MSE:", metrics.macro_mse(y_test, pred))
    print(classification_report(np.array(y_test), pred))


def run_cnn(x_train, y_train, x_val, y_val, x_test, y_test):
    if constants.wandb_mode == "CNN":
        callbacks = [WandbCallback()]
    else:
        callbacks = []
    
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

    history = model.fit(x_train, y_train, epochs=40, batch_size=64,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks)

    pred = convert_ordinal_prob_to_grade(model.predict(x_test))

    print("F1:", f1_score(np.array(y_test), pred, average='weighted'))
    print("Accuracy:", accuracy_score(np.array(y_test), pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("MAE:", mean_absolute_error(y_test, pred))
    print("Macro MSE:", metrics.macro_mse(y_test, pred))
    print(classification_report(np.array(y_test), pred))
