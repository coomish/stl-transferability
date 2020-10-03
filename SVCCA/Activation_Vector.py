import os
import errno
import numpy as np
import pandas as pd
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from utils import forecast


def get_activation_vector(model, dataset, layer_number, save_path):
    # create directory for saving activations
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # load model
    m = load_model(model)

    # define layer
    layer = m.layers[layer_number].output  # e.g. 4
    # build layer model
    layer_model = Model(inputs=m.input,
                        outputs=layer)
    layer_model.summary()

    # load dataset
    df = pd.read_pickle(dataset)
    df = pd.DataFrame(data=df.values, index=df.index, columns=['netto'])
    # assign features for month, weekday, year
    df = df.assign(month=df.index.month)
    df = df.assign(weekday=df.index.weekday)
    df = df.assign(year=df.index.year)
    # define the time period for which to calculate the activation vectors
    df = df.loc['2014-01-01':]  # has to be dividable into years (365 days)
    # split into first_year and other_years of time period
    first_year, other_years = df[:365], df[365:]
    # restructure into windows of yearly data
    first_year = np.array(np.split(first_year.values, len(first_year) / 365))
    other_years = np.array(np.split(other_years.values, len(other_years) / 365))  # prepare input data
    # define input size (365 days)
    n_input = 365

    # history is a list of yearly data
    history = [x for x in first_year]

    # walk-forward validation over each year
    prediction = list()
    for i in range(len(other_years)):
        # predict the year based on last year history
        yhat_year = forecast(layer_model, history, n_input)
        # store the predictions
        prediction.append(yhat_year)
        # get real observation and add to history for predicting the next year
        history.append(other_years[i, :])
    # predict the last year in history
    last_year = forecast(layer_model, history, n_input)
    prediction.append(last_year)

    activations = np.array(prediction)
    num_data_points, number_inputs, neurons = activations.shape
    reshaped_activations = activations.reshape((num_data_points * number_inputs, neurons))
    np.savetxt(save_path + str(layer_number) + ".csv", reshaped_activations, delimiter=",")
    del m, layer_model
    K.clear_session()


models = ['../models/pretrained/branch1_cnn.h5',
          '../models/pretrained/branch2_cnn.h5',
          '../models/pretrained/branch3_cnn.h5',
          '../models/pretrained/branch4_cnn.h5',
          '../models/pretrained/branch5_cnn.h5',
          '../models/pretrained/branch6_cnn.h5'
          ]

data_sets = ['../data/preprocessed/branch1.pkl',
             '../data/preprocessed/branch2.pkl',
             '../data/preprocessed/branch3.pkl',
             '../data/preprocessed/branch4.pkl',
             '../data/preprocessed/branch5.pkl',
             '../data/preprocessed/branch6.pkl'
             ]

for model in models:
    for dataset in data_sets:
        i = 4
        while i < 24:
            path_string = "activations/m" + model[-8] + "_x" + dataset[-5] + "/"
            get_activation_vector(model, dataset, i, path_string)
            i += 1
