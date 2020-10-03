import os
import errno
from os import listdir
from os.path import isfile, join
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
    layer = m.layers[layer_number].output # e.g. 4
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
    df = df.loc['2014-01-05':]  # has to be dividable into weeks (7 days)
    # split into first_week and other_weeks of time period
    first_week, other_weeks = df[:7], df[7:]
    # restructure into windows of weekly data
    first_week = np.array(np.split(first_week.values, len(first_week) / 7))
    other_weeks = np.array(np.split(other_weeks.values, len(other_weeks) / 7))  # prepare input data
    # define input size (7 days)
    n_input = 7

    # history is a list of weekly data
    history = [x for x in first_week]

    # walk-forward validation over each week
    prediction = list()
    for i in range(len(other_weeks)):
        # predict the year based on last week history
        yhat_week = forecast(layer_model, history, n_input)
        # print(yhat_sequence.shape)
        # store the predictions
        prediction.append(yhat_week)
        # get real observation and add to history for predicting the next week
        history.append(other_weeks[i, :])
    # predict the last week in history
    last_week = forecast(layer_model, history, n_input)
    prediction.append(last_week)

    activations = np.array(prediction)
    num_datapoints, number_inputs, neurons = activations.shape
    reshaped_activations = activations.reshape((num_datapoints * number_inputs, neurons))
    np.savetxt(save_path + str(layer_number) + ".csv", reshaped_activations, delimiter=",")
    del m, layer_model
    K.clear_session()

'''
models = ['../models/pretrained/start2014/branch1_cnn_weekly.h5',
          '../models/pretrained/start2014/branch2_cnn_weekly.h5',
          '../models/pretrained/start2014/branch3_cnn_weekly.h5',
          '../models/pretrained/start2014/branch4_cnn_weekly.h5',
          '../models/pretrained/start2014/branch5_cnn_weekly.h5',
          '../models/pretrained/start2014/branch6_cnn_weekly.h5'
          ]
'''
# set model folder - iterative for degrees of transfer
model_folder = '../weekly_forecast/temp-start2014/transfer4/'
models = sorted([f for f in listdir(model_folder) if isfile(join(model_folder, f))])

datasets = ['../data/preprocessed/branch1_weekly.pkl',
            '../data/preprocessed/branch2_weekly.pkl',
            '../data/preprocessed/branch3_weekly.pkl',
            '../data/preprocessed/branch4_weekly.pkl',
            '../data/preprocessed/branch5_weekly.pkl',
            '../data/preprocessed/branch6_weekly.pkl'
            ]

for model in models:
    for dataset in datasets:
        i = 4
        if dataset[-12] not in model[17:-3]: # dataset[-12] == model[-4]
            while i < 12:  # 16 for max-pooling layers
                path_string = "activations/activations_crosswise_weekly_start2014/transfer4/m" + model[17:-3] + "_x" + dataset[-12] + "_weekly/"
                model_path = model_folder + model
                get_activation_vector(model_path, dataset, i, path_string)
                i += 1
        else:
            continue
