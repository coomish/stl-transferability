# multi headed multi-step cnn
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras import backend as K
from utils import calculate_various_scores, calculate_scores, split_dataset, to_supervised, forecast, \
    save_prediction_plot, save_results
seed = 49
np.random.seed(seed)

# train the model
def build_model(train, n_input, epochs, batch_size, filters, kernel_size, pool_size, dense_1, dense_2):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, 365)
    # define parameters
    verbose = 0
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # create a channel for each variable
    in_layers, out_layers = list(), list()
    for i in range(n_features):
        inputs = Input(shape=(n_timesteps, 1))
        conv1 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
        conv2 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(conv1)
        conv3 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(conv2)
        conv4 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(conv3)
        pool1 = MaxPooling1D(pool_size=pool_size)(conv4)
        flat = Flatten()(pool1)
        # store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # merge heads
    merged = concatenate(out_layers)
    # interpretation
    dense1 = Dense(dense_1, activation='relu')(merged)
    dense2 = Dense(dense_2, activation='relu')(dense1)
    outputs = Dense(n_outputs)(dense2)
    model = Model(inputs=in_layers, outputs=outputs)
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # fit network
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    model.fit(input_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


def evaluate_model(train, test, n_input, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2):
    # fit model
    model = build_model(train, n_input, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2)
    # history is a list of yearly data
    history = [x for x in train]
    # walk-forward validation over each year
    prediction = list()
    for i in range(len(test)):
        # predict the year
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        prediction.append(yhat_sequence)
        # get real observation and add to history for predicting the next year
        history.append(test[i, :])
    # get array of predictions
    prediction = np.array(prediction)
    prediction = np.ravel(prediction)
    # get array of actual values from test set
    actual = test[:, :, 0]
    actual[actual == 0] = np.nanmean(actual)
    actual = np.ravel(actual)
    # print test parameters
    print('Epochs: %d Batch Size: %d Filters: %d Kernels: %d Pool Size: %d Dense 1: %d Dense 2: %d'
          % (epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2))
    # calaculate and print scores
    scores = calculate_various_scores(actual, prediction)
    rmse, mape = calculate_scores(actual, prediction)
    print('RMSE: %.3f' % rmse)
    print('MAPE: %.3f' % mape)
    # plot prediction plot
    save_prediction_plot(actual, prediction, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2)
    # save result summary
    result_summary = [epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2, rmse, mape]
    save_results('temp/multihead_gridsearch_result_summary.csv', result_summary)
    # save all results incl prediction values and various scores
    result_all = [epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2, scores, prediction]
    save_results('temp/multihead_gridsearch_results.csv', result_all)
    # clear keras model
    K.clear_session()


def gridsearch_model(train, test, n_input, runs, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2):
    # fit model
    i = 0
    while i < runs:
        for epoch in epochs:
            for batch in batch_size:
                for filter in filters:
                    for kernel in kernel_size:
                        for pool in pool_size:
                            for d1 in dense1:
                                for d2 in dense2:
                                    evaluate_model(train, test, n_input, epoch, batch, filter, kernel, pool, d1, d2)
        i += 1


# load the data from pickle file for Branch 1 to 6, preprocessed in Preprocessing.py
df = pd.read_pickle("data/preprocessed/branch1.pkl")
# create dataframe with netto sales, month, weekday, year
df = pd.DataFrame(data=df.values, index=df.index, columns=['netto'])
df = df.assign(month=df.index.month)
df = df.assign(weekday=df.index.weekday)
df = df.assign(year=df.index.year)
# split into train and test
train, test = split_dataset(df.values, 365)
# evaluate model and get scores
n_input = 365
# define parameters for grid search
runs = 50
epochs = [10, 20, 30, 50]  # removed 5 from christmas test
batch_size = [4, 8, 16]  # 32, 64
filters = [32]  # 64, 128, 256
kernel_size = [8]  # 10, 12
pool_size = [4]  # 5, 6, 10
dense1 = [500]  # 100, 150, 200
dense2 = [200]  # 100, 150, 500
# hyperparameter gridsearch
gridsearch_model(train, test, n_input, runs, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2)
