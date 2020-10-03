# multichannel multi-step cnn
import csv
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import seaborn as sns
from keras import backend as K
from utils import calculate_various_scores, forecast_multichannel, to_supervised, split_dataset, save_results

seed = 49
np.random.seed(seed) # fix random seed for reproducibility

# save plot method fpr multichannel cnn
def save_prediction_plot(actual, prediction, epochs, batch_size):
    sns.set()
    pyplot.plot(actual, color='blue', label='actual')
    pyplot.plot(prediction, color='red', label='prediction')
    pyplot.ylabel('daily netto revenue')
    pyplot.xlabel('day of 2017')
    pyplot.title('multichannel CNN')
    pyplot.legend()
    pyplot.savefig('temp/multichannel_epochs_' + str(epochs) + '_batchsize_' + str(batch_size) + '.png')
    pyplot.close()


# train the model
def build_model(train, n_input, epochs, batch_size):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, 365)
    # define parameters
    verbose = 0
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# evaluate a single model
def evaluate_model(train, n_input, epochs, batch_size):
    # fit model
    model = build_model(train, n_input, epochs, batch_size)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    prediction = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast_multichannel(model, history, n_input)
        # store the predictions
        prediction.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    # get array of predictions
    prediction = np.array(prediction)
    prediction = np.ravel(prediction)
    # get array of actual values from test set
    actual = test[:, :, 0].copy()
    actual[actual == 0] = np.nanmean(actual)
    actual = np.ravel(actual)
    # print test parameters
    print('Epochs: %d Batch Size: %d' % (epochs, batch_size))
    # calaculate and print scores
    scores = calculate_various_scores(actual, prediction)
    # save prediction plot
    save_prediction_plot(actual, prediction, epochs, batch_size)
    # save results
    result = [epochs, batch_size, scores, prediction]
    save_results('temp/multichannel_gridsearch_summary.csv', result)
    # clear keras model
    K.clear_session()


def gridsearch_model(train, n_input, epochs, batch_size):
    # fit model
    for epoch in epochs:
        for batch in batch_size:
            evaluate_model(train, n_input, epoch, batch)


# load the data from pickle file for Branch 1 to 6
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
epochs = [5, 10, 20, 30]
batch_size = [8, 16, 32]
# hyperparameter gridsearch
gridsearch_model(train, n_input, epochs, batch_size)
