import csv
import pandas as pd
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
from numpy import array, split, mean
import numpy
import seaborn as sns


# CNN relevant methods:

# split a univariate dataset into train/test sets
def split_dataset(data, split_size):
    # split into standard years
    train, test = data[:-split_size], data[-split_size:]
    # restructure into windows of yearly data
    train = array(split(train, len(train) / split_size))
    test = array(split(test, len(test) / split_size))
    return train, test


# restructure into windows of yearly data
def restructure_dataset(train, test):
    # restructure into windows of yearly data
    train = array(split(train, len(train) / 365))
    test = array(split(test, len(test) / 365))
    return train, test


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into n input arrays
    input_x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
    # forecast the next year
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# make a forecast for multichannel cnn
def forecast_multichannel(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat


# utility functions:

def mape(y_true, y_pred):
    y_true, y_pred = array(y_true), array(y_pred)
    return mean(numpy.abs((y_true - y_pred) / y_true)) * 100


def print_calculated_scores(actual, prediction):
    # calculate RMSE
    rmse = sqrt(mean_squared_error(actual, prediction))
    # calculate MAPE
    mape_score = mape(actual, prediction)
    # print both scores
    print('RMSE: %.3f' % rmse)
    print('MAPE : %3f' % mape_score)


def plot_prediction(actual, prediction):
    sns.set()
    fig = pyplot.figure(figsize=(18, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(actual, color='blue', label='actual')
    ax.plot(prediction, color='red', label='prediction')
    ax.set_ylabel('daily netto revenue')
    ax.set_xlabel('days')
    ax.set_title('CNN')
    ax.legend()
    fig.show()


def save_plot(actual, prediction, filename):
    fig = pyplot.figure(figsize=(18, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(actual, color='blue', label='actual')
    ax.plot(prediction, color='red', label='prediction')
    ax.set_ylabel('daily netto revenue')
    ax.set_xlabel('days')
    ax.set_title('CNN')
    ax.legend()
    fig.savefig('temp/' + str(filename) + '.png')
    pyplot.close()


def calculate_scores(actual, prediction):
    mape12 = mape(actual, prediction)
    rmse = sqrt((mean_squared_error(actual, prediction)))
    return rmse, mape12


def calculate_various_scores(actual, prediction):
    # calculate RMSE scores
    rmse1 = sqrt(mean_squared_error(actual[:31], prediction[:31]))
    rmse3 = sqrt(mean_squared_error(actual[:90], prediction[:90]))
    rmse6 = sqrt(mean_squared_error(actual[:182], prediction[:182]))
    rmse12 = sqrt(mean_squared_error(actual, prediction))
    # calculate MAPE
    mape1 = mape(actual[:31], prediction[:31])
    mape3 = mape(actual[:90], prediction[:90])
    mape6 = mape(actual[:182], prediction[:182])
    mape12 = mape(actual, prediction)
    # create df for scores
    df = pd.DataFrame({'1 month': [rmse1, mape1], '3 months': [rmse3, mape3],
                       '6 months': [rmse6, mape6], '12 months': [rmse12, mape12]}, index=['RMSE', 'MAPE'])
    return df


def plot_seaborn(actual, prediction):
    return actual, prediction


# plot training history
def plot_history(history):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('loss', y=0, loc='center')
    pyplot.legend()
    # plot rmse
    pyplot.subplot(2, 1, 2)
    pyplot.plot(history.history['rmse'], label='train')
    pyplot.plot(history.history['val_rmse'], label='test')
    pyplot.title('rmse', y=0, loc='center')
    pyplot.legend()
    pyplot.show()


def save_prediction_plot(actual, prediction, epochs, batch_size, filters, kernel_size, pool_size, dense1, dense2):
    sns.set()
    pyplot.plot(actual, color='blue', label='actual')
    pyplot.plot(prediction, color='red', label='prediction')
    pyplot.ylabel('daily netto revenue')
    pyplot.xlabel('day of 2017')
    pyplot.title('CNN model')
    pyplot.legend()
    pyplot.savefig('temp/epochs_' + str(epochs) +
                   '_batchsize_' + str(batch_size) +
                   '_filters_'+str(filters) +
                   '_kernels_'+str(kernel_size) +
                   '_poolsize_'+str(pool_size) +
                   '_dense1_'+str(dense1) +
                   '_dense2_'+str(dense2) + '.png')
    pyplot.close()


def save_results(path, results):
    with open(path, 'a') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(results)

