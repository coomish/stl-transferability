# multi-channel multi-step CNN
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from keras.utils import plot_model
import numpy as np
from utils import calculate_scores, save_plot, to_supervised, split_dataset, forecast_multichannel, plot_prediction

seed = 49
np.random.seed(seed) # fix random seed for reproducibility


# train the multichannel cnn model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, 365)
    # define parameters
    epochs, batch_size, verbose = 20, 8, 0
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    # compile model
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # plot the model
    # plot_model(model, show_shapes=True, to_file='plots/model_plot_multichannel.png')
    return model


# evaluate the model performance
def evaluate_model(train, test, n_input, save_model_path, save_plot_path):
    # fit model
    model = build_model(train, n_input)
    # history is a list of yearly data
    history = [x for x in train]
    # walk-forward validation over period
    predictions = list()
    for i in range(len(test)):
        # predict the period
        yhat_sequence = forecast_multichannel(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next period
        history.append(test[i, :])
    # get array of predictions
    prediction = np.array(predictions)
    prediction = np.ravel(prediction)
    # get array of actual values from test set
    actual = test[:, :, 0]
    actual_plot = np.ravel(actual)
    actual[actual == 0] = np.nanmean(actual)
    actual = np.ravel(actual)
    # calaculate and print scores
    rmse, mape = calculate_scores(actual, prediction)
    print('RMSE: %.3f' % rmse)
    print('MAPE: %.3f' % mape)
    # save plot in /temp/ path as png file
    save_plot(actual_plot, prediction, save_plot_path)
    plot_prediction(actual_plot, prediction)
    # save model
    model.save(save_model_path)


# load the data from pickle file for Branch 1 to 6, preprocessed in Preprocessing.py
df = pd.read_pickle("data/preprocessed/branch5.pkl")
# create dataframe with netto sales, month, weekday, year
df = pd.DataFrame(data=df.values, index=df.index, columns=['netto'])
df = df.assign(month=df.index.month)
df = df.assign(weekday=df.index.weekday)
df = df.assign(year=df.index.year)
# split into train and test set
train, test = split_dataset(df.values, 365)
# evaluate model and get scores
n_input = 365
save_model_path = 'temp/branch5_cnn_multichannel.h5'
save_plot_path = 'multichannel_CNN_branch5'
evaluate_model(train, test, n_input, save_model_path, save_plot_path)
