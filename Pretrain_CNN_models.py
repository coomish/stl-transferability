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
# from keras import backend as K
from utils import calculate_scores, to_supervised, forecast, split_dataset

seed = 49
np.random.seed(seed)


# train the model
def build_model(train):
    # model parameters
    n_input = 365
    epochs = 20
    batch_size = 16
    filters = 64
    kernel_size = 8
    pool_size = 4
    dense_1 = 500
    dense_2 = 200
    # prepare data
    train_x, train_y = to_supervised(train, n_input, 365)
    # define parameters
    verbose = 2
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
    # model.summary() # added
    return model


def evaluate_model(train, test):
    n_input = 365
    # fit model
    model = build_model(train)
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
    # calaculate and print scores
    rmse, mape = calculate_scores(actual, prediction)
    print('RMSE: %.3f' % rmse)
    print('MAPE: %.3f' % mape)
    # clear keras model
    # K.clear_session()
    return model


def pretrain_model(path):
    df = pd.read_pickle(path)
    # create dataframe with netto sales, month, weekday, year
    df = pd.DataFrame(data=df.values, index=df.index, columns=['netto'])
    df = df.assign(month=df.index.month)
    df = df.assign(weekday=df.index.weekday)
    df = df.assign(year=df.index.year)
    # split into train and test
    train, test = split_dataset(df.values, 365)
    # evaluate model and get scores
    model = evaluate_model(train, test)
    # save model
    model.save("models/" + path + ".h5")
    del model


paths = ["data/preprocessed/branch1.pkl",
         "data/preprocessed/branch2.pkl",
         "data/preprocessed/branch3.pkl",
         "data/preprocessed/branch4.pkl",
         "data/preprocessed/branch5.pkl",
         "data/preprocessed/branch6.pkl"]

for path in paths:
    pretrain_model(path)
