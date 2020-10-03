import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
from utils import calculate_scores, plot_prediction, split_dataset, to_supervised, forecast

seed = 49
np.random.seed(seed)


def evaluate_model(train, test, model):
    n_input = 365
    # fit model
    # model = build_model(train)
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
    # plot prediction
    plot_prediction(actual, prediction)
    # clear keras model
    K.clear_session()
    return mape


def transfer(source_model, target_path, frozen_layer):
    # load data of branch
    df = pd.read_pickle(target_path)
    # create dataframe with netto sales, month, weekday, year
    df = pd.DataFrame(data=df.values, index=df.index, columns=['netto'])
    df = df.assign(month=df.index.month)
    df = df.assign(weekday=df.index.weekday)
    df = df.assign(year=df.index.year)
    # split into train and test
    train, test = split_dataset(df.values, 365)
    # prepare input data for branch
    n_input = 365
    train_x, train_y = to_supervised(train, n_input, 365)

    # load pre-trained model of source branch as base model
    base_model = load_model(source_model)
    # freeze specific layers of base model
    for layer in base_model.layers[:frozen_layer]:
        layer.trainable = False
    print("frozen layers: " + str(frozen_layer))

    # compile the model
    base_model.compile(loss='mse', optimizer='adam')

    # fit base_model with new data from branch
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    base_model.fit(input_data, train_y, epochs=20, batch_size=16, verbose=0)
    # evaluate fitted model
    mape = evaluate_model(train, test, base_model)
    return mape


# Goal: try using pretrained model of source branch to improve predictions for target branch
models = ['models/pretrained/branch1_cnn.h5',
          'models/pretrained/branch2_cnn.h5',
          'models/pretrained/branch3_cnn.h5',
          'models/pretrained/branch4_cnn.h5',
          'models/pretrained/branch5_cnn.h5',
          'models/pretrained/branch6_cnn.h5'
          ]

datasets = ["data/preprocessed/branch1.pkl",
            "data/preprocessed/branch2.pkl",
            "data/preprocessed/branch3.pkl",
            "data/preprocessed/branch4.pkl",
            "data/preprocessed/branch5.pkl",
            "data/preprocessed/branch6.pkl"]

# define number of layers to be frozen
frozen_layers = [32, 31, 30, 29, 28, 24, 20, 16, 12, 8, 4, 0]
# create list to store mape results
mape_values = []

# transfer base models for all branch datasets
for dataset in datasets:
    for model in models:
        print('base_model: ' + model + " Dataset: " + dataset)
        # try freezing different layers of base model
        layer_mapes = []
        for frozen_layer in frozen_layers:
            mape_val = transfer(model, dataset, frozen_layer)
            layer_mapes.append(mape_val)
        mape_values.append(layer_mapes)

print(mape_values)
