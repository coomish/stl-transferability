import os, sys
from os import listdir
from os.path import isfile, join, isdir
from matplotlib import pyplot as plt
import numpy as np
# sys.path.append("..")
import cca_core
import pandas as pd


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def SVCCA(activations1, activations2, layer_number):
    # SVCCA different x
    # print("Results using SVCCA keeping 30 dims")
    # load activations
    acts1 = np.genfromtxt(activations1 + str(layer_number) + '.csv', delimiter=',')
    acts2 = np.genfromtxt(activations2 + str(layer_number) + '.csv', delimiter=',')

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=0, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=0, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:30] * np.eye(30), V1[:30])  # default: np.dot(s1[:20]*np.eye(20), V1[:20]), 49
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:30] * np.eye(30), V2[:30])  # default: np.dot(s2[:20]*np.eye(20), V2[:20]), 49
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)  # 1e-10
    # print("Layer Number:", layer_number)
    # print("SVCCA Correlation Coefficient:", np.mean(svcca_results["cca_coef1"]))
    return np.mean(svcca_results["cca_coef1"])  # , acts1, cacts1, U1, s1, V1, svacts1


def calaculate_mean_model_correlation(activation1, activation2, conv_layers):
    # list for storing layer correlations
    layer_corr = []
    # calculate and save SVCCA correlation between all layers of two base models
    for conv in conv_layers:
        corr = SVCCA(activation1, activation2, conv)  # , acts1, cacts1, U1, s1, V1, svacts1
        layer_corr.append(corr)
    # calculate mean model correlation of stored layer correlation
    mean_model_corr = np.mean(layer_corr)
    print(activation1[56:-8] + " " + activation2[61:-8] + " Mean Model Correlation:", mean_model_corr)
    return mean_model_corr


# list of convolutional layers for which we can calculate SVCCA
conv_layers = [4, 5, 6, 7, 8, 9, 10, 11]

# list of target branch activations on target branch input data
act_targets = ['activations/activations_crosswise_weekly_start2014/base/m1_x1_weekly/',
               'activations/activations_crosswise_weekly_start2014/base/m2_x2_weekly/',
               'activations/activations_crosswise_weekly_start2014/base/m3_x3_weekly/',
               'activations/activations_crosswise_weekly_start2014/base/m4_x4_weekly/',
               'activations/activations_crosswise_weekly_start2014/base/m5_x5_weekly/',
               'activations/activations_crosswise_weekly_start2014/base/m6_x6_weekly/']

# list to store mean model correlations
model_correlations = pd.DataFrame([])

# list of source model activations on target input data
activation_folder = 'activations/activations_crosswise_weekly_start2014/transfer1/'
act_source = sorted([f for f in listdir(activation_folder) if isdir(join(activation_folder, f))])

transfer_degree = 2

for target_act in act_targets:
    for source_act in act_source:
        # check if activations have the same data input x_i and model has not been trained on data input already
        if source_act[-9:-7] == target_act[-10:-8] and target_act[-9] not in source_act[:-9]:
            source_path = activation_folder + source_act + "/"
            mean_corr = calaculate_mean_model_correlation(target_act, source_path, conv_layers)
            model_correlations = model_correlations.append(pd.DataFrame([[target_act[56:-8], source_act[:-7], transfer_degree, mean_corr]], columns=['target_model', 'source_model', 'transfer_degree', 'mean_model_SVCCA_corr']))
        else:
            continue


# save data frame to /temp folder
model_correlations.to_csv('../temp/net_similarity_crosswise_TD1.csv', index=False)
