import os, sys
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
    print(activation1[-13:-8] + activation2[-14:-8] + " Mean Model Correlation:", mean_model_corr)
    return mean_model_corr


# list of convolutional layers for which we can calculate SVCCA
conv_layers = [4, 5, 6, 7, 8, 9, 10, 11]

# list of target branch activations on target branch input data
act_targets = ['activations/weekly_start2014/m1_x1_weekly/',
               'activations/weekly_start2014/m2_x2_weekly/',
               'activations/weekly_start2014/m3_x3_weekly/',
               'activations/weekly_start2014/m4_x4_weekly/',
               'activations/weekly_start2014/m5_x5_weekly/',
               'activations/weekly_start2014/m6_x6_weekly/']

# list to store mean model correlations
model_correlations = []

# list of activations on target branch input data
act1_list = ['activations/weekly_start2014/m1_x1_weekly/',
             'activations/weekly_start2014/m2_x1_weekly/',
             'activations/weekly_start2014/m3_x1_weekly/',
             'activations/weekly_start2014/m4_x1_weekly/',
             'activations/weekly_start2014/m5_x1_weekly/',
             'activations/weekly_start2014/m6_x1_weekly/']

for act1 in act1_list:
    cor1 = calaculate_mean_model_correlation(act_targets[0], act1, conv_layers)
    model_correlations.append(cor1)

act2_list = ['activations/weekly_start2014/m1_x2_weekly/',
             'activations/weekly_start2014/m2_x2_weekly/',
             'activations/weekly_start2014/m3_x2_weekly/',
             'activations/weekly_start2014/m4_x2_weekly/',
             'activations/weekly_start2014/m5_x2_weekly/',
             'activations/weekly_start2014/m6_x2_weekly/']

for act2 in act2_list:
    cor2 = calaculate_mean_model_correlation(act_targets[1], act2, conv_layers)
    model_correlations.append(cor2)

act3_list = ['activations/weekly_start2014/m1_x3_weekly/',
             'activations/weekly_start2014/m2_x3_weekly/',
             'activations/weekly_start2014/m3_x3_weekly/',
             'activations/weekly_start2014/m4_x3_weekly/',
             'activations/weekly_start2014/m5_x3_weekly/',
             'activations/weekly_start2014/m6_x3_weekly/']

for act3 in act3_list:
    cor3 = calaculate_mean_model_correlation(act_targets[2], act3, conv_layers)
    model_correlations.append(cor3)

act4_list = ['activations/weekly_start2014/m1_x4_weekly/',
             'activations/weekly_start2014/m2_x4_weekly/',
             'activations/weekly_start2014/m3_x4_weekly/',
             'activations/weekly_start2014/m4_x4_weekly/',
             'activations/weekly_start2014/m5_x4_weekly/',
             'activations/weekly_start2014/m6_x4_weekly/']

for act4 in act4_list:
    cor4 = calaculate_mean_model_correlation(act_targets[3], act4, conv_layers)
    model_correlations.append(cor4)

act5_list = ['activations/weekly_start2014/m1_x5_weekly/',
             'activations/weekly_start2014/m2_x5_weekly/',
             'activations/weekly_start2014/m3_x5_weekly/',
             'activations/weekly_start2014/m4_x5_weekly/',
             'activations/weekly_start2014/m5_x5_weekly/',
             'activations/weekly_start2014/m6_x5_weekly/']

for act5 in act5_list:
    cor5 = calaculate_mean_model_correlation(act_targets[4], act5, conv_layers)
    model_correlations.append(cor5)

act6_list = ['activations/weekly_start2014/m1_x6_weekly/',
             'activations/weekly_start2014/m2_x6_weekly/',
             'activations/weekly_start2014/m3_x6_weekly/',
             'activations/weekly_start2014/m4_x6_weekly/',
             'activations/weekly_start2014/m5_x6_weekly/',
             'activations/weekly_start2014/m6_x6_weekly/']

for act6 in act6_list:
    cor6 = calaculate_mean_model_correlation(act_targets[5], act6, conv_layers)
    model_correlations.append(cor6)

# transform into array and split it into array of 6
model_correlations_array = np.array(model_correlations)
model_correlations_array = np.split(model_correlations_array, 6)
# transform into data frame
df = pd.DataFrame(model_correlations_array)
# restructure/transpose data frame so it has the right shape
df = df.T
# save data frame to /temp folder
df.to_csv('../temp/net_similarity_weekly_start2014.csv', index=False)
