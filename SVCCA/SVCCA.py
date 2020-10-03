import os, sys
from matplotlib import pyplot as plt
import numpy as np
# sys.path.append("..")
import cca_core


def _plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def SVCCA(activations1, activations2, layer_number):
    # SVCCA different x
    # print("Results using SVCCA keeping 60 dims")
    # load activations
    acts1 = np.genfromtxt(activations1 + str(layer_number) + '.csv', delimiter=',')
    acts2 = np.genfromtxt(activations2 + str(layer_number) + '.csv', delimiter=',')

    # Mean subtract activations
    cacts1 = acts1  # - np.mean(acts1, axis=0, keepdims=True)
    cacts2 = acts2  # - np.mean(acts2, axis=0, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:60] * np.eye(60), V1[:60])  # default: np.dot(s1[:20]*np.eye(20), V1[:20]), 49
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:60] * np.eye(60), V2[:60])  # default: np.dot(s2[:20]*np.eye(20), V2[:20]), 49
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
    print(activation1[-6:] + activation2[-6:-1] + " Mean Model Correlation:", mean_model_corr)


# list of convolutional layers for which we can calculate SVCCA
conv_layers = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
               23]

act1_target = 'activations/m1_x1/'
act1_list = ['activations/m1_x1/',
             'activations/m2_x1/',
             'activations/m3_x1/',
             'activations/m4_x1/',
             'activations/m5_x1/',
             'activations/m6_x1/']
for act1 in act1_list:
    calaculate_mean_model_correlation(act1_target, act1, conv_layers)

act2_target = 'activations/m2_x2/'
act2_list = ['activations/m1_x2/',
             'activations/m2_x2/',
             'activations/m3_x2/',
             'activations/m4_x2/',
             'activations/m5_x2/',
             'activations/m6_x2/']
for act2 in act2_list:
    calaculate_mean_model_correlation(act2_target, act2, conv_layers)

act3_target = 'activations/m3_x3/'
act3_list = ['activations/m1_x3/',
             'activations/m2_x3/',
             'activations/m3_x3/',
             'activations/m4_x3/',
             'activations/m5_x3/',
             'activations/m6_x3/']
for act3 in act3_list:
    calaculate_mean_model_correlation(act3_target, act3, conv_layers)

act4_target = 'activations/m4_x4/'
act4_list = ['activations/m1_x4/',
             'activations/m2_x4/',
             'activations/m3_x4/',
             'activations/m4_x4/',
             'activations/m5_x4/',
             'activations/m6_x4/']
for act4 in act4_list:
    calaculate_mean_model_correlation(act4_target, act4, conv_layers)

act5_target = 'activations/m5_x5/'
act5_list = ['activations/m1_x5/',
             'activations/m2_x5/',
             'activations/m3_x5/',
             'activations/m4_x5/',
             'activations/m5_x5/',
             'activations/m6_x5/']
for act5 in act5_list:
    calaculate_mean_model_correlation(act5_target, act5, conv_layers)

act6_target = 'activations/m6_x6/'
act6_list = ['activations/m1_x6/',
             'activations/m2_x6/',
             'activations/m3_x6/',
             'activations/m4_x6/',
             'activations/m5_x6/',
             'activations/m6_x6/']
for act6 in act6_list:
    calaculate_mean_model_correlation(act6_target, act6, conv_layers)
