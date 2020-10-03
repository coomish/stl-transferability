import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns

from dtaidistance import dtw, dtw_visualisation as dtw_vis
from dictances import cosine, euclidean, canberra, kullback_leibler, hellinger, bhattacharyya
import scipy.spatial.distance as sdist
from statsmodels.tsa.seasonal import seasonal_decompose

# define time series origin for branches
data_sets = ["../data/preprocessed/branch1.pkl",
              "../data/preprocessed/branch2.pkl",
              "../data/preprocessed/branch3.pkl",
              "../data/preprocessed/branch4.pkl",
              "../data/preprocessed/branch5.pkl",
              "../data/preprocessed/branch6.pkl"]


distances = []
start = '2017-01-01'

branches = [pd.read_pickle(data_sets[0]).loc[start:],
pd.read_pickle(data_sets[1]).loc[start:],
pd.read_pickle(data_sets[2]).loc[start:],
pd.read_pickle(data_sets[3]).loc[start:],
pd.read_pickle(data_sets[4]).loc[start:],
pd.read_pickle(data_sets[5]).loc[start:]]


#
# plt.figure();
# branch1.plot(style='r', label='Branch1');
# branch2.plot(style='b', label='Branch2');
# plt.legend()
# plt.show()

# components_branch1 = seasonal_decompose(branch1, model='additive')
# components_branch2 = seasonal_decompose(branch2, model='additive')

# calculate distances
cosine_distances = []
canberra_distances = []
euklidean_distances = []
for branch_a in branches:
    cosine_dist_row = []
    canberra_dist_row = []
    euclidean_dist_row = []
    for branch_b in branches:
        cosine_dist_row.append(sdist.cosine(branch_a, branch_b))
        canberra_dist_row.append(sdist.canberra(branch_a, branch_b))
        euclidean_dist_row.append(sdist.euclidean(branch_a, branch_b))
    cosine_distances.append(cosine_dist_row)
    canberra_distances.append(canberra_dist_row)
    euklidean_distances.append(euclidean_dist_row)
print(distances)

sns.set(font_scale=1.1)

f, ax = plt.subplots(figsize=(9, 8))
ax.set_title('cosine distance', fontsize=26)
sns.heatmap(cosine_distances, annot=True, fmt="f", linewidths=.5, ax=ax)
plt.show()

f, ax = plt.subplots(figsize=(9, 8))
ax.set_title('canberra distance', fontsize=26)
sns.heatmap(canberra_distances, annot=True, fmt="f", linewidths=.5, ax=ax)
plt.show()

f, ax = plt.subplots(figsize=(9, 8))
ax.set_title('euklidean distance', fontsize=26)
sns.heatmap(euklidean_distances, annot=True, fmt="f", linewidths=.5, ax=ax)
plt.show()

# print(dtw.distance(branch1, branch2))


# path = dtw.warping_path(branch1,branch2)
# dtw_vis.plot_warping(branch1, branch2, path, filename="warp.png")
#
# d, paths = dtw.warping_paths(branch1, branch2, window=25, psi=2)
# best_path = dtw.best_path(paths)
# dtw_vis.plot_warpingpaths(branch1, branch2, paths, best_path, filename="best.png")

#print(euclidean(branch1.to_dict(), branch2.to_dict()))
#print(canberra(branch1.to_dict(), branch2.to_dict()))
# print(kullback_leibler(branch1.to_dict(), branch2.to_dict()))
# print(hellinger(branch1.to_dict(), branch2.to_dict()))
# print(bhattacharyya(branch1.to_dict(), branch2.to_dict()))
# >>> 0.52336690346601
#
# print(euclidean(branch1, branch2))
# # >>> 15119.400349404095
#
# print(canberra(branch1, branch2))
# # >>> 624.9088876554047

# for set1 in data_sets:
#     for set2 in data_sets:
#         distance = dtw_distance(set1, set2)
#         distances.append(distance)
