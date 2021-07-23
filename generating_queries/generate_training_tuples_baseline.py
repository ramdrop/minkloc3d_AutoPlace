# Code taken from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "../../nuscenes_radar/"

runs_folder = "./"
filename = "database_train.csv"
pointcloud_fols = "/pcl/"

all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))

folders = []

#All runs are used for training (both full and partial)
index_list = range(len(all_folders))
print("Number of runs: " + str(len(index_list)))
for index in index_list:
    folders.append(all_folders[index])
print(folders)

##########################################


def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=9)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=18)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query": query, "positives": positives, "negatives": negatives}

    with open(os.path.join(base_path, filename), 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


####Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])

for folder in folders:
    df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, filename), sep=',')
    # df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(df_locations['index']))
    df_locations = df_locations.rename(columns={'index': 'file'})
    df_locations = df_locations.rename(columns={'x': 'northing'})
    df_locations = df_locations.rename(columns={'y': 'easting'})

    for index, row in df_locations.iterrows():
        row['file'] = '../../nuscenes_radar/' + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(row['file']))
        df_train = df_train.append(row, ignore_index=True)

print("Number of training submaps: " + str(len(df_train['file'])))
# print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
construct_query_dict(df_train, "training_queries_baseline.pickle")
# construct_query_dict(df_test, "test_queries_baseline.pickle")
