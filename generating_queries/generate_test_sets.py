# Code taken from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        # df_database
        df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, 'database.csv'), sep=',')
        df_locations = df_locations.rename(columns={'index': 'file'})
        df_locations = df_locations.rename(columns={'x': 'northing'})
        df_locations = df_locations.rename(columns={'y': 'easting'})
        for index, row in df_locations.iterrows():
            row['file'] = '/home/kaiwen/Documents/github/MinkLoc3D_ws/nuscenes_radar/' + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(row['file']))
            df_database = df_database.append(row, ignore_index=True)

        # df_database
        df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, 'test.csv'), sep=',')
        df_locations = df_locations.rename(columns={'index': 'file'})
        df_locations = df_locations.rename(columns={'x': 'northing'})
        df_locations = df_locations.rename(columns={'y': 'easting'})
        for index, row in df_locations.iterrows():
            row['file'] = '/home/kaiwen/Documents/github/MinkLoc3D_ws/nuscenes_radar/' + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(row['file']))
            df_test = df_test.append(row, ignore_index=True)
        
        # df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, filename), sep=',')
        # # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # # df_locations=df_locations.rename(columns={'timestamp':'file'})
        # for index, row in df_locations.iterrows():
        #     #entire business district is in the test set
        #     if (output_name == "business"):
        #         df_test = df_test.append(row, ignore_index=True)
        #     elif (check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
        #         df_test = df_test.append(row, ignore_index=True)
        #     df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        database_trees.append(database_tree)    # double append to simulate cross-trajectory database vs. test query
        test_trees.append(test_tree)
        test_trees.append(test_tree)            # double append to simulate cross-trajectory database vs. test query

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}

        # database
        df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, 'database.csv'), sep=',')
        df_locations = df_locations.rename(columns={'index': 'file'})
        df_locations = df_locations.rename(columns={'x': 'northing'})
        df_locations = df_locations.rename(columns={'y': 'easting'})
        for index, row in df_locations.iterrows():
            row['file'] = '/home/kaiwen/Documents/github/MinkLoc3D_ws/nuscenes_radar/' + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(row['file']))
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}

        # test
        df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, 'test.csv'), sep=',')
        df_locations = df_locations.rename(columns={'index': 'file'})
        df_locations = df_locations.rename(columns={'x': 'northing'})
        df_locations = df_locations.rename(columns={'y': 'easting'})
        for index, row in df_locations.iterrows():
            row['file'] = '/home/kaiwen/Documents/github/MinkLoc3D_ws/nuscenes_radar/' + folder + pointcloud_fols + '{:0>5d}.bin'.format(int(row['file']))
            test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}



        # df_locations = pd.read_csv(os.path.join(BASE_DIR, base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        # df_locations = df_locations.rename(columns={'timestamp': 'file'})
        # for index, row in df_locations.iterrows():
        #     #entire business district is in the test set
        #     if (output_name == "business"):
        #         test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
        #     elif (check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
        #         test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
        #     database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
        database_sets.append(database)
        database_sets.append(database)
        test_sets.append(test)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if (i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=9)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, output_name + '_evaluation_query.pickle')


# Building database and query files for evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "../../nuscenes_radar/"

# For Oxford
folders = []
runs_folder = "./"
all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
index_list = [0]
print(len(index_list))
for index in index_list:
    folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pcl/", "pointcloud_locations_20m.csv", "oxford")

# #For University Sector
# folders = []
# runs_folder = "inhouse_datasets/"
# all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
# uni_index = range(10, 15)
# for index in uni_index:
#     folders.append(all_folders[index])

# print(folders)
# construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["university"], "university")

# #For Residential Area
# folders = []
# runs_folder = "inhouse_datasets/"
# all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
# res_index = range(5, 10)
# for index in res_index:
#     folders.append(all_folders[index])

# print(folders)
# construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["residential"], "residential")

# #For Business District
# folders = []
# runs_folder = "inhouse_datasets/"
# all_folders = sorted(os.listdir(os.path.join(BASE_DIR, base_path, runs_folder)))
# bus_index = range(5)
# for index in bus_index:
#     folders.append(all_folders[index])

# print(folders)
# construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/", "pointcloud_centroids_25.csv", p_dict["business"], "business")
