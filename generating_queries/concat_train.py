# %%
import pandas as pd
from os.path import join
import numpy as np

# DATABASE = join('./../../nuscenes_radar/7n5s_xy11', 'database.csv')
# TRAIN = join('./../../nuscenes_radar/7n5s_xy11', 'train.csv')

DATABASE = join('./../../nuscenes_radar/7n5s_xy11', 'database.csv')
TRAIN = join('./../../nuscenes_radar/7n5s_xy11', 'train.csv')

database = pd.read_csv(DATABASE).values
train = pd.read_csv(TRAIN).values
database_train = np.vstack((database, train))
dataframe = pd.DataFrame({'index': np.int32(database_train[:, 0]), 'x': database_train[:, 1], 'y': database_train[:, 2]})
dataframe.to_csv(DATABASE.replace('database', 'database_train'), index=False, sep=',')
