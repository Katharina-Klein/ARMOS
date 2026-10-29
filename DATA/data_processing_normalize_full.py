##### This code was almost entirely written by Daniel
##### Normalizes all values including the wind speed

import os
import pickle as pkl
import tensorflow as tf
import numpy as np
from data_loading import gen_dataset
from data_loading import load_to_memory
import json
from tqdm import tqdm


# Data folders
nobackup    = 'Data/'
data_folder = nobackup + 'train_data/'                                             # training data 2015 - 2017
norm_data_folder = data_folder + 'norm_train_data/'                                     # training data 2015 - 2017
                                      

    
# Get files
files = [file for file in os.listdir(data_folder) if '.pkl' in file]
data_files = [data_folder + file for file in files]



# Station information (file created by Daniel)
with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)


# Define variables and stations to use
omit = ['229', '285', '323']
var_list = ['WIND850', 'WIND925', 'PRESS', 'KINETIC', 'HUMID', 'GEOPOT']

# Build datasets
sts  = [st for st in st_info.keys() if st not in omit]

dataset = gen_dataset(data_files, sts=sts, var=var_list, grid_size=1)
_, _, variables, _ = load_to_memory(dataset, grid_size=1, grid_S2=True, add_wind10=False)


grid = np.zeros([len(data_files),49,300,300])

for i, file in enumerate(data_files):
    with open(file, 'rb') as f:
        data = pkl.load(f)

    grid[i] = data.pop('GRID')


##### Normalize training data and save mean and standard deviation values for each variable and the wind speed grid

mean_grid, std_grid = np.mean(grid), np.std(grid)
mean, std = tf.math.reduce_mean(variables, axis=(0,1)), tf.math.reduce_std(variables, axis=(0,1))


norm_values = {var : {'Mean' : mean[i].numpy(), 'Std' : std[i].numpy()} for i, var in enumerate(var_list)}
norm_values['WIND10'] = {'Mean' : mean_grid, 'Std' : std_grid}

print(norm_values)



# Save norm value list
with open(data_folder + 'normalization_values_new.json', 'w') as f:
    json.dump(str(norm_values), f)



# Normalize training data and save to new folder
for file in tqdm(files):
    with open(data_folder + file, 'rb') as f:
        data = pkl.load(f)
    data['GRID'] -= mean_grid
    data['GRID'] /= std_grid
    for key in data.keys():
        if not key == 'GRID':
            for i, var in enumerate(var_list):
                data[key][var] -= mean[i]
                data[key][var] /= std[i]
    with open(norm_data_folder + file,'wb') as f:
        pkl.dump(data,f)





##### Normalize test data with same mean and standard deviation as the training data


# Folder with stored test data
data_folder = nobackup + 'test_data/'                                              # test data 2018 - 2019
norm_data_folder = data_folder + 'norm_test_data/'                                      # test data 2018 - 2019


# Get files
files = [file for file in os.listdir(data_folder) if '.pkl' in file]


# Normalize test data and save to new folder
for file in tqdm(files):
    with open(data_folder + file, 'rb') as f:
        data = pkl.load(f)
    data['GRID'] -= mean_grid
    data['GRID'] /= std_grid
    for key in data.keys():
        if not key == 'GRID':
            for i, var in enumerate(var_list):
                data[key][var] -= mean[i]
                data[key][var] /= std[i]
    with open(norm_data_folder + file,'wb') as f:
        pkl.dump(data,f)


        


