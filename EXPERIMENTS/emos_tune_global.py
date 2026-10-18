import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from DATA.data_loading import gen_dataset
from DATA.data_loading import load_to_memory
from EMOS.emos_model import EMOS
from tqdm import tqdm


# Workaround for Tensorflow and Keras compatibility issue
os.environ["TF_USE_LEGACY_KERAS"] = "1"



# Data and result folders
nobackup    = 'Data/'
train_data_folder = nobackup + 'train_data/norm_train_data/'
test_data_folder = nobackup + 'test_data/norm_test_data/'

result_folder = 'Results/Validation/Emos/'


# Load station information
with open('Data/st-info.pkl', 'rb') as f:
    info = pkl.load(f)


# Define variables and stations to use
var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']
omit = ['229', '285', '323']
sts  = [st for st in info.keys() if st not in omit]


# Define fixed parameters for training
T = 49
num_var = 5
gridsize = 9 # Optimal grid size determined through grid size selection
reduce_grid = True
timesteps = 4000
log_timesteps = 20
log_step = 200




# If true, then model will be trained on the training set and evaluated on the validation set.
# True for hyperparameter tuning.
validation = True


# Load training and validation/test data
if validation:

    val_files  = [train_data_folder + file for file in os.listdir(train_data_folder) if (('20151' in file or '20160' in file) and ('pkl' in file))]
    
    val_dataset = gen_dataset(val_files, sts=sts, var=var_list, grid_size=gridsize)
    val_tag, val_S2, val_var, val_obs = load_to_memory(val_dataset, grid_size=gridsize, grid_S2=reduce_grid)


    train_files  = [train_data_folder + file for file in os.listdir(train_data_folder) if ((train_data_folder + file not in val_files) and ('.pkl' in file))]

    train_dataset = gen_dataset(train_files, sts=sts, var=var_list, grid_size=gridsize)
    tag, S2, var, obs = load_to_memory(train_dataset, grid_size=gridsize, grid_S2=reduce_grid)


    test_files  = [test_data_folder + file for file in os.listdir(test_data_folder) if '.pkl' in file]

    test_dataset = gen_dataset(test_files, sts=sts, var=var_list, grid_size=gridsize)
    test_tag, test_S2, test_var, test_obs = load_to_memory(test_dataset, grid_size=gridsize, grid_S2=reduce_grid)


else:

    train_files  = [train_data_folder + file for file in os.listdir(train_data_folder) if 'pkl' in file]

    train_dataset = gen_dataset(train_files, sts=sts, var=var_list, grid_size=gridsize)
    tag, S2, var, obs = load_to_memory(train_dataset, grid_size=gridsize, grid_S2=reduce_grid)


    test_files  = [test_data_folder + file for file in os.listdir(test_data_folder) if 'pkl' in file]

    test_dataset = gen_dataset(test_files, sts=sts, var=var_list, grid_size=gridsize)
    test_tag, test_S2, test_var, test_obs = load_to_memory(test_dataset, grid_size=gridsize, grid_S2=reduce_grid)



fitting_loss = []
validation_loss = []
labels = []


# Train and evaluate the model for all combinations of learning rates and gradient clipping values.
# Hyperparameters for final model training will be selected through visual inspection of the training and validation losses.
for i, lr in enumerate([0.001, 0.005, 0.01, 0.05, 0.1]):

    for j, cv in enumerate([None, 0.5, 1.0, 1.5]):

        emos   = EMOS(num_var, T)
        emos.opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=cv)

        cv_loss = []
        cv_val_loss = []

        for k in tqdm(range(log_timesteps)):
            loss = emos.fit(var, S2, obs, steps=log_step)
            cv_loss.append(loss[len(loss)-1])
            cv_val_loss.append(np.mean(emos.loss(val_var, val_S2, val_obs)))

            emos_params = emos.get_params()
            with open(result_folder + 'GlobalEmos_' + str(i) + str(j) + str(k) + '.pkl', 'wb') as f:
                pkl.dump(emos_params,f)
            

        fitting_loss.append(cv_loss)
        validation_loss.append(cv_val_loss)
        labels.append('LR = '+str(lr)+', CV = '+str(cv))

        emos_losses = {
            'Fitting Loss': fitting_loss,
            'Validation Loss': validation_loss,
            'Labels': labels
        }

        with open(result_folder + 'GlobalEmos_losses.pkl', 'wb') as f:
            pkl.dump(emos_losses,f)
        
    











