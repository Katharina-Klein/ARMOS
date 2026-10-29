import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from DATA.data_loading import gen_dataset
from DATA.data_loading import load_to_memory
from ARMOS.armos_model import ARMOS_LS as ARMOS
from RESULTS.scores import log_score_armos
from tqdm import tqdm


# Workaround for Tensorflow and Keras compatibility issue
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# Data and result folders
nobackup    = 'Data/'
train_data_folder = nobackup + 'train_data/norm_train_data/'
test_data_folder = nobackup + 'test_data/norm_test_data/'

result_folder = 'Results/Final/ArmosLS/'


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

# Learning rates and gradient clipping values chosen through hyperparameter tuning, for each of the models with p = 1,2,3
lr = [0.01, 0.01, 0.01]
cv = [1.5, 0.5, 1.0]


# If false, then model will be trained on the training and validationset and evaluated on the test set.
# False for final model training.
validation = False


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






for j, p in enumerate([1, 2, 3]):

    

    fitting_loss = {}
    test_loss = {}
    log_score_test = {}
    log_score_fitting = {}
    errors = []


    print('Fitting Global')

    # Initialize the ARMOS model and define the optimizer.
    armos   = ARMOS(num_var, p, T)
    armos.opt = tf.keras.optimizers.Adam(learning_rate=lr[j], clipvalue=cv[j])

    cv_loss = []
    cv_test_loss = []
    cv_ls = []
    cv_test_ls = []

    # Train the ARMOS model and store the training and test losses for each logging step.
    for k in tqdm(range(log_timesteps)):
        loss = armos.fit(var, S2, obs, steps=log_step)
        cv_loss.append(loss[len(loss)-1])
        cv_test_loss.append(np.mean(armos.loss(test_var, test_S2, test_obs)))
        cv_ls.append(np.mean(log_score_armos(armos, var, S2, obs), axis=(0,1)))
        cv_test_ls.append(np.mean(log_score_armos(armos, test_var, test_S2, test_obs), axis=(0,1)))

        armos_params = armos.get_params()
        with open(result_folder + 'GlobalArmos' + str(p) + '_' + str(k) + '.pkl', 'wb') as f:
            pkl.dump(armos_params,f)


    armos_params = armos.get_params()
    with open(result_folder + 'GlobalArmos' + str(p) + '.pkl', 'wb') as f:
        pkl.dump(armos_params,f)
        

    fitting_loss['Global'] = cv_loss
    test_loss['Global'] = cv_test_loss
    log_score_fitting['Global'] = cv_ls
    log_score_test['Global'] = cv_test_ls


    armos_losses = {
        'Fitting Loss': fitting_loss,
        'Validation Loss': test_loss,
        'Log Score Fitting': log_score_fitting,
        'Log Score': log_score_test
    }

    with open(result_folder + 'FinalArmos'+ str(p) + '_losses.pkl', 'wb') as f:
        pkl.dump(armos_losses,f)

    

















