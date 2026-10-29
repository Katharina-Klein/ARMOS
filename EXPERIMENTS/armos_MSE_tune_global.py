import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from DATA.data_loading import gen_dataset
from DATA.data_loading import load_to_memory
from ARMOS.armos_model import ARMOS_MSE as ARMOS
from RESULTS.scores import log_score_armos
from tqdm import tqdm


# Workaround for Tensorflow and Keras compatibility issue
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# Data and result folders
nobackup    = 'Data/'
train_data_folder = nobackup + 'train_data/norm_train_data/'
test_data_folder = nobackup + 'test_data/norm_test_data/'

result_folder = 'Results/Validation/ArmosMSE/'


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



# Train and evaluate the model for different values of p
for p in [1, 2, 3, 4]:

    fitting_loss = []
    validation_loss = []
    log_score_validation = []
    log_score_fitting = []
    labels = []

    for i, lr in enumerate([0.001, 0.005, 0.01, 0.05, 0.1]):

        for j, cv in enumerate([None, 0.5, 1.0, 1.5]):

            # Initialize the ARMOS model and define the optimizer.
            armos   = ARMOS(num_var, p, T)
            armos.opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=cv)

            cv_loss = []
            cv_val_loss = []
            cv_ls = []
            cv_val_ls = []

            # Train the ARMOS model and store the training and validation losses for each logging step.
            for k in tqdm(range(log_timesteps)):
                loss = armos.fit(var, S2, obs, steps=log_step)
                cv_loss.append(loss[len(loss)-1])
                cv_val_loss.append(np.mean(armos.loss(val_var, val_S2, val_obs)))
                cv_ls.append(np.mean(log_score_armos(armos, var, S2, obs), axis=(0,1)))
                cv_val_ls.append(np.mean(log_score_armos(armos, val_var, val_S2, val_obs), axis=(0,1)))

                armos_params = armos.get_params()
                with open(result_folder + 'GlobalArmos' + str(p) + '_' + str(i) + str(j) + str(k) + '.pkl', 'wb') as f:
                    pkl.dump(armos_params,f)
                

            fitting_loss.append(cv_loss)
            validation_loss.append(cv_val_loss)
            log_score_fitting.append(cv_ls)
            log_score_validation.append(cv_val_ls)
            labels.append('LR = '+str(lr)+', CV = '+str(cv))


            armos_losses = {
                'Fitting Loss': fitting_loss,
                'Validation Loss': validation_loss,
                'Log Score Fitting': log_score_fitting,
                'Log Score': log_score_validation,
                'Labels': labels
            }

            with open(result_folder + 'GlobalArmos'+ str(p) + '_losses.pkl', 'wb') as f:
                pkl.dump(armos_losses,f)













