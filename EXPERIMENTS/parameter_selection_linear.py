import os
import tensorflow as tf
import pickle as pkl
import numpy as np
from DATA.data_loading import gen_dataset, load_to_memory
from EMOS.emos_model import EMOS
from EMOS.copula import find_similar_past_obs, rank_shuffling
from RESULTS.scores import energy_score, variogram_score
from tqdm import tqdm


# Workaround for Tensorflow and Keras compatibility issue
os.environ["TF_USE_LEGACY_KERAS"] = "1"



# Data and result folders
nobackup    = 'Data/'
train_data_folder = nobackup + 'train_data/norm_train_data/'
test_data_folder = nobackup + 'test_data/norm_test_data/'

result_folder = 'Results/Validation/'


# Load station information
with open('Data/st-info.pkl', 'rb') as f:
    info = pkl.load(f)


# Define variables and stations to use
var_list = ['PRESS', 'KINETIC', 'HUMID', 'GEOPOT']
omit = ['229', '285', '323']
sts  = [st for st in info.keys() if st not in omit]


# Define parameters for training
T = 49
grid_size = 20 # Maximum grid size to consider for selection of the optimal grid size. The optimal grid size will be selected from the range [0, grid_size].
timesteps = 5000
log_timesteps = 25
log_step = 200

lr = 0.1
cv = None

n_samples = 100 # Number of samples used for evaluation of the energy and variogram scores


# Load training and validation data
val_files  = [train_data_folder + file for file in os.listdir(train_data_folder) if (('20151' in file or '20160' in file) and ('pkl' in file))]
val_dataset   = gen_dataset(val_files, sts=sts, var=var_list, grid_size=grid_size)
val_tag, val_grid, val_var, val_obs = load_to_memory(val_dataset, grid_size)

train_files  = [train_data_folder + file for file in os.listdir(train_data_folder) if ((train_data_folder + file not in val_files) and ('.pkl' in file))]
train_dataset = gen_dataset(train_files, sts=sts, var=var_list, grid_size=grid_size)
tag, grid, var, obs = load_to_memory(train_dataset, grid_size)


print('Start preparation')


# Uncomment the following line to find similar past observations for rank shuffling.
# This is a time-consuming step and can be skipped if the reference observations are already saved.
#ref_obs, ref_tags = find_similar_past_obs(val_tag, val_var[:,:,0], tag, var[:,:,0], obs, n_samples=n_samples)


# Load reference observations for rank shuffling.
# The reference observations may have been stored in a different order, so they need to be matched with the current order of validation tags.
with open(result_folder + 'Emos/Emos_reference.pkl', 'rb') as f:
    emos_ref_obs = pkl.load(f)

ref_obs = []
ref_tags = []

for i, t1 in enumerate(val_tag.numpy().astype('str')):
    ind = [j for j, t2 in enumerate(emos_ref_obs['Tags'].numpy().astype('str')) if t1 == t2]
    ref_obs.append(emos_ref_obs['Ref Obs'][:,ind[0],:])
    ref_tags.append(emos_ref_obs['Ref Tags'][:,ind[0]])

ref_obs = np.transpose(ref_obs, [1,0,2])
ref_tags = np.transpose(ref_tags)


print('Preparation finished')


##### Fitting globally #####



# Select grid size
training_losses = {}
final_losses = {}
for S2_size in tqdm(range(grid_size+1)):
    # Compute the spatial variance for the current grid size
    val_S2  = val_grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
    val_S2  = tf.math.reduce_std(val_S2, axis=(-2,-1))
    S2  = grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
    S2  = tf.math.reduce_std(S2, axis=(-2,-1))

    # Fit the EMOS model
    emos = EMOS(1,T)
    emos.opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=cv)
    hist = emos.fit(var[...,:1], S2, obs, steps=timesteps)

    # Store the training losses and final losses
    training_losses[S2_size] = hist[::log_step]
    final_losses[S2_size] = tf.reduce_mean(emos.loss(val_var[...,:1], val_S2, val_obs)).numpy()


# Select the minimum final loss to determine the optimal grid size
opt_gridsize = min(final_losses,key=final_losses.get)

selection_results = {
    'Train losses': training_losses,
    'Results': final_losses,
    'Optimal gridsize': opt_gridsize
}

with open(result_folder + 'Emos/gridsize_selection.pkl', 'wb') as f:
    pkl.dump(selection_results,f)


''' Uncomment the following lines to load the results of the grid size selection.
with open(result_folder + 'Emos/gridsize_selection.pkl', 'rb') as f:
    selection_results = pkl.load(f)


opt_gridsize = selection_results['Optimal gridsize']
'''

print('Optimal gridsize: ', opt_gridsize)


# Select variables

training_losses_vars = {}
selection_results = {}
forward_selection = {}
S2_size = opt_gridsize

# Compute the spatial variance for the optimal grid size
val_S2  = val_grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
val_S2  = tf.math.reduce_std(val_S2, axis=(-2,-1))
S2  = grid[...,grid_size-S2_size:grid_size+S2_size+1,grid_size-S2_size:grid_size+S2_size+1]
S2  = tf.math.reduce_std(S2, axis=(-2,-1))



print("Base ['WIND10']")

# Fit the EMOS model with only the first variable (WIND10)
emos = EMOS(1,T)
emos.opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=cv)
hist  = emos.fit(var[...,:1], S2, obs, steps=timesteps)

print('Fitted')

# Store the training losses for the first variable
training_losses_vars[str([0])] = hist[::log_step]

# Compute the validation loss, energy score, and variogram score for the first variable
ls = tf.reduce_mean(emos.loss(val_var[...,:1], val_S2, val_obs)).numpy()
samples = emos.sample(val_var[...,:1], val_S2, n_samples=n_samples)
samples_ss = rank_shuffling(samples, ref_obs[:n_samples])
es = np.mean(energy_score(samples_ss, val_obs, batchsize=500))
vs = np.mean(variogram_score(samples_ss, val_obs, batchsize=500))
score = {
    'LS': ls,
    'ES': es,
    'VS': vs
}

print('Score :', score)

selection_results[str([0])] = score

forward = [(0,'WIND10',score)]
var_index = [(i+1, var_name) for i, var_name in enumerate(var_list)]

# Perform forward selection of variables based on the energy score, by adding one variable at a time
# and selecting the one that minimizes the energy score on the validation set.
while var_index:
    tests = []
    for i, var_name in var_index:
        print('Testing', [k for _,k,_ in forward] + [var_name])
        idx = [j for j,_,_ in forward] + [i]

        emos = EMOS(len(forward)+1,T)
        emos.opt = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=cv)

        var_sel = tf.gather(var, idx, axis=-1)
        val_var_sel = tf.gather(val_var, idx, axis=-1)

        hist  = emos.fit(var_sel, S2, obs, steps=timesteps)
        training_losses_vars[str(idx)] = hist[::log_step]

        ls = tf.reduce_mean(emos.loss(val_var_sel, val_S2, val_obs)).numpy()
        samples = emos.sample(val_var_sel, val_S2, n_samples=n_samples)
        samples_ss = rank_shuffling(samples, ref_obs[:n_samples])
        es = np.mean(energy_score(samples_ss, val_obs, batchsize=500))
        vs = np.mean(variogram_score(samples_ss, val_obs, batchsize=500))

        score = {
            'LS': ls,
            'ES': es,
            'VS': vs
        }

        tests.append((i, var_name, score))
        print('Score :', score)

        selection_results[str(idx)] = score

    best_var = min(tests, key=lambda x:x[-1]['ES'])
    forward.append(best_var)
    var_index.remove(tuple(best_var[:-1]))
    

selection_results = {
    'Train losses': training_losses_vars,
    'Results': selection_results,
    'Optimal variables': forward
}

with open(result_folder + 'Emos/variable_selection.pkl', 'wb') as f:
    pkl.dump(selection_results,f)


print('Optimal gridsize: ', opt_gridsize)
print('Variables: ', forward)


