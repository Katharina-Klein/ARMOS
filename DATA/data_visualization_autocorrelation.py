import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from statsmodels.graphics.tsaplots import acf, plot_acf


# Target folder in nobackup hard drive
nobackup    = 'Data/'
data_folder = nobackup + 'all_data/'

path_length = len(data_folder)


# Data dimensions
T = 49
grid_size = 300


# Station information (file created by Daniel)
with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)


files_all   = sorted([data_folder + file for file in os.listdir(data_folder)])
files = files_all[:545]

stations_individual = ['248', '258', '278', '290', '348']
omit = ['229', '285', '323']
stations_all = [st for st in st_info.keys() if st not in omit]
locs = [st_info[st]['GRID'] for st in stations_all]



########## Mean and standard deviation of the bias (error)


len_files = len(files)
len_stats = len(stations_all)


wind10_obs = np.zeros([len_files,len_stats,T])
wind10_fct = np.zeros([len_files,len_stats,T])



for i, file in enumerate(files):
    with open(file, 'rb') as f:
        data = pkl.load(f)
    
    date = file[path_length:path_length+8]

    grid = data.pop('GRID')

    for j, station in enumerate(stations_all):
        
        locx, locy = locs[j]
        wind10_fct[i][j] = grid[:,locx,locy]

        try:
            st_obs = data.pop(date + '_' + station)
            wind10_obs[i][j] = st_obs['OBS']
        except:
            st_obs = -np.ones(T)
            wind10_obs[i][j] = st_obs
    

ind = np.array([i for i, obs_array in enumerate(wind10_obs.reshape((len_files*len_stats,T))) if obs_array[0] > -1], dtype='int16')


wind10_error = wind10_fct - wind10_obs

wind10_error_red = wind10_error.reshape((len_files*len_stats,T))[ind]

mean = np.mean(wind10_error_red, 0)
std = np.std(wind10_error_red, 0)


norm_res  = (wind10_error_red - mean) / std


# Get correlation matrix
mat = norm_res.T @ norm_res / norm_res.shape[0]



std_diag = np.zeros(T)
for i in range(T):
    diag = np.diagonal(mat,i)
    std_diag[i] = np.std(diag)

max_std = std_diag.max()
print(max_std)



# Plot matrix
fig, ax = plt.subplots(1, figsize=(5,5))

cax = ax.matshow(mat, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_title('Leadtime correlations')
ax.set_xlabel('j')
ax.set_ylabel('i')
ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
ax.set_yticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
ax.xaxis.set_label_position('top')
fig.colorbar(cax)

plt.savefig('Pictures/Autocorrelation/correlation_coefficients_global.png')
plt.show()




########## Autocorrelation


mean = np.mean(wind10_error_red, 1)
std = np.std(wind10_error_red, 1)

norm_res  = ((wind10_error_red.T - mean.T) / std.T).T




acf_corr = np.zeros(49)
acf_confint = np.zeros((49,2))
temp = []
temp_confint = []


for i in range(1,len(norm_res)):
    temp, temp_confint = acf(wind10_error_red[i],nlags=49,alpha=0.05)
    acf_corr = acf_corr + temp
    acf_confint = acf_confint + temp_confint

acf_corr = acf_corr / len(wind10_error_red)
acf_confint = acf_confint / len(wind10_error_red)
acf_confint[:,0] = acf_confint[:,0] - acf_corr
acf_confint[:,1] = acf_confint[:,1] - acf_corr

plt.figure()
plt.plot(acf_corr, 'bo')
plt.fill_between(np.arange(T), acf_confint[:,0], acf_confint[:,1], alpha=0.2)
plt.hlines(0, 0, T, 'k', linestyle='dashed', linewidth=1)
plt.title('Autocorrelation')
plt.xlabel('Lag (h)')
plt.ylabel('Average autocorrelation')
plt.xlim([0,T])
plt.xticks(range(T)[::6])
plt.savefig('Pictures/Autocorrelation/acf_plot_all.png')
plt.show()


acf_corr, acf_confint = acf(wind10_error_red[10],nlags=49,alpha=0.05)
acf_confint[:,0] = acf_confint[:,0] - acf_corr
acf_confint[:,1] = acf_confint[:,1] - acf_corr

print(files[0])
print(stations_all[10])

plt.figure()
plt.plot(acf_corr, 'bo')
plt.fill_between(np.arange(T), acf_confint[:,0], acf_confint[:,1], alpha=0.2)
plt.hlines(0, 0, T, 'k', linestyle='dashed', linewidth=1)
#plt.title('Autocorrelation - Single')
plt.xlabel('Lag (h)')
plt.ylabel('Autocorrelation')
plt.xlim([0,T])
plt.xticks(range(T)[::6])
plt.savefig('Pictures/Autocorrelation/acf_plot_single.pdf')
plt.show()








