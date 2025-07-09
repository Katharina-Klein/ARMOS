import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


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
#files = files_all[545:]



omit = ['229', '285', '323']
stations_all = [st for st in st_info.keys() if st not in omit]
locs = [st_info[st]['GRID'] for st in stations_all]

stations_individual = stations_all


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
    



wind10_error = wind10_fct - wind10_obs

ind = np.where(wind10_obs > -1, True, False)

mean = np.mean(wind10_error, (0,1), where=ind)
std = np.std(wind10_error, (0,1), where=ind)


wind10_error_all = np.reshape(wind10_error, (len_files*len_stats,T))[np.all(np.reshape(wind10_obs, (len_files*len_stats,T))>-1, axis=1)==True,:]
confint = [mean-std, mean+std]


fig = plt.figure(figsize=(10, 10))
plt.plot(np.arange(T),mean,label='Mean')
plt.fill_between(np.arange(T), confint[0], confint[1], alpha=0.1)
#plt.plot(np.arange(T),std,label='Standard deviation')
#plt.title('Forecast error (fct - obs) - Global')
plt.xlabel('Lead time (h)')
plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
plt.xlim([0, 48])
plt.ylabel('Wind speed forecast error (m/s)')
#plt.legend()
plt.savefig('Pictures/Bias/windspeed_error_global.png')
plt.show()


'''
for station in stations_individual:

    st_index = stations_all.index(station)
    st_ind = np.where(wind10_obs[:,st_index,:] > -1, True, False)
    st_mean = np.mean(wind10_error[:,st_index,:], 0, where=st_ind)
    st_std = np.std(wind10_error[:,st_index,:], 0, where=st_ind)

    plt.figure()
    plt.plot(np.arange(T),st_mean,label='Mean')
    plt.plot(np.arange(T),st_std,label='Standard deviation')
    plt.title('Forecast error (fct - obs) - Station ' + station)
    plt.xlabel('Lead time (h)')
    plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    plt.xlim([0, 48])
    plt.ylabel('Wind speed (m/s)')
    plt.legend()
    plt.savefig('Pictures/Bias/windspeed_error_' + station + '.png')
    plt.show()
'''














