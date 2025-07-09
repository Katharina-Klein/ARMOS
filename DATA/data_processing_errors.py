##### This code was entirely written by Daniel

import os
import json
import numpy as np
import pickle as pkl
from tqdm import tqdm

# Get Pickle files, station info and initialize error list
nobackup    = 'Data/'
data_folder = nobackup + 'train_data/'                                             # training data 2015 - 2017
#data_folder = nobackup + 'test_data/'                                              # test data 2018 - 2019

pkl_files   = [data_folder + file for file in sorted(os.listdir(data_folder)) if '.pkl' in file]


with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)

errors = []


# Check data consistency for all files, all values should be non negative
for pkl_file in tqdm(pkl_files):
    
    # Load .pkl file
    with open(pkl_file, 'rb') as f:
        data = pkl.load(f)
        
    # Get wind speed 10m grid data
    # This way it won't get stuck in the loop below
    grid = data.pop('GRID')
    
    # Check local data for all stations
    for stationTag, locData in data.items():
        date, code = stationTag.split('_')
        # Check for correct location
        if not st_info[code]['GRID'] == locData.pop('LOC'):
            print(f'ERROR: {stationTag} has wrong location.')
            errors.append((stationTag, 'LOC'))
        # Check for valid data for all variables
        for var, values in locData.items():
            if not np.all(values >= 0):
                # ROUGH is a problem variable that was later removed
                if not var == 'ROUGH':
                    print(f'ERROR: {stationTag}.{var} has invalid data.')
                    errors.append((stationTag, var))



# Save error list
with open(data_folder + 'errors_to_remove.json', 'w') as f:
    json.dump(errors, f)



# Remove dictionary entries with inconsistent data
# Read error list from errors.txt
with open(data_folder + 'errors_to_remove.json', 'r') as f:
    errors = json.load(f)
    
# Loop over dates and remove samples with errors from Pickle files
error_dates = list({error[0][:8] for error in errors})

for date in tqdm(error_dates):
    # Get pickle file
    file = data_folder + date + '.pkl'
    with open(file, 'rb') as f:
        data = pkl.load(f)
    # Remove dictionary keys with errors
    keys  = [key for key, _ in errors if date in key]
    for key in keys:
        try:
            data.pop(key)
        except:
            print("No key for " + key)
    # Save error free file
    with open(file, 'wb') as f:
        pkl.dump(data,f)




