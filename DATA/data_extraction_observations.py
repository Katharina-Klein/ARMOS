import os
import pickle as pkl
import pandas as pd
import datetime as dt
from tqdm import tqdm
import json



# Source folder with CSV files containing observations
csv_folder = 'Data/CSV/'

# Target folder in nobackup hard drive
nobackup    = 'Data/'
data_folder = nobackup + 'train_data/'                                             # training data 2015 - 2017
#data_folder = nobackup + 'test_data/'                                              # test data 2018 - 2019

# Data dimensions
T = 49
grid_size = 300


# Station information (file created by Daniel)
with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)



##########   Observations


# Get CSV file names from folder
csv_files = sorted(os.listdir(csv_folder))

# Get all dates for which a Pickle file (with forecast data) already exists
pkl_dates = {file : dt.datetime.strptime(file[:-4],'%Y%m%d') for file in os.listdir(data_folder) if '.pkl' in file}



errors = []


# Loop over CSV files and store relevant data in the corresponding Pickle files
for csv_file in tqdm(csv_files):
    
    # Each CSV file contains observations for a given location (station) and a given range of dates
    # Get station code and range of dates for current CSV file     
    splits = csv_file.split('_')
    code = splits[2]
    start = dt.datetime.strptime(splits[-1][:8], '%Y%m%d') - dt.timedelta(days=1)
    end = dt.datetime.strptime(splits[-1][9:17], '%Y%m%d') + dt.timedelta(days=1)
    
    # Get corresponding Pickle files that fit in this date range     
    # Skip CSV file if no Pickle files are found in this date range
    pkl_files = [
        file for file, date in pkl_dates.items() if (start <= date) and (date <= end)
    ]
    if not pkl_files:
        continue
    
    # Group data from CSV file in hourly observations
    df = pd.read_csv(csv_folder + csv_file)
    df = df.rename(columns={'IT_DATETIME':'DATETIME', 'FF_10M_10':'OBS'}, inplace=False)
    df = df[['DATETIME', 'OBS']]
    df['DATETIME'] = pd.to_datetime(df['DATETIME'].str.slice(stop=11), format='%Y%m%d_%H')
    df = df.groupby('DATETIME').first()
    
    # Save observations to corresponding Pickle files
    for pkl_file in pkl_files:
        tag = pkl_file[:8] + '_' + code
        lead00 = pkl_dates[pkl_file]
        lead48 = lead00 + dt.timedelta(days=2)
        obs = df.loc[(lead00 <= df.index) & (df.index <= lead48)]
        index = list(map(lambda d: int(d.total_seconds() // 3600), obs.index - lead00))
        obs = obs.values[:,0]
        with open(data_folder + pkl_file, 'rb') as f:
            sample = pkl.load(f)
        try:
            sample[tag]['OBS'][index] = obs
        except:
            print("No key found: " + tag)
            errors.append(tag)
        with open(data_folder + pkl_file, 'wb') as f:
            pkl.dump(sample, f)



print(errors)
# Save error list
with open(data_folder + 'data_errors_obs.json', 'w') as f:
    json.dump(errors, f)







