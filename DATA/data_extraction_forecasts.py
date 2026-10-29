##### This code was almost entirely written by Daniel

import os
import pygrib
import numpy as np
import pickle as pkl
import json
from tqdm import tqdm


# Source folder with GRIB files containing NWP forecast data
grb_folder  = 'Data/GRIB/'                # training data 2015 - 2017
#grb_folder  = 'Data/GRIB/test/'           # test data 2018 - 2019


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



##########   NWP forecast data

# Get GRIB files and corresponding dates
grb_files = sorted([file for file in os.listdir(grb_folder) if (('test' not in file) and ('0000_0' in file))])
grb_dates = sorted(list(set([file.split('_')[3][:8] for file in grb_files])))



errors = []

for date in tqdm(grb_dates, position=0, leave=True):
#for date in grb_dates:

    # Create data format and initialize values at -1
    data = {date + '_' + code : {
        'LOC': info['GRID'],
        'OBS': -np.ones(T),
        'WIND850':-np.ones(T),
        'WIND925':-np.ones(T),
        'PRESS':-np.ones(T),
        'KINETIC':-np.ones(T),
        'HUMID':-np.ones(T),
        'GEOPOT':-np.ones(T)}
    for code, info in st_info.items()}
    data['GRID'] = - np.ones((T, grid_size, grid_size))

    # Loop over GRIB files (each such file contains data for all stations for a single lead time)
    grbs = sorted([file for file in grb_files if date in file])
    try:
        for i, grb_file in enumerate(grbs):
            # Load GRIB data for the whole grid and calculate wind speeds from u and v components
            with pygrib.open(grb_folder + grb_file) as grb_open:
                wind10_u  = grb_open.select(indicatorOfParameter=33,  level=10 )[0].values
                wind10_v  = grb_open.select(indicatorOfParameter=34,  level=10 )[0].values
                wind10    = np.linalg.norm([wind10_u,wind10_v], axis=0)
                wind850_u = grb_open.select(indicatorOfParameter=33,  level=850)[0].values
                wind850_v = grb_open.select(indicatorOfParameter=34,  level=850)[0].values
                wind850   = np.linalg.norm([wind850_u,wind850_v], axis=0)
                wind925_u = grb_open.select(indicatorOfParameter=33,  level=925)[0].values
                wind925_v = grb_open.select(indicatorOfParameter=34,  level=925)[0].values
                wind925   = np.linalg.norm([wind925_u,wind925_v], axis=0)
                press     = grb_open.select(indicatorOfParameter=1,   level=0  )[0].values
                kinetic   = grb_open.select(indicatorOfParameter=200, level=47 )[0].values
                humid     = grb_open.select(indicatorOfParameter=52,  level=2  )[0].values
                geopot    = grb_open.select(indicatorOfParameter=6,   level=700)[0].values

                # Store wind speeds at 10m for the full grid
                data['GRID'][i] = wind10
                # Store other variables for local stations
                for tag, sample in data.items():
                    if not tag == 'GRID':
                        loc = sample['LOC']
                        sample['WIND850'][i] = wind850[loc]
                        sample['WIND925'][i] = wind925[loc]
                        sample['PRESS'  ][i] = press[loc]
                        sample['KINETIC'][i] = kinetic[loc]
                        sample['HUMID'  ][i] = humid[loc]
                        sample['GEOPOT' ][i] = geopot[loc]

        # Store data in a pickle file   
        if data:
            with open(data_folder + date + '.pkl', 'wb') as f:
                pkl.dump(data, f)
        else:
            print(f'{date} has no files!')
    except:
        print(date + ' has a problem!')
        errors.append(date)


print(errors)
# Save error list
with open(data_folder + 'data_errors.json', 'w') as f:
    json.dump(errors, f)



