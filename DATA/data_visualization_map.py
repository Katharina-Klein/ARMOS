import pygrib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import cartopy.crs as ccrs
import cartopy.feature as cfeature



# Source folder with GRIB files containing NWP forecast data
grb_folder  = 'Data/GRIB/'

# Target folder in nobackup hard drive
nobackup    = 'Data/'
data_folder = nobackup + 'all_data/'


# Longitude/Latitude limits and grid dimensions of the forecast data
lonmin = 0.0
lonmax = 11.063
latmin = 49.0
latmax = 55.877

grid_size = 300


# Generate the grid for longitudes and latitudes
lons = np.linspace(lonmin, lonmax, grid_size)
lats = np.linspace(latmin, latmax, grid_size)
grid_lon, grid_lat = np.meshgrid(lons, lats)


# Station information (file created by Daniel)
with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)

# Get grid location (latitude/longitude) for each station
latlon = {st:(info['LAT'], info['LON']) for st, info in st_info.items()}


# Get forecast data for 01-01-2015 at 48h lead time
grb_file = pygrib.open(grb_folder + 'HA40_N25_WINDFC_201501010000_04800_GB')


wind10_u  = grb_file.select(indicatorOfParameter=33,  level=10 )[0].values
wind10_v  = grb_file.select(indicatorOfParameter=34,  level=10 )[0].values
wind10_norm    = np.linalg.norm([wind10_u,wind10_v], axis=0)


with open(data_folder + '20150101.pkl', 'rb') as file:
    wind10_norm = pkl.load(file)['GRID'][48]
    




# Plot with Cartopy
fig = plt.figure(figsize=(10, 10))

# Create the map
m = plt.axes(projection=ccrs.Mercator())

m.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
m.add_feature(cfeature.BORDERS)
m.add_feature(cfeature.COASTLINE)
gl = m.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

# Add stations
for st, (lat, lon) in latlon.items():
    x, y = lon, lat
    m.plot(x, y, markersize=3, marker='D', color='r', transform=ccrs.PlateCarree())

# Add forecast data
clb = m.pcolormesh(grid_lon, grid_lat, wind10_norm, cmap = plt.cm.nipy_spectral, transform=ccrs.PlateCarree())
plt.colorbar(clb)

plt.title('Forecast for valid time January 1st, 2015, at 48h lead time')

plt.savefig('Pictures/test_forecast_map.png')
plt.show()




