import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import cartopy.crs as ccrs
import cartopy.feature as cfeature



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

delta_lon = grid_lon[0][1] - grid_lon[0][0]
delta_lat = grid_lat[1][0] - grid_lat[0][0]

delta_lon = (lonmax - lonmin)/grid_size
delta_lat = (latmax - latmin)/grid_size


# Station information (file created by Daniel)
with open('Data/st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)

# Get grid location (latitude/longitude) for each station
latlon = {st:(info['LAT'], info['LON']) for st, info in st_info.items()}
grid_position = {st:info['GRID'] for st, info in st_info.items()}



for st, (lat, lon) in latlon.items():
    x, y = lat, lon
    x2, y2 = grid_position[st]
    dist_lon = np.abs(grid_lon[0][y2] - y)
    dist_lat = np.abs(grid_lat[x2][0] - x)
    
    if dist_lon > delta_lon/2 or dist_lat > delta_lat/2:
        print(st + ' has a problem!')
        print('x = ' + str(x) + ', y = ' + str(y))
        print('x = ' + str(grid_lat[x2][0]) + ', y = ' + str(grid_lon[0][y2]))






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

    x2, y2 = grid_position[st]
    m.plot(grid_lon[0][y2], grid_lat[x2][0], markersize=3, marker='D', color='b', transform=ccrs.PlateCarree())
    


plt.title('Weather stations')

plt.savefig('Pictures/test_station_map.png')
plt.show()





