import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd
import requests
from siphon.catalog import TDSCatalog
from bs4 import BeautifulSoup
import wget
import cfgrib
from metpy.io import parse_metar_file
from metpy.plots import USCOUNTIES
from metpy.interpolate import interpolate_to_grid, remove_nan_observations
from datetime import datetime as dt, timedelta
from time import gmtime, strftime

# Request HRRR data
url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl'
soup = BeautifulSoup(requests.get(url).text, 'html.parser')
urls_gfs = [link.get('href') for link in soup.find_all('a')]
date = urls_gfs[0][-8:]

# Download HRRR Data
utc_hour = (int(strftime('%H', gmtime())) - 1) % 24
time2 = f'{utc_hour:02}'
x2 = '01'

ds = wget.download(f'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl'
                    f'?file=hrrr.t{time2}z.wrfsfcf{x2}.grib2&lev_2_m_above_ground=on'
                    f'&var_TMP=on&subregion=&dir=%2Fhrrr.{date}%2Fconus')

# Process HRRR Data
ds_temp = xr.open_dataset(ds, engine='cfgrib', filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround'})
lon, lat = ds_temp.longitude.values, ds_temp.latitude.values
hrrr_t_arr = (ds_temp['t2m'].squeeze() - 273.15) * 9/5 + 32

# Get METAR data
catalog_url = 'https://thredds-test.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml'
catalog = TDSCatalog(catalog_url)
dataset = catalog.datasets.filter_time_nearest(ds_temp['valid_time'].values)
dataset.download()
df = parse_metar_file(dataset.name)

df['date_time'] = pd.to_datetime(df['date_time'])
df = df.loc[(df['longitude'] >= -90) & (df['latitude'] >= 35.8) & (df['longitude'] <= 63) & (df['latitude'] <= 48)]

# Filter METAR data to be close to the model valid time
model_valid_time = pd.to_datetime(ds_temp['valid_time'].values)
if hasattr(model_valid_time, '__len__') and len(model_valid_time) > 0:
    model_valid_time = model_valid_time[0]
    
# Keep your original time window approach
df = df.loc[(df['date_time'].dt.minute >= 50) | (df['date_time'].dt.minute <= 10)]

lats_i, lons_i, ts_i = df['latitude'], df['longitude'], (df['air_temperature'] * 9/5) + 32

# Compute HRRR vs Observation Differences
def compute_diff(i):
    try:
        stn_lat, stn_lon = lats_i.iloc[i], lons_i.iloc[i]
        
        # Convert HRRR longitudes to match METAR convention if needed
        hrrr_lon = lon.copy()
        if np.mean(hrrr_lon) > 180:  # Check if HRRR is in 0-360 format
            hrrr_lon = np.where(hrrr_lon > 180, hrrr_lon - 360, hrrr_lon)
        
        # Calculate distances properly
        y_dist = np.abs(lat - stn_lat)
        x_dist = np.abs(hrrr_lon - stn_lon)
        total_dist = np.sqrt(x_dist**2 + y_dist**2)
        
        x1, y1 = np.unravel_index(np.argmin(total_dist), total_dist.shape)
        return hrrr_t_arr[x1, y1] - ts_i.iloc[i]
    except Exception as e:
        print(f"Error processing station {i}: {e}")
        return np.nan

diff_arr = np.array([compute_diff(i) for i in range(len(lats_i))])

# Plot
fig = plt.figure(figsize=(18, 12), dpi=125)
proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42, standard_parallels=(30, 60))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent((275, 294, 36.2, 47.2))
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='#edfbff'))
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'lakes', '50m', edgecolor='face', facecolor='#edfbff'))
ax.add_feature(USCOUNTIES.with_scale('500k'), linewidth=0.5, edgecolor='gray')

# Plotting difference grid
clevs = np.arange(-5, 5.1, 0.2)
cmap = plt.get_cmap('bwr')
norm = col.BoundaryNorm(clevs[clevs != 0], cmap.N)
sc = ax.scatter(lons_i, lats_i, c=diff_arr, cmap=cmap, norm=norm, edgecolors='k', transform=ccrs.PlateCarree())

plt.colorbar(sc, ax=ax, orientation='vertical', label='Temperature Difference (°F)')

# Add a more descriptive title with the valid time
valid_time_str = pd.to_datetime(ds_temp['valid_time'].values[0]).strftime('%Y-%m-%d %H:%M UTC')
plt.title(f"2m Temperature Differences: HRRR vs Observed\nValid: {valid_time_str}")

plt.show()
