# Import Libraries
import numpy as np  # Numerical operations
from siphon.catalog import TDSCatalog  # To access THREDDS datasets
import matplotlib  # Plotting library
import cartopy.crs as ccrs  # Cartopy for map projections
import cartopy.feature as cfeature  # Cartopy feature (e.g., oceans, countries)
import matplotlib.pyplot as plt  # Plotting with matplotlib
from metpy.io import parse_metar_file  # Parsing METAR data
import cartopy.feature as cfeature  # Re-imported Cartopy feature
import xarray as xr  # Working with multi-dimensional datasets
import matplotlib.colors as col  # Colormap for plotting
from metpy.plots import USCOUNTIES  # US county boundaries for maps
import scipy.ndimage as ndimage  # For image processing (used in smoothing)
from datetime import datetime, timedelta  # Date and time utilities
from metpy.interpolate import interpolate_to_grid, remove_nan_observations  # Interpolation functions from MetPy
import xarray  # Importing Xarray again (duplicate, unnecessary)
from bs4 import BeautifulSoup  # Web scraping library to parse HTML
import wget  # For downloading files from URLs
import cfgrib  # GRIB file handling
import requests  # For HTTP requests
import pandas as pd  # Data manipulation with pandas
from datetime import datetime as dt, timedelta  # Re-imported datetime (duplicate, unnecessary)
from time import gmtime, strftime  # Time formatting
import os  # Operating system functions (file handling)
import glob, os  # File handling (glob used to list files)
from matplotlib.colors import ListedColormap  # Custom colormap
import matplotlib.colors as colors  # Colormap utilities

# Request HRRR data (High-Resolution Rapid Refresh)
url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl'
reqs = requests.get(url)  # Send request to HRRR data URL
soup = BeautifulSoup(reqs.text, 'html.parser')  # Parse the HTML response

# Extract file URLs from the HRRR catalog
urls_gfs = []
for link in soup.find_all('a'):
    urls_gfs.append(link.get('href'))

# Extract date from the first URL (formatting)
date = urls_gfs[0][-8::]

# Download HRRR Data for a specific forecast hour
x = 1
utc_hour = strftime('%H', gmtime())  # Get current UTC hour
utc_hour = int(utc_hour) - 1  # Subtract 1 for the previous hour
if utc_hour == -1:
    utc_hour = 23  # Wrap around if hour is 0
utc_hour = str(utc_hour)
time = utc_hour
if len(str(time)) == 1:
    time2 = f'0{time}'  # Format to 2 digits
else:
    time2 = f'{time}'

# Format the forecast hour
if len(str(x)) == 1:
    x2 = f'0{x}'
else:
    x2 = f'{x}'

# Download the specific HRRR dataset based on the formatted time
ds = wget.download(f'https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?file=hrrr.t{time2}z.wrfsfcf{x2}.grib2&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_250_mb=on&lev_500_mb=on&lev_700_mb=on&lev_850_mb=on&lev_1000_mb=on&lev_mean_sea_level=on&lev_surface=on&var_GUST=on&var_APCP=on&var_CFRZR=on&var_CICEP=on&var_CRAIN=on&var_CSNOW=on&var_REFD=on&var_DPT=on&var_HGT=on&var_MSLMA=on&var_PRATE=on&var_TMP=on&var_UGRD=on&var_VGRD=on&subregion=&leftlon=229&rightlon=297.5&toplat=55.5&bottomlat=20&dir=%2Fhrrr.{date}%2Fconus')

################################## TEMPERATURES #####################################

# Open the HRRR temperature data from the downloaded file
ds_temp =  xr.open_mfdataset(ds, engine='cfgrib', filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround'})
lon = ds_temp.longitude.values  # Longitude values of the HRRR grid
lat = ds_temp.latitude.values  # Latitude values of the HRRR grid

###################################  METAR  #########################################

# Retrieve METAR data time from the HRRR dataset
hrrr_init = ds_temp.time.dt

# Access the METAR data catalog from NOAA thredds server
catalog_url = 'https://thredds-test.unidata.ucar.edu/thredds/catalog/noaaport/text/metar/catalog.xml'
catalog = TDSCatalog(catalog_url)

# Extract the initialization and valid times from the HRRR dataset
valid_time = ds_temp['valid_time']
init_time = ds_temp['time']
init_time = init_time.values
init_time = dt.utcfromtimestamp(init_time.tolist()/1e9)  # Convert to datetime
init_time_str = init_time.strftime('%Y-%m-%d %H:%MZ')

valid_time = valid_time.values
valid_time = dt.utcfromtimestamp(valid_time.tolist()/1e9)
valid_time_str = valid_time.strftime('%Y-%m-%d %H:%MZ')

# Check if METAR and HRRR valid times match
ds1 = catalog.datasets.filter_time_nearest(valid_time)
metar_init = datetime.strptime(ds1.name[6:17], '%Y%m%d_%H')

if metar_init != valid_time:
    raise ValueError("---> METAR time and valid time are not the same!")
else:
    print("---> METAR and valid time are the same")

# Download METAR data from NOAA thredds server
ds1.download()
df = parse_metar_file(ds1.name)  # Parse METAR data into a dataframe

# Limit data to specific region and observation times near the valid time
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.loc[(df['longitude'] >= -90) & (df['latitude'] >= 35.8) & (df['longitude'] <= 63) & (df['latitude'] <= 48)]
df = df.loc[(df['date_time'].dt.minute >= 50) | (df['date_time'].dt.minute <= 10)]

# Prepare data for comparison (convert temperature from Celsius to Fahrenheit)
lats_i = df['latitude']
lons_i = df['longitude']
ts_i = (df['air_temperature']) * (9/5) + 32  # Convert to Fahrenheit
hrrr_t = (ds_temp['t2m'].squeeze() - 273.15) * (9/5) + 32  # Convert to Fahrenheit from Kelvin
hrrr_t_arr = hrrr_t.values

# Compute HRRR vs observation temperature differences
lats = ds_temp['latitude'].values
lons = ds_temp['longitude'].values

print('---> Starting Differencing')

# Define a function to compute the temperature difference between HRRR and METAR observations
@np.vectorize
def compute_diff(i):
    try:
        stn_lat = lats_i[i]
        stn_lon = lons_i[i]
        abslat = np.abs(lat - stn_lat)
        abslon = np.abs((lon - 360) - stn_lon)
        c = np.maximum(abslon, abslat)
        x1, y1 = np.where(c == np.min(c))
        hrrr_t = hrrr_t_arr[x1[0], y1[0]]
        actual_t = ts_i[i]
        diff = hrrr_t - actual_t
    except:
        diff = np.nan
    return diff

# Apply the function to all observation data
diff_arr = compute_diff(np.arange(len(lats_i)))

print('---> Finished Differencing')

#################################    PLOT    #######################################

# Set up the plot figure
print('---> Starting Plot')
fig = plt.figure(figsize=(18, 12), dpi=125)
proj = ccrs.PlateCarree()
mapcrs = ccrs.LambertConformal(central_longitude=-75, central_latitude=42, standard_parallels=(30, 60))
datacrs = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=mapcrs)

# Set map bounds for the plot
sub_w_ne = 275
sub_e_ne = 294
sub_n_ne = 47.2
sub_s_ne = 36.2
ax.set_extent((sub_w_ne, sub_e_ne, sub_s_ne, sub_n_ne))

# Add geography features (e.g., oceans, lakes, country borders)
ax.coastlines(resolution='10m', linewidth=1.5, color='black', zorder=16)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.6, color='black', zorder=15)
ax.add_feature(cfeature.STATES.with_scale('10m'), linewidth=1, edgecolor='black', zorder=15)
ax.add_feature(USCOUNTIES.with_scale('5m'), alpha=0.15, linewidth=0.5, edgecolor='black', zorder=13)

# Define colormap and color normalization for temperature differences
clevs = np.arange(-5, 5.1, 0.2)
cmap = plt.get_cmap('bwr')
norm = col.BoundaryNorm(clevs[clevs != 0], cmap.N)

# Set title for the plot
fig.suptitle(f"2m Temperatures | HRRR vs. Observed", transform=ax.transAxes, fontweight="bold", fontsize=24.5, y=1.075, x=0, ha="left", va="top")
plt.title(f"Init: {init_time_str} |
