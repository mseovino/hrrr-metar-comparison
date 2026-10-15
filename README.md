# hrrr-metar-comparison

Maps HRRR 2 m temperature minus METAR observed temperature at every ASOS/AWOS station
in the Northeast (default box 85–66°W, 36.2–47.2°N), with bias, MAE and RMSE in the title.

```bash
pip install -r requirements.txt
python hrrr-metar-comparison.py                        # latest HRRR f01
python hrrr-metar-comparison.py --valid 2026-10-15T12  # any past hour (AWS archive goes back to 2014)
python hrrr-metar-comparison.py --fhour 6 --no-elev-adjust --show
```

Writes `output/hrrr_minus_obs_<YYYYMMDDHH>_f<FF>.png` and a matching CSV with one row per station.

## Data

- **HRRR**: NOAA's AWS archive (`noaa-hrrr-bdp-pds`). Only the TMP 2 m, HGT surface and
  LAND surface messages are downloaded, using byte ranges from the `.idx` file (~3.5 MB per run).
- **METARs**: Iowa Environmental Mesonet ASOS service, routine and special reports only.

## Method

- **f01 by default.** The f00 analysis has already assimilated these METARs, so it is not an independent check.
- **Observed temperature** comes from the remarks T-group (tenths of °C) when present. Otherwise it is
  the whole-°C body value, which carries up to ±0.9 °F of rounding error. Those stations plot as squares.
- **Time matching.** Each station keeps the one ob closest to the valid time, within ±10 minutes.
- **Grid matching.** Each station is matched to the nearest HRRR land point within 5 km, so coastal sites
  aren't compared against water. Stations off the grid are dropped.
- **Elevation.** The model temperature is moved from HRRR terrain height to station elevation with
  a 6.5 K/km lapse rate. A fixed lapse rate is crude under nighttime inversions, so compare with
  `--no-elev-adjust` for valley stations.
- One hour is a noisy sample. Run it across many hours and average by station to find systematic bias.
