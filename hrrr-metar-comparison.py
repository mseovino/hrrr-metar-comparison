"""Map HRRR 2 m temperature minus METAR observed temperature for the Northeast.

    python hrrr-metar-comparison.py                       # latest available HRRR f01
    python hrrr-metar-comparison.py --valid 2026-10-15T12 # any past hour back to 2014
    python hrrr-metar-comparison.py --fhour 6 --show

HRRR comes from the NOAA AWS archive (only the TMP 2 m, HGT and LAND fields are
downloaded, via byte ranges from the .idx file). METARs come from the Iowa
Environmental Mesonet. All times are UTC.

Method notes:
- Default is f01, not f00: the f00 analysis has already assimilated these METARs.
- Observed temperature uses the tenths-of-degC T-group from the remarks when present,
  otherwise the whole-degC body value (rounding up to +/-0.9 F).
- Each station keeps the one ob closest to the valid time, within +/-10 min.
- Each station is matched to the nearest HRRR land point, so coastal stations are not
  compared against water temperatures.
- The model temperature is moved from HRRR terrain height to station elevation with
  a 6.5 K/km lapse rate (disable with --no-elev-adjust).
"""
import argparse
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from metpy.plots import USCOUNTIES
from scipy.spatial import cKDTree

HRRR_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf{fh:02d}.grib2"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = (":TMP:2 m above ground:", ":HGT:surface:", ":LAND:surface:")
NETWORKS = [
    *(f"{s}_ASOS" for s in "ME NH VT MA RI CT NY NJ PA DE MD VA WV OH KY NC TN MI IN".split()),
    *(f"CA_{p}_ASOS" for p in "ON QC NB NS".split()),
]
BBOX = (-85.0, -66.0, 36.2, 47.2)  # west, east, south, north
LAPSE_K_PER_M = 0.0065
EARTH_KM = 6371.0
OB_WINDOW = pd.Timedelta("10min")

_TGROUP = r"\bT([01])(\d{3})[01]\d{3}\b"   # remarks T-group, tenths degC
_BODY_T = r"\s(M?\d{2})/(?:M?\d{2})?(?=\s)"  # body TT/TdTd, whole degC

session = requests.Session()


def get(url, **kw):
    r = session.get(url, timeout=120, **kw)
    r.raise_for_status()
    return r


def hrrr_exists(init, fh):
    return session.head(HRRR_URL.format(init=init, fh=fh) + ".idx", timeout=30).status_code == 200


def pick_init(valid, fh):
    """Init time for the requested valid time, or the newest cycle whose valid time has passed."""
    if valid is not None:
        return valid - timedelta(hours=fh)
    newest_valid = datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    for back in range(6):
        init = newest_valid - timedelta(hours=fh + back)
        if hrrr_exists(init, fh):
            return init
    raise SystemExit("No HRRR cycle found on AWS in the last 6 hours")


def download_hrrr(init, fh, cache):
    """Fetch just the needed GRIB messages using byte ranges from the .idx file."""
    out = cache / f"hrrr.{init:%Y%m%d}.t{init:%H}z.f{fh:02d}.subset.grib2"
    if out.exists():
        return out
    url = HRRR_URL.format(init=init, fh=fh)
    lines = get(url + ".idx").text.splitlines()
    starts = [int(l.split(":")[1]) for l in lines]
    chunks = []
    for field in FIELDS:
        i = next((i for i, l in enumerate(lines) if field in l), None)
        if i is None:
            raise SystemExit(f"{field.strip(':')} not in {url}.idx")
        end = f"{starts[i + 1] - 1}" if i + 1 < len(lines) else ""
        chunks.append(get(url, headers={"Range": f"bytes={starts[i]}-{end}"}).content)
    out.write_bytes(b"".join(chunks))
    return out


def load_hrrr(path):
    """Return lat, lon, 2 m temperature (F), terrain height (m) and land mask as 2-D arrays."""
    def open_level(level):
        return xr.open_dataset(path, engine="cfgrib", decode_timedelta=True,
                               backend_kwargs={"indexpath": "", "filter_by_keys": {"typeOfLevel": level}})

    t2m = open_level("heightAboveGround")["t2m"]
    sfc = open_level("surface")
    return dict(
        lat=t2m.latitude.values,
        lon=t2m.longitude.values,
        t_f=(t2m.values - 273.15) * 9 / 5 + 32,
        hgt=sfc["orog"].values,
        land=sfc["lsm"].values > 0.5,
        valid=pd.Timestamp(t2m.valid_time.values),
    )


def fetch_metars(valid, networks):
    """Routine and special METARs within +/-OB_WINDOW of valid (HF 5-min obs excluded)."""
    params = [("network", n) for n in networks] + [
        ("data", "metar"), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
        ("elev", "yes"), ("missing", "M"), ("report_type", "3"), ("report_type", "4"),
        ("sts", f"{valid - OB_WINDOW:%Y-%m-%dT%H:%MZ}"),
        ("ets", f"{valid + OB_WINDOW + pd.Timedelta('1min'):%Y-%m-%dT%H:%MZ}"),
    ]
    text = get(IEM_URL, params=params).text
    if not text.startswith("station,"):
        raise SystemExit(f"Unexpected IEM response: {text[:200]}")
    return pd.read_csv(StringIO(text), dtype={"station": str, "metar": str})


def observed_temps(df, valid, bbox):
    """Parse temperature (tenths when available) and keep one ob per station."""
    w, e, s, n = bbox
    df = df.dropna(subset=["metar"])
    df = df[df.lon.between(w, e) & df.lat.between(s, n)].copy()
    df["valid"] = pd.to_datetime(df["valid"], format="%Y-%m-%d %H:%M")

    parts = df["metar"].str.split(" RMK ", n=1, expand=True)
    tg = (" " + parts[1].fillna("") + " ").str.extract(_TGROUP)
    t_group = pd.to_numeric(tg[1], errors="coerce") / 10
    t_group = t_group.where(tg[0] != "1", -t_group)
    body = (" " + parts[0] + " ").str.extract(_BODY_T)[0]
    t_body = pd.to_numeric(body.str.lstrip("M"), errors="coerce")
    t_body = t_body.where(~body.str.startswith("M", na=False), -t_body)

    df["tenths"] = t_group.notna()
    df["obs_f"] = t_group.fillna(t_body) * 9 / 5 + 32
    df["offset"] = (df["valid"] - valid).abs()
    df = df.dropna(subset=["obs_f"])
    df = df[df["offset"] <= OB_WINDOW]
    # Closest to valid time wins; on ties prefer an ob with a T-group.
    df = df.sort_values(["offset", "tenths"], ascending=[True, False])
    return df.drop_duplicates("station").reset_index(drop=True)


def to_xyz(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def match_to_grid(obs, g, max_km=5.0):
    """Nearest HRRR land point within max_km; stations off the grid or with no land nearby are dropped."""
    land = np.flatnonzero(g["land"].ravel())
    tree = cKDTree(to_xyz(g["lat"].ravel()[land], g["lon"].ravel()[land]))
    dist, k = tree.query(to_xyz(obs.lat.values, obs.lon.values), distance_upper_bound=max_km / EARTH_KM)
    ok = np.isfinite(dist)
    flat = land[k[ok]]
    obs = obs[ok].copy()
    obs["grid_km"] = dist[ok] * EARTH_KM
    obs["hrrr_f"] = g["t_f"].ravel()[flat]
    obs["hrrr_hgt"] = g["hgt"].ravel()[flat]
    return obs, int((~ok).sum())


def plot(obs, valid, init, fh, bbox, elev_adjust, levels, out, show):
    proj = ccrs.LambertConformal(central_longitude=-75, central_latitude=42, standard_parallels=(30, 60))
    fig = plt.figure(figsize=(16, 11), dpi=125)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(bbox, crs=ccrs.PlateCarree())
    water = "#edfbff"
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=water, edgecolor="face")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=water, edgecolor="face")
    ax.add_feature(USCOUNTIES.with_scale("5m"), linewidth=0.3, edgecolor="#b0b0b0")
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.8, edgecolor="#555555")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=1.0, edgecolor="#333333")

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend="both")
    # Obs without a T-group are whole degC; draw them as squares so they can be told apart.
    for tenths, marker in ((True, "o"), (False, "s")):
        sub = obs[obs.tenths == tenths]
        sc = ax.scatter(sub.lon, sub.lat, c=sub["diff"], cmap=cmap, norm=norm, s=46, marker=marker,
                        edgecolors="k", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=5)
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, extend="both", ticks=levels,
                 label="HRRR − observed 2 m temperature (°F)")

    d = obs["diff"]
    adj = "elevation-adjusted" if elev_adjust else "no elevation adjustment"
    ax.set_title(
        f"HRRR f{fh:02d} (init {init:%Y-%m-%d %H}Z) − METAR 2 m temperature, valid {valid:%Y-%m-%d %H:%M}Z\n"
        f"N={len(d)}  bias={d.mean():+.2f}°F  MAE={d.abs().mean():.2f}°F  RMSE={np.sqrt((d**2).mean()):.2f}°F"
        f"   ({adj}; ● T-group tenths, ■ whole °C)",
        fontsize=12,
    )
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def parse_valid(s):
    return datetime.strptime(s, "%Y-%m-%dT%H")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid", type=parse_valid, help="valid time YYYY-MM-DDTHH UTC (default: latest)")
    ap.add_argument("--fhour", type=int, default=1, help="HRRR forecast hour (default 1)")
    ap.add_argument("--bbox", type=float, nargs=4, default=BBOX, metavar=("W", "E", "S", "N"))
    ap.add_argument("--networks", nargs="+", default=NETWORKS, help="IEM ASOS networks to query")
    ap.add_argument("--no-elev-adjust", dest="elev_adjust", action="store_false")
    ap.add_argument("--range", type=float, default=6, help="colorbar half-range in F (default 6)")
    ap.add_argument("--cache", type=Path, default=Path("data"), help="directory for downloaded GRIB")
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument("--show", action="store_true", help="open an interactive window as well")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    a.outdir.mkdir(parents=True, exist_ok=True)

    init = pick_init(a.valid, a.fhour)
    print(f"HRRR init {init:%Y-%m-%d %H}Z f{a.fhour:02d}")
    g = load_hrrr(download_hrrr(init, a.fhour, a.cache))
    valid = g["valid"]

    raw = fetch_metars(valid, a.networks)
    obs = observed_temps(raw, valid, a.bbox)
    obs, off_grid = match_to_grid(obs, g)
    obs["dz_m"] = obs.hrrr_hgt - obs.elevation
    obs["hrrr_adj_f"] = obs.hrrr_f + (LAPSE_K_PER_M * obs.dz_m * 9 / 5 if a.elev_adjust else 0)
    obs["diff"] = obs.hrrr_adj_f - obs.obs_f
    obs = obs.dropna(subset=["diff"])
    print(f"{len(raw)} raw obs -> {len(obs)} stations "
          f"({obs.tenths.mean():.0%} with tenths, {off_grid} dropped off-grid/no land within 5 km)")

    d = obs["diff"]
    print(f"bias {d.mean():+.2f} F, MAE {d.abs().mean():.2f} F, RMSE {np.sqrt((d**2).mean()):.2f} F")
    cols = ["station", "valid", "obs_f", "hrrr_f", "dz_m", "hrrr_adj_f", "diff"]
    print("Largest |HRRR - obs|:")
    print(obs.reindex(d.abs().sort_values(ascending=False).index)[cols].head(10)
          .to_string(index=False, float_format="%.1f"))

    stem = f"hrrr_minus_obs_{valid:%Y%m%d%H}_f{a.fhour:02d}"
    obs.drop(columns="offset").to_csv(a.outdir / f"{stem}.csv", index=False, float_format="%.2f")
    levels = np.arange(-a.range, a.range + 0.5, 1.0)
    plot(obs, valid, init, a.fhour, a.bbox, a.elev_adjust, levels, a.outdir / f"{stem}.png", a.show)


if __name__ == "__main__":
    main()
