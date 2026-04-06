"""
===============================================================
Israel Silber
===============================================================
Wrapper / test script for the lidar profile intercomparison
(REAL) data pipeline.
Each instrument is loaded, corrected (if required), and
interpolated onto a common uniform time x range grid.
Instruments with missing data files are skipped with a warning.
===============================================================
"""

from pathlib import Path
import warnings

import numpy as np

from real_prof_io import load_and_process_arm_data, export_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE       = "sgp"
FACILITY   = "C1"
CONFIG_DIR = str(Path(__file__).parent / "configs")

TIME_MIN = np.datetime64("2026-03-10T20:00:00", "ns")
TIME_MAX = np.datetime64("2026-03-10T21:00:00", "ns")

# ARM nested archive: /data/archive/{site}/{site}{instrument_class}{facility}.{level}/
DATA_PATH_TEMPLATE = "/data/archive/{site}/{site}{instrument_class}{facility}.{level}"

# All instruments for data loading (verify a corresponding JSON config exists)
ALL_INSTRUMENTS = [
    "ceil",
    "ceilpol",
    "dl",
    "hsrl",
    "minimpl",
    "mpl",
    "rl",
    "interpolatedsonde",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> "xr.Dataset":

    print("=" * 70)
    print(f"Site: {SITE}  Facility: {FACILITY}")
    print(f"Period: {TIME_MIN} → {TIME_MAX}")
    print(f"Path template: {DATA_PATH_TEMPLATE}")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = load_and_process_arm_data(
            instrument_type    = ALL_INSTRUMENTS,
            site               = SITE,
            facility           = FACILITY,
            data_path          = None,
            time_min           = TIME_MIN,
            time_max           = TIME_MAX,
            config_dir         = CONFIG_DIR,
            data_path_template = DATA_PATH_TEMPLATE,
            interpolate        = True,
        )

    skipped = [str(w.message) for w in caught]
    if skipped:
        print("\nSkipped instruments:")
        for msg in skipped:
            print(f"  {msg}")

    print(f"\nMerged dataset: {dict(ds.sizes)}  vars={len(ds.data_vars)}")
    if "time" in ds.coords:
        print(f"  time: {str(ds.time.values[0])[:19]} → {str(ds.time.values[-1])[:19]}")
    if "range" in ds.coords:
        print(f"  range: [{float(ds.coords['range'].min()):.3f},"
              f" {float(ds.coords['range'].max()):.3f}] km")

    return ds


if __name__ == "__main__":
    result = main()
    if result.data_vars:
        out = export_dataset(result, site=SITE, facility=FACILITY)
        print(f"\nExported: {out}")
