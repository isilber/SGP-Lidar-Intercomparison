"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison initialization module
===============================================================
"""

import json
from pathlib import Path
from typing import Dict


def load_config(instrument_type: str, config_dir: str = "./configs") -> Dict:
    """
    Load configuration for a specific instrument type.
    
    Parameters
    ----------
    instrument_type : str
        The instrument type (e.g., 'vaisala_ceilometer', 'hsrl', etc.)
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    Dict
        Configuration dictionary loaded from the JSON file
    """
    config_file = Path(config_dir) / f"{instrument_type}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file}: {e}")


def get_variable_mapping(instrument_type: str, config_dir: str = "./configs") -> Dict[str, dict]:
    """
    Get variable mapping for an instrument type.
    
    Parameters
    ----------
    instrument_type : str
        The instrument type
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    Dict[str, dict]
        Dictionary mapping standard names to dicts with ``name_in_file`` and
        ``units`` keys.
    """
    config = load_config(instrument_type, config_dir)
    return config["variables"]


def get_corrections_mapping(instrument_type: str, config_dir: str = "./configs") -> Dict[str, str]:
    """
    Get the corrections field-name mapping for an instrument type.

    Parameters
    ----------
    instrument_type : str
        The instrument type (must have a ``"corrections"`` section in its
        JSON config, e.g. ``"minimpl"``).
    config_dir : str, optional
        Directory containing the JSON configuration files, by default
        ``"./configs"``.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping standard correction keys to instrument-specific
        NetCDF field names.

    Raises
    ------
    KeyError
        If the config file has no ``"corrections"`` section.
    """
    config = load_config(instrument_type, config_dir)
    if "corrections" not in config:
        raise KeyError(
            f"No 'corrections' section found in config for '{instrument_type}'."
        )
    return config["corrections"]


def find_instrument_type(
    source_datastream: str,
    site: str,
    facility: str,
    config_dir: str = "./configs",
) -> str | None:
    """
    Return the instrument_type key whose config matches *source_datastream*.

    Parses ``{instrument_class}`` and ``{level}`` from *source_datastream*
    (formatted as ``{site}{instrument_class}{facility}.{level}``) and scans
    every JSON config in *config_dir* for a matching ``instrument_class`` /
    ``level`` pair.  Returns ``None`` if no match is found.

    Parameters
    ----------
    source_datastream : str
        ARM datastream string, e.g. ``"sgpceil10mC1.b1"``.
    site : str
        ARM site code, e.g. ``"sgp"``.
    facility : str
        ARM facility code, e.g. ``"C1"``.
    config_dir : str, optional
        Directory containing JSON configuration files.  Default is
        ``"./configs"``.

    Returns
    -------
    str or None
        The instrument_type string (JSON filename stem), or ``None`` if no
        matching config is found.
    """
    stem = source_datastream
    if stem.startswith(site):
        stem = stem[len(site):]
    parts = stem.split(".")
    if len(parts) < 2:
        return None
    level = parts[1]                  # e.g. "b1"
    class_fac = parts[0]              # e.g. "ceil10mC1"
    if class_fac.endswith(facility):
        instrument_class = class_fac[: -len(facility)]  # e.g. "ceil10m"
    else:
        return None

    for cfg_path in sorted(Path(config_dir).glob("*.json")):
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            info = cfg.get("instrument_info", {})
            if (
                info.get("instrument_class") == instrument_class
                and info.get("level") == level
            ):
                return cfg_path.stem
        except Exception:
            continue
    return None


def load_facility_pairs(config_dir: str = "./configs") -> Dict:
    """
    Load the facility-pairing configuration.

    Reads ``facility_pairs.json`` from *config_dir* to determine which
    (site, facility) combinations have supplementary facilities. Returns
    an empty dict if the file does not exist.

    Parameters
    ----------
    config_dir : str, optional
        Directory containing JSON configuration files, by default "./configs"

    Returns
    -------
    Dict
        Parsed dictionary with structure::

            {
              "site_code": {
                "facility_code": { "supplementary_facilities": ["fac1", "fac2"] }
              }
            }

        Returns ``{}`` if facility_pairs.json does not exist.
    """
    pairs_file = Path(config_dir) / "facility_pairs.json"

    if not pairs_file.exists():
        return {}

    try:
        with open(pairs_file, "r") as f:
            pairs = json.load(f)
        return pairs
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in facility pairs file {pairs_file}: {e}")


def get_supplementary_facilities(
    site: str, facility: str, config_dir: str = "./configs"
) -> list[str]:
    """
    Get the list of supplementary facilities for a given (site, facility) pair.

    Consults the facility-pairing configuration (loaded via
    :func:`load_facility_pairs`) and returns any supplementary facilities
    configured for the given primary (site, facility) combination.

    Parameters
    ----------
    site : str
        ARM site code, e.g. ``"sgp"``.
    facility : str
        ARM facility code, e.g. ``"C1"``.
    config_dir : str, optional
        Directory containing JSON configuration files, by default "./configs"

    Returns
    -------
    list[str]
        List of supplementary facility codes, or empty list if none are
        defined for this (site, facility) pair.
    """
    pairs = load_facility_pairs(config_dir)
    return pairs.get(site, {}).get(facility, {}).get("supplementary_facilities", [])