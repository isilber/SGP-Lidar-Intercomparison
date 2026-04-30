"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison plotting functions
===============================================================
"""
from __future__ import annotations

import warnings

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


from real_prof_init import find_instrument_type
from real_prof_io import load_arm_data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_log_var(varname: str) -> bool:
    """Return True if the variable should be rendered with a log colour scale."""
    return "backscatter" in varname or "extinction" in varname


def _safe_norm(
    data: np.ndarray,
    log: bool,
    vmin: float | None = None,
    vmax: float | None = None,
    varname: str = "",
) -> mcolors.Normalize:
    """
    Build a Normalize (or LogNorm) from *data*, ignoring non-finite values.

    For log-scale normalization only positive values are considered.
    Falls back to linear normalization if no positive finite values are found.
    """
    candidates = data[np.isfinite(data) & (data > 0)] if log else data[np.isfinite(data)]
    if candidates.size == 0:
        return mcolors.Normalize(vmin=0, vmax=1)
    # Variable-specific hard-coded defaults
    if "linear_depol_ratio" in varname:
        _vmin = vmin if vmin is not None else 0.0
        _vmax = vmax if vmax is not None else 1.0
    else:
        _vmin = vmin if vmin is not None else float(np.percentile(candidates, 1))
        _vmax = vmax if vmax is not None else float(np.percentile(candidates, 99))
    if log and _vmin > 0 and _vmax > _vmin:
        return mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
    return mcolors.Normalize(vmin=_vmin, vmax=_vmax)


def _plot_curtain(
    ax: plt.Axes,
    da: xr.DataArray,
    norm: mcolors.Normalize,
    cmap: str = "viridis",
    title: str = "",
    ylabel: str = "Range (km)",
) -> plt.cm.ScalarMappable:
    """Render a 2-D (time x spatial) curtain on *ax*.

    The spatial dimension may have any name (``"range"``, ``"height"``, etc.).
    Returns the :class:`~matplotlib.collections.QuadMesh` artist.
    """
    spatial_dims = [d for d in da.dims if d != "time"]
    if not spatial_dims:
        raise ValueError(
            f"DataArray '{da.name}' has no spatial dimension besides 'time'."
        )
    sdim = spatial_dims[0]

    if list(da.dims) != ["time", sdim]:
        da = da.transpose("time", sdim)

    t = da.time.values
    r = da[sdim].values if sdim in da.coords else np.arange(da.sizes[sdim])

    mesh = ax.pcolormesh(t, r, da.values.T, norm=norm, cmap=cmap, shading="nearest")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (UTC)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    return mesh


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _cloud_layer_scatter(
    ax: plt.Axes,
    da: xr.DataArray,
    color: str,
) -> None:
    """Scatter-plot all layers of a ``(time, layer)`` cloud variable on *ax*.

    Assumes values are already in km (as produced by the interpolation
    pipeline when ``interpolate=True``).  NaN layers are skipped silently.
    """
    t = da.time.values
    n_layers = da.sizes["layer"]
    for li in range(n_layers):
        vals = da.isel(layer=li).values
        valid = np.isfinite(vals)
        if not valid.any():
            continue
        ax.scatter(t[valid], vals[valid], s=10, c=color, marker=".", linewidths=0)


def plot_variable_curtain(
    ds: xr.Dataset,
    variable: str,
    overlay_field: str | None = None,
    overlay_levels: int | list = 8,
    overlay_colors: str = "magenta",
    overlay_cloud_layers: bool = True,
    cloud_base_color: str = "k",
    cloud_top_color: str = "lightgray",
    reference_cloud_instrument: str | None = None,
    cmap: str = "viridis",
    shared_norm: bool = False,
    ylim: tuple[float, float] | None = None,
    fig_width: float = 10.0,
    panel_height: float = 3.0,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot curtain panels for every dataset variable whose name contains *variable*.

    One panel is created per matching variable (``qc_`` fields and variables
    without a range dimension are excluded).  When ``shared_norm=False``
    (default) all panels share the same color scale, enabling direct
    inter-instrument comparison.  Variables whose name contains
    ``"backscatter"`` or ``"extinction"`` are rendered with a logarithmic
    color scale.

    When ``overlay_cloud_layers=True`` (default), cloud base and cloud top
    layer variables are overlaid as scatter points (marker size 3).  By
    default each panel uses the cloud layers from the same instrument as the
    plotted variable.  Set *reference_cloud_instrument* to a specific
    instrument prefix (e.g. ``"ceil"``) to overlay that instrument's cloud
    layers on **every** panel instead.

    Parameters
    ----------
    ds : xr.Dataset
        Processed multi-instrument dataset (variables prefixed with instrument
        name, e.g. ``"hsrl_attenuated_backscatter"``).
    variable : str
        Substring to search for in variable names, e.g.
        ``"attenuated_backscatter"``.
    overlay_field : str, optional
        Name of a dataset variable to overlay as iso-contours on every panel
        (e.g. ``"interpolatedsonde_temp"``).
    overlay_levels : int or list, optional
        Number of contour levels or explicit level values.  Default is 8.
    overlay_colors : str, optional
        Color for contour lines.  Default is ``"k"`` (black).
    overlay_cloud_layers : bool, optional
        Overlay cloud base / cloud top scatter points when present.
        Default is ``True``.
    cloud_base_color : str, optional
        Marker color for cloud base points.  Default is ``"white"``.
    cloud_top_color : str, optional
        Marker color for cloud top points.  Default is ``"lightgray"``.
    reference_cloud_instrument : str, optional
        When set, use this instrument's cloud layers on every panel instead
        of matching each panel's own instrument.  E.g. ``"ceil"`` will
        scatter ``ceil_cloud_base`` / ``ceil_cloud_top`` on all panels.
        Ignored when *overlay_cloud_layers* is ``False``.
    cmap : str, optional
        Colormap name.  Default is ``"viridis"``.
    shared_norm : bool, optional
        Share color normalization across all panels.  Default is ``False``.
    ylim : tuple of float, optional
        ``(ymin, ymax)`` range limits applied to every panel.  When ``None``
        (default) matplotlib chooses the limits automatically.
    fig_width : float, optional
        Figure width in inches.  Default is 10.
    panel_height : float, optional
        Height of each panel in inches.  Default is 3.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    _sep = f"_{variable}"
    matches = [
        v for v in ds.data_vars
        if v.endswith(_sep)
        and "_" not in v[: len(v) - len(_sep)]
        and not v.startswith("qc_")
        and "range" in ds[v].dims
    ]
    if not matches:
        raise ValueError(
            f"No range-dimension variables containing '{variable}' found in the dataset."
        )

    log = _is_log_var(variable)

    norm_shared: mcolors.Normalize | None = None
    if shared_norm:
        all_data = np.concatenate([ds[v].values.ravel() for v in matches])
        norm_shared = _safe_norm(all_data, log, varname=variable)

    n = len(matches)
    fig, _axes = plt.subplots(
        n, 1, figsize=(fig_width, panel_height * n), squeeze=False, **kwargs
    )
    axes: np.ndarray = _axes.ravel()

    # Resolve overlay DataArray once
    overlay_da: xr.DataArray | None = None
    if overlay_field is not None:
        if overlay_field in ds:
            overlay_da = ds[overlay_field]
        else:
            warnings.warn(
                f"overlay_field '{overlay_field}' not found in dataset; skipping.",
                stacklevel=2,
            )

    for ax, var in zip(axes, matches):
        da = ds[var]
        panel_norm = norm_shared if shared_norm else _safe_norm(da.values.ravel(), log, varname=var)
        mesh = _plot_curtain(ax, da, norm=panel_norm, cmap=cmap, title=var)
        plt.colorbar(mesh, ax=ax, pad=0.02, label=da.attrs.get("units", ""))

        if overlay_da is not None and "range" in overlay_da.dims:
            ov = overlay_da
            if list(ov.dims) != ["time", "range"]:
                ov = ov.transpose("time", "range")
            cs = ax.contour(
                ov.time.values,
                ov["range"].values,
                ov.values.T,
                levels=overlay_levels,
                colors=overlay_colors,
                linewidths=0.6,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.4g")

        # Overlay cloud base / cloud top scatter points
        if overlay_cloud_layers:
            # If a reference instrument is specified, use its cloud layers on
            # every panel; otherwise derive the prefix from the plotted variable
            # so only the matching instrument's layers are shown.
            if reference_cloud_instrument is not None:
                prefix = f"{reference_cloud_instrument}_"
            else:
                _sep = f"_{variable}"
                prefix = var[: var.index(_sep) + 1] if _sep in var else ""
            for cloud_key, color in (
                ("cloud_base", cloud_base_color),
                ("cloud_top",  cloud_top_color),
            ):
                # Prefer prefixed name, fall back to unprefixed
                cloud_var = next(
                    (c for c in (f"{prefix}{cloud_key}", cloud_key)
                     if c in ds and "layer" in ds[c].dims),
                    None,
                )
                if cloud_var is not None:
                    _cloud_layer_scatter(ax, ds[cloud_var], color=color)

        if ylim is not None:
            ax.set_ylim(ylim)

    fig.tight_layout()
    return fig, axes


def compare_variable_to_orig(
    ds: xr.Dataset,
    variable: str,
    ds_orig: xr.Dataset | None = None,
    data_path: "str | Path | None" = None,
    data_path_template: str | None = None,
    config_dir: str = "./configs",
    match_scale: bool = True,
    cmap: str = "viridis",
    fig_width: float = 14.0,
    panel_height: float = 4.0,
    ylim: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a processed/interpolated variable side-by-side with its original.

    The original field name is read from the ``source_fieldname`` attribute of
    the processed variable.  The original dataset can be supplied directly via
    *ds_orig*, or left as ``None`` to let the function discover and load the
    source data file automatically using the ``source_datastream`` attribute
    together with *data_path* / *data_path_template*.

    Both panels share the same color scale when ``match_scale=True`` (default).
    For NRB-corrected variables—where the source field is physically distinct
    from the output (raw signal vs. normalized backscatter)—pass
    ``match_scale=False`` to use independent color scales.

    Variables whose name contains ``"backscatter"`` or ``"extinction"`` are
    rendered with a logarithmic color scale.

    Parameters
    ----------
    ds : xr.Dataset
        Processed (interpolated) dataset.
    variable : str
        Variable name as it appears in *ds*.
    ds_orig : xr.Dataset or None, optional
        Original dataset.  When ``None``, the function attempts to load the
        source file automatically using *data_path* or *data_path_template*
        together with metadata stored in ``ds[variable].attrs``.
    data_path : str or Path, optional
        Directory containing the original instrument data files.  Used only
        when *ds_orig* is ``None``.
    data_path_template : str, optional
        ARM archive path template (see :func:`real_prof_io.find_arm_files`).
        Used only when *ds_orig* is ``None`` and *data_path* is also ``None``.
    config_dir : str, optional
        Directory containing JSON configuration files.  Default is
        ``"./configs"``.
    match_scale : bool, optional
        Share the color scale between both panels.  Pass ``False`` for
        NRB-corrected variables where source and output are physically
        different quantities.  Default is ``True``.
    cmap : str, optional
        Colormap name.  Default is ``"viridis"``.
    fig_width : float, optional
        Figure width in inches.  Default is 14.
    panel_height : float, optional
        Panel height in inches.  Default is 4.
    ylim : tuple of float, optional
        ``(ymin, ymax)`` y-axis limits applied to both panels.  When ``None``
        (default) limits are derived from the finite data extent across both
        panels.
    **kwargs
        Forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes  (shape (2,))
    """
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in ds.")

    da_proc = ds[variable]
    orig_name = da_proc.attrs.get("source_fieldname", variable)

    # ------------------------------------------------------------------
    # Auto-load original dataset when ds_orig is not supplied
    # ------------------------------------------------------------------
    if ds_orig is None:
        site = ds.attrs.get("site", "")
        facility = ds.attrs.get("facility", "")
        source_ds = da_proc.attrs.get("source_datastream", "")

        if not source_ds:
            raise ValueError(
                f"Cannot auto-load original data: 'source_datastream' attribute "
                f"is missing from ds['{variable}'].  Pass ds_orig explicitly."
            )
        if not site or not facility:
            raise ValueError(
                "Cannot auto-load original data: 'site' and 'facility' global "
                "attributes are missing from ds.  Pass ds_orig explicitly."
            )

        instrument_type = find_instrument_type(
            source_ds, site, facility, config_dir
        )
        if instrument_type is None:
            raise ValueError(
                f"No config in '{config_dir}' matches source_datastream "
                f"'{source_ds}'.  Pass ds_orig explicitly."
            )

        t_min = ds.time.values[0].astype("datetime64[ns]")
        t_max = ds.time.values[-1].astype("datetime64[ns]")

        ds_orig = load_arm_data(
            instrument_type, site, facility, data_path,
            t_min, t_max,
            config_dir=config_dir,
            data_path_template=data_path_template,
        )

    if orig_name not in ds_orig:
        raise ValueError(
            f"Original field '{orig_name}' (source_fieldname of '{variable}') "
            f"not found in ds_orig.  Available: {list(ds_orig.data_vars)}"
        )

    da_orig = ds_orig[orig_name]

    # Convert original range coordinate to km so both panels share the same
    # y-axis scale.  Handles both a 'range' coord and other spatial dim names.
    spatial_orig = [d for d in da_orig.dims if d != "time"]
    if spatial_orig:
        sdim = spatial_orig[0]
        if sdim in da_orig.coords:
            r_units = da_orig[sdim].attrs.get("units", "m")
            if r_units.lower() == "m":
                r_km = da_orig[sdim].values / 1000.0
                da_orig = da_orig.assign_coords({sdim: r_km})

    log = _is_log_var(variable)

    if match_scale:
        all_data = np.concatenate([da_proc.values.ravel(), da_orig.values.ravel()])
        norm = _safe_norm(all_data, log, varname=variable)
        norm_proc = norm_orig = norm
    else:
        norm_proc = _safe_norm(da_proc.values.ravel(), log, varname=variable)
        norm_orig = _safe_norm(da_orig.values.ravel(), log, varname=variable)

    fig, _axes = plt.subplots(
        1, 2, figsize=(fig_width, panel_height), squeeze=False, **kwargs
    )
    axes: np.ndarray = _axes.ravel()

    # Left panel — processed / interpolated
    mesh_proc = _plot_curtain(
        axes[0], da_proc, norm=norm_proc, cmap=cmap,
        title=f"{variable}\n(processed / interpolated)",
    )
    plt.colorbar(mesh_proc, ax=axes[0], pad=0.02, label=da_proc.attrs.get("units", ""))

    # Right panel — original
    src_ds = da_proc.attrs.get("source_datastream", "")
    orig_title = f"{orig_name}\n(original" + (f"  ·  {src_ds}" if src_ds else "") + ")"
    mesh_orig = _plot_curtain(
        axes[1], da_orig, norm=norm_orig, cmap=cmap, title=orig_title,
    )
    plt.colorbar(mesh_orig, ax=axes[1], pad=0.02, label=da_orig.attrs.get("units", ""))

    # Lock both panels to the processed dataset's time extent
    t_min_ds = ds.time.values[0]
    t_max_ds = ds.time.values[-1]
    for _ax in axes:
        _ax.set_xlim(t_min_ds, t_max_ds)

    # Y-axis limits: explicit or derived from finite data extent across both panels
    if ylim is not None:
        y_min, y_max = ylim
    else:
        def _finite_ylim(da):
            sdim = next(d for d in da.dims if d != "time")
            r = da[sdim].values if sdim in da.coords else np.arange(da.sizes[sdim])
            if list(da.dims) != ["time", sdim]:
                da = da.transpose("time", sdim)
            mask = np.any(np.isfinite(da.values), axis=0)
            return float(r[0]), float(r[mask].max() if mask.any() else r[-1])

        y0_p, y1_p = _finite_ylim(da_proc)
        y0_o, y1_o = _finite_ylim(da_orig)
        y_min = min(y0_p, y0_o)
        y_max = max(y1_p, y1_o)
    for _ax in axes:
        _ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Example / quick-look
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob
    from pathlib import Path

    # ------------------------------------------------------------------
    # Configuration — adjust to match your local setup
    # ------------------------------------------------------------------
    SITE            = "sgp"
    FACILITY        = "C1"
    INSTRUMENT_CLASS = "realbundle"
    LEVEL           = "c0"
    OUTPUT_DIR      = "."
    DATA_PATH_TEMPLATE = "/data/archive/{site}/{site}{instrument_class}{facility}.{level}"

    # ------------------------------------------------------------------
    # Locate the most-recent bundle file matching the convention
    # ------------------------------------------------------------------
    pattern = str(
        Path(OUTPUT_DIR)
        / f"{SITE}{INSTRUMENT_CLASS}{FACILITY}.{LEVEL}.*.nc"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No bundle file found matching: {pattern}\n"
            "Run real_prof_main.py first to produce the output file."
        )
    bundle_file = files[-1]
    print(f"Loading: {bundle_file}")
    ds = xr.open_dataset(bundle_file)

    # ------------------------------------------------------------------
    # 1. Curtain plot — attenuated backscatter from all instruments,
    #    with ceil cloud base overlaid on every panel as a reference
    # ------------------------------------------------------------------
    fig1, axes1 = plot_variable_curtain(
        ds,
        variable="attenuated_backscatter",
        overlay_field="interpolatedsonde_temp",
        overlay_levels=[],
        overlay_colors="cyan",
        overlay_cloud_layers=True,
        reference_cloud_instrument="ceil",
        shared_norm=False,
        cmap="jet",
        ylim=(0, 10.0),
    )
    fig1.savefig("curtain_attenuated_backscatter.png", dpi=150, bbox_inches="tight")
    print("Saved: curtain_attenuated_backscatter.png")

    # ------------------------------------------------------------------
    # 2. Comparison plot — processed vs. original for one instrument.
    #    Uses source_fieldname / source_datastream attrs to auto-load
    #    the original file; supply data_path_template for file discovery.
    # ------------------------------------------------------------------
    # Pick the first available attenuated_backscatter variable in the bundle
    comp_var = next(
        (v for v in ds.data_vars if "attenuated_backscatter" in v and not v.startswith("qc_")),
        None,
    )
    if comp_var is not None:
        fig2, axes2 = compare_variable_to_orig(
            ds,
            variable=comp_var,
            data_path_template=DATA_PATH_TEMPLATE,
            match_scale=True,
            cmap="jet",
        )
        out_name = f"compare_{comp_var}.png"
        fig2.savefig(out_name, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_name}")
    else:
        print("No attenuated_backscatter variable found — skipping comparison plot.")

    plt.show()

