"""
===============================================================
Israel Silber
===============================================================
Deadtime-factor extrapolation methods for photon-counting lidars.

Handles count rates above the maximum of the deadtime LUT.  The deadtime
correction itself (LUT interpolation) is applied by
``real_lidar_corrections.apply_deadtime_correction``.
===============================================================
"""
import json

import numpy as np


# ---------------------------------------------------------------------------
# Deadtime extrapolation methods (count rates above the LUT maximum)
# ---------------------------------------------------------------------------
# A method is either a name or a dict (see ``DEADTIME_PRESETS``):
#
# "loglog1" : polynomial of degree ``poly_degree`` (default 1) in log-log
#             space fitted to the last ``n_extrap_samples`` LUT points
# "loglog2" : 2nd-order polynomial in log-log space, usingup to last 5 LUT points
# "family" /  dict(kind="family", blend=w, curvature=lam, n_slope=3, n_curv=5) :
#             Log-log extrapolation through the last LUT point (R0, f0),
#             ln f = ln f0 + s * x + c * x**2, with x = ln(R / R0).  It
#             interpolates between loglog1 and loglog2:
#             blend (0-1): slope from the anchored fit to the last ``n_slope``
#                 points (0, ~loglog1) to the last-interval slope (1, tangent).
#             curvature (0-1): adds the quadratic fit to the last ``n_curv``
#                 points (0 = none, 1 ~ loglog2).
# "tangent" : preset of the family with blend=1, curvature=0
# "linear" /  dict(kind="linear") : straight line in LINEAR space
#             through the last two LUT points:
#             f = f0 + (R - R0) * (f0 - f1) / (R0 - R1)
# "semilog" / dict(kind="semilog", n_fit=5, order=2) : ln f is a polynomial
#             (order 1 or 2) of the count RATE (not log rate), anchored at the
#             last LUT point, least-squares fitted to the last ``n_fit`` points.
#             Aggressive extrapolation that grows faster than any log-log power law.
# "nonpara" / dict(kind="nonpara", n_fit=5, pole=None, max_tau_r=0.98) :
#             non-paralyzable detector model assumption,
#             f = 1 / (1 - tau * R_meas), where R_meas
#             is the measured rate (the LUT count rate), scaled to pass
#             through the last LUT point.  tau is fitted to the last ``n_fit``
#             LUT points (1 - 1/f vs R_meas), or set through ``pole`` = 1 / tau
#             (counts/us, must exceed the LUT maximum).  The factor diverges
#             at R_meas = pole, so tau * R_meas is limited to ``max_tau_r``; rates at or
#             beyond that point get a constant factor (further limited by
#             ``max_factor``).
#
DEADTIME_PRESETS = {
    "tangent": dict(kind="family", blend=1.0, curvature=0.0),
    "linear": dict(kind="linear"),
    "semilog": dict(kind="semilog"),
    "nonpara": dict(kind="nonpara"),
}
DEADTIME_METHODS = ("loglog1", "loglog2") + tuple(DEADTIME_PRESETS)
# Method used by the processing pipeline when none is requested.  "loglog1"
# reproduces the earlier (legacy) behavior.
DEFAULT_DEADTIME_METHOD = "nonpara"
DEADTIME_MAX_FACTOR = 100.0
_FAMILY_DEFAULTS = dict(blend=1.0, curvature=0.0, n_slope=3, n_curv=5,
                        max_factor=DEADTIME_MAX_FACTOR)
_SEMILOG_DEFAULTS = dict(n_fit=5, order=2, max_factor=DEADTIME_MAX_FACTOR)
_NONPARA_DEFAULTS = dict(n_fit=5, pole=None, max_tau_r=0.98,
                         max_factor=1000.0)


def resolve_deadtime_method(method) -> dict:
    """
    Normalize a deadtime extrapolation method to a validated option dict.

    Parameters
    ----------
    method : str or dict
        A name from ``DEADTIME_METHODS``, or a dict with an optional
        ``kind`` key (``"loglog1"``, ``"loglog2"``, ``"linear"``,
        ``"semilog"``, ``"nonpara"`` or ``"family"``, the default) plus the
        options valid for that kind (see the module comment above
        ``DEADTIME_PRESETS``).

    Returns
    -------
    spec : dict
        Validated options with ``kind`` and all defaults filled in,
        including ``max_factor``.

    Raises
    ------
    ValueError
        If the name or kind is unknown, an option is not valid for the kind,
        or an option value is out of range.
    TypeError
        If ``method`` is neither a str nor a dict.
    """
    if isinstance(method, str):
        if method in DEADTIME_PRESETS:
            method = DEADTIME_PRESETS[method]
        elif method in ("loglog1", "loglog2"):
            return dict(kind=method, max_factor=DEADTIME_MAX_FACTOR)
        else:
            raise ValueError(f"unknown deadtime method {method!r}; names: "
                             f"{DEADTIME_METHODS}, or pass a dict with kind='family'")
    if not isinstance(method, dict):
        raise TypeError(f"deadtime method must be a str or dict, got {type(method).__name__}")
    spec = dict(method)
    kind = spec.pop("kind", "family")
    if kind in ("loglog1", "loglog2"):
        spec.setdefault("max_factor", DEADTIME_MAX_FACTOR)
        return dict(kind=kind, **spec)
    if kind == "linear":
        bad = set(spec) - {"max_factor"}
        if bad:
            raise ValueError(f"unknown linear option(s) {sorted(bad)}; valid: ['max_factor']")
        return dict(kind="linear", max_factor=spec.get("max_factor", DEADTIME_MAX_FACTOR))
    if kind in ("nonpara", "semilog"):
        defaults = _NONPARA_DEFAULTS if kind == "nonpara" else _SEMILOG_DEFAULTS
        bad = set(spec) - set(defaults)
        if bad:
            raise ValueError(f"unknown {kind} option(s) {sorted(bad)}; "
                             f"valid: {sorted(defaults)}")
        out = {**defaults, **spec}
        if int(out["n_fit"]) < 3:
            raise ValueError("n_fit must be at least 3")
        if kind == "nonpara":
            if out["pole"] is not None and not float(out["pole"]) > 0:
                raise ValueError("pole must be a positive count rate (counts/us)")
            if not 0.0 < float(out["max_tau_r"]) < 1.0:
                raise ValueError("max_tau_r must be within (0, 1)")
        elif out["order"] not in (1, 2):
            raise ValueError("semilog order must be 1 or 2")
        return dict(kind=kind, **out)
    if kind != "family":
        raise ValueError(f"unknown deadtime kind {kind!r}")
    bad = set(spec) - set(_FAMILY_DEFAULTS)
    if bad:
        raise ValueError(f"unknown family option(s) {sorted(bad)}; "
                         f"valid: {sorted(_FAMILY_DEFAULTS)}")
    out = {**_FAMILY_DEFAULTS, **spec}
    if not (0.0 <= out["blend"] <= 1.0 and 0.0 <= out["curvature"] <= 1.0):
        raise ValueError("blend and curvature must be within [0, 1]")
    return dict(kind="family", **out)


def monotonic_deadtime_factors(
    correction_counts: np.ndarray,
    correction_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove decreasing (non-physical) deadtime LUT factors.

    The factor should not decrease with count rate.  A point lower than any
    earlier one is handled as follows:

    * inside the LUT: re-estimated by linear interpolation (in count rate)
      between the nearest kept points;
    * at the end of the LUT: dropped, so the LUT ends at the last
      non-decreasing point.

    Parameters
    ----------
    correction_counts, correction_factors : np.ndarray
        1D deadtime LUT (count rate in counts/us, multiplicative factor).

    Returns
    -------
    counts, factors : np.ndarray
        Copies of the LUT without decreasing values; possibly shorter than
        the input when trailing points were dropped.
    """
    c = np.asarray(correction_counts, dtype=float)
    f = np.asarray(correction_factors, dtype=float)
    keep = f >= np.maximum.accumulate(f)
    last = int(np.flatnonzero(keep)[-1])   # the first point is always kept
    c, f, keep = c[:last + 1], f[:last + 1], keep[:last + 1]
    if keep.all():
        return c.copy(), f.copy()
    return c.copy(), np.interp(c, c[keep], f[keep])


def make_deadtime_extrapolator(
    correction_counts: np.ndarray,
    correction_factors: np.ndarray,
    method="loglog1",
    poly_degree: int = 1,
    n_extrap_samples: int = 3,
):
    """
    Build the deadtime-factor extrapolation function for count rates above
    the LUT maximum.

    Parameters
    ----------
    correction_counts, correction_factors : np.ndarray
        1D deadtime LUT (count rate in counts/us, multiplicative factor).
    method : str or dict, optional
        A name from ``DEADTIME_METHODS`` or a dict such as
        ``dict(kind="family", blend=0.5, curvature=0.0)`` (see the module
        comment above ``DEADTIME_PRESETS``).  Default ``"loglog1"``.
    poly_degree, n_extrap_samples : int, optional
        Used by ``"loglog1"`` only (see
        :func:`real_lidar_corrections.apply_deadtime_correction`).

    Returns
    -------
    func : callable
        ``func(rate)`` returns the factor for rates (array) above
        ``correction_counts[-1]``.
    description : str
        Short description including fitted parameters.
    """
    spec = resolve_deadtime_method(method)
    kind = spec["kind"]
    c = np.asarray(correction_counts, dtype=float)
    f = np.asarray(correction_factors, dtype=float)
    c_last, f_last = c[-1], f[-1]

    if kind == "loglog1":
        if poly_degree == 0:
            return (lambda x: np.full(np.shape(x), f_last)), "nearest extrapolation to last LUT factor"
        n = min(n_extrap_samples, c.size)
        coeffs = np.polyfit(np.log(c[-n:]), np.log(f[-n:]), poly_degree)
        return (
            (lambda x: np.exp(np.polyval(coeffs, np.log(x)))),
            f"log-log polynomial (degree {poly_degree}) on last {n} LUT points",
        )

    if kind == "loglog2":
        n = min(5, c.size)
        coeffs = np.polyfit(np.log(c[-n:]), np.log(f[-n:]), 2)
        raw = lambda x: np.exp(np.polyval(coeffs, np.log(x)))
        desc = f"log-log 2nd-order polynomial on last {n} LUT points"
    elif kind == "linear":
        slope = float((f[-1] - f[-2]) / (c[-1] - c[-2]))
        raw = lambda x: f_last + slope * (np.asarray(x, dtype=float) - c_last)
        desc = (f"linear in rate through the last two LUT points: slope {slope:.4f} per "
                f"count/us (factor = {f_last:.3f} + {slope:.4f} * (R - {c_last:.2f}))")
    elif kind == "nonpara":
        # Physical model: a non-paralyzable detector with dead time tau
        # (us) measures a rate R_meas = R_true / (1 + R_true * tau), so the
        # correction factor (true / measured) is
        #     f = 1 / (1 - tau * R_meas)
        # which diverges at R_meas = 1 / tau (the "pole").  Equivalently:
        #     1 - 1/f = tau * R_meas
        # i.e. a straight line through the origin with slope tau when
        # y = 1 - 1/f is plotted against R.  That is what is fitted below.
        n = min(int(spec["n_fit"]), c.size)
        if spec["pole"] is None:
            # Least-squares slope of y = tau * rate (a line through the
            # origin), using the last n (highest-rate) LUT points.
            rate = c[-n:]
            y = 1.0 - 1.0 / f[-n:]
            tau = float(np.dot(rate, y) / np.dot(rate, rate))
            src = f"tau fitted to the last {n} LUT points"
        else:
            # User-specified pole = 1 / tau; it must lie above the LUT range,
            # otherwise the model would diverge inside the LUT.
            if not float(spec["pole"]) > c_last:
                raise ValueError(f"pole ({spec['pole']}) must exceed the LUT maximum "
                                 f"({c_last:g} counts/us)")
            tau = 1.0 / float(spec["pole"])
            src = "pole given"
        if not tau > 0:
            raise ValueError("non-paralyzable fit gave a non-positive tau; "
                             "give an explicit 'pole'")
        # Limit tau * R to max_tau_r (< 1) to avoid the singularity at the
        # pole: for R >= max_tau_r / tau, h stays constant (a plateau).
        tr_max = float(spec["max_tau_r"])
        h = lambda x: 1.0 / (1.0 - np.minimum(tau * x, tr_max))
        # The fitted model does not exactly reproduce the LUT at its maximum,
        # so rescale by f_last / h(c_last): the extrapolation then starts
        # exactly at the last LUT factor (continuous with the LUT) and
        # grows with the shape of h for higher rates.
        h0 = float(h(c_last))
        raw = lambda x: f_last * h(x) / h0
        desc = (f"non-paralyzable 1/(1 - tau R), tau = {tau:.5f} us ({src}); pole at "
                f"{1.0 / tau:.1f} counts/us, constant beyond {tr_max / tau:.1f} counts/us")
    elif kind == "semilog":
        n = min(int(spec["n_fit"]), c.size)
        order = int(spec["order"])
        dR = c[-n:] - c_last  # rate offset from the LUT maximum
        Yl = np.log(f[-n:]) - np.log(f_last)  # log factor relative to the last point
        # The fit is forced through the last LUT point
        A = np.stack([dR ** k for k in range(1, order + 1)], axis=1)
        coef = np.linalg.lstsq(A, Yl, rcond=None)[0]

        def raw(x):
            d = np.asarray(x, dtype=float) - c_last
            return f_last * np.exp(sum(coef[k - 1] * d ** k for k in range(1, order + 1)))

        desc = (f"ln(factor) polynomial of rate, order {order}, anchored at the LUT maximum, "
                f"last {n} LUT points: coefficients "
                + ", ".join(f"{v:.4g}" for v in coef))
    else:  # family
        # coordinates relative to the last LUT point
        X = np.log(c) - np.log(c_last)
        Y = np.log(f) - np.log(f_last)
        ns = max(2, min(int(spec["n_slope"]), c.size))
        nc = max(3, min(int(spec["n_curv"]), c.size))
        # anchored log-log slope (least squares through the last LUT point)
        s_anch = float(np.sum(X[-ns:] * Y[-ns:]) / np.sum(X[-ns:] ** 2))
        # log-log slope of the last LUT interval
        s_tan = float((Y[-1] - Y[-2]) / (X[-1] - X[-2]))
        # log-log quadratic fit: a2 * x**2 + a1 * x + const
        a2, a1, _ = np.polyfit(X[-nc:], Y[-nc:], 2)
        w, lam = spec["blend"], spec["curvature"]
        s_base = (1.0 - w) * s_anch + w * s_tan  # blend: anchored -> tangent slope
        s = (1.0 - lam) * s_base + lam * float(a1)  # curvature: toward quadratic-fit slope
        cq = lam * float(a2)  # quadratic term scaled by curvature

        def raw(x):
            xx = np.log(x) - np.log(c_last)
            return f_last * np.exp(s * xx + cq * xx ** 2)

        desc = (f"log-log family blend={w:g}, curvature={lam:g}: slope {s:.2f}, "
                f"quadratic term {cq:.2f} (anchored slope {s_anch:.2f}, "
                f"last-interval slope {s_tan:.2f}, quadratic fit slope {float(a1):.2f})")

    cap = max(float(spec["max_factor"]), f_last)
    return (lambda x: np.minimum(np.maximum(raw(x), f_last), cap)), desc


def deadtime_attributes(
    correction_counts: np.ndarray,
    correction_factors: np.ndarray,
    method=DEFAULT_DEADTIME_METHOD,
    poly_degree: int = 1,
    n_extrap_samples: int = 3,
) -> dict:
    """
    Provenance attributes describing how the deadtime correction was applied.

    Intended for the fields of a dataset whose raw signal was deadtime
    corrected (the corrected signal and anything derived from it).

    Parameters
    ----------
    correction_counts, correction_factors : np.ndarray
        1D deadtime LUT (count rate in counts/us, multiplicative factor).
    method : str or dict, optional
        Extrapolation method above the LUT maximum, as passed to
        :func:`make_deadtime_extrapolator`.  Default
        ``DEFAULT_DEADTIME_METHOD``.
    poly_degree, n_extrap_samples : int, optional
        Options of the ``"loglog1"`` method; recorded only for it.

    Returns
    -------
    dict
        ``deadtime_method`` (the method name, or its ``kind`` when given as a
        dict), ``deadtime_method_options`` (JSON of all resolved options,
        including defaults), ``deadtime_extrapolation`` (description with the
        values fitted to this LUT, e.g. dead time and pole),
        ``deadtime_lut_max_count_rate_counts_per_us`` and
        ``deadtime_lut_max_factor``.

    Raises
    ------
    ValueError, TypeError
        If ``method`` is invalid (see :func:`resolve_deadtime_method`).

    Notes
    -----
    The fitted description is only computed if the method can be evaluated
    for this LUT; otherwise it reads ``"not evaluated: <reason>"`` (the
    correction itself raises only if rates above the LUT maximum occur).
    """
    spec = resolve_deadtime_method(method)
    if spec["kind"] == "loglog1":
        spec = {**spec, "poly_degree": int(poly_degree),
                "n_extrap_samples": int(n_extrap_samples)}
    name = method if isinstance(method, str) else spec["kind"]
    try:
        _, desc = make_deadtime_extrapolator(
            correction_counts, correction_factors, method,
            poly_degree=poly_degree, n_extrap_samples=n_extrap_samples)
    except ValueError as exc:
        desc = f"not evaluated: {exc}"
    c = np.asarray(correction_counts, dtype=float)
    f = np.asarray(correction_factors, dtype=float)
    return {
        "deadtime_method": str(name),
        "deadtime_method_options": json.dumps(spec, sort_keys=True, default=str),
        "deadtime_extrapolation": desc,
        "deadtime_lut_max_count_rate_counts_per_us": round(float(c[-1]), 4),
        "deadtime_lut_max_factor": round(float(f[-1]), 4),
    }
