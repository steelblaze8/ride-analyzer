"""
Ride file parser — .fit and .gpx
----------------------------------
Parses ride files into a common record format, then computes the metrics
that get fed into the AI coach prompt (ride_analyzer.py).

Key design point: GPX is a location/elevation format. Power, heart rate,
and cadence are NOT part of the core GPX spec — they only exist if the
recording device/app added them as vendor extensions (most commonly
Garmin's TrackPointExtension). So GPX files vary a lot in what data they
actually contain. This parser detects what's available and computes
whatever metrics the data supports, rather than assuming power is always
present.

Dependencies:
    pip install fitparse pandas
    (GPX parsing uses only the standard library's xml.etree)
"""

import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

try:
    import fitparse
except ImportError:
    fitparse = None


# ---------------------------------------------------------------------
# FIT parsing
# ---------------------------------------------------------------------

def parse_fit(path: str) -> list[dict]:
    """Parse a .fit file into a list of per-sample records."""
    if fitparse is None:
        raise ImportError("fitparse is required for .fit files: pip install fitparse")

    fit = fitparse.FitFile(path)
    records = []
    for msg in fit.get_messages("record"):
        row = {f.name: f.value for f in msg}
        records.append({
            "timestamp": row.get("timestamp"),
            "power": row.get("power"),
            "heart_rate": row.get("heart_rate"),
            "cadence": row.get("cadence"),
            "elevation": row.get("altitude") or row.get("enhanced_altitude"),
            "distance": row.get("distance"),
            "speed": row.get("speed") or row.get("enhanced_speed"),
        })
    return records


# ---------------------------------------------------------------------
# GPX parsing
# ---------------------------------------------------------------------

def _localname(tag: str) -> str:
    """Strip XML namespace from a tag, e.g. '{ns}hr' -> 'hr'."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_gpx(path: str) -> list[dict]:
    """
    Parse a .gpx file into the same record shape as parse_fit().
    Power/HR/cadence are pulled from vendor extensions if present;
    they'll be None if the file doesn't include them.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    records = []
    for trkpt in root.iter():
        if _localname(trkpt.tag) != "trkpt":
            continue

        lat = trkpt.get("lat")
        lon = trkpt.get("lon")
        ele = hr = cad = power = timestamp = None

        for child in trkpt:
            name = _localname(child.tag)
            if name == "ele":
                ele = float(child.text)
            elif name == "time":
                timestamp = datetime.fromisoformat(child.text.replace("Z", "+00:00"))
            elif name == "extensions":
                # Extensions are nested; search recursively by localname
                for ext in child.iter():
                    ext_name = _localname(ext.tag).lower()
                    if ext_name in ("hr", "heartrate") and ext.text:
                        hr = int(ext.text)
                    elif ext_name in ("cad", "cadence") and ext.text:
                        cad = int(ext.text)
                    elif ext_name == "power" and ext.text:
                        power = int(ext.text)

        records.append({
            "timestamp": timestamp,
            "power": power,
            "heart_rate": hr,
            "cadence": cad,
            "elevation": ele,
            "distance": None,   # GPX doesn't give cumulative distance directly;
                                 # computed from lat/lon if needed later
            "speed": None,
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
        })

    return records


# ---------------------------------------------------------------------
# Unified loading
# ---------------------------------------------------------------------

def load_ride(path: str) -> list[dict]:
    """Detect file type by extension and parse accordingly."""
    if path.lower().endswith(".fit"):
        return parse_fit(path)
    elif path.lower().endswith(".gpx"):
        return parse_gpx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}. Use .fit or .gpx")


# ---------------------------------------------------------------------
# Metric computation (shared by both file types)
# ---------------------------------------------------------------------

def _haversine_m(lat1, lon1, lat2, lon2):
    import math
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


STOPPED_SPEED_MPS = 0.5  # ~1.8 km/h — below this, treat the rider as stopped


def _derive_speed_series(df, time_deltas):
    """
    Returns a per-sample speed series (m/s) if derivable, else None.
    Prefers the device's own speed field (FIT); falls back to computing
    speed from consecutive GPS points (needed for GPX, which has no
    speed field of its own).
    """
    if "speed" in df and df["speed"].notna().any():
        return df["speed"]

    if {"lat", "lon"}.issubset(df.columns) and df["lat"].notna().any():
        lat, lon = df["lat"].values, df["lon"].values
        speeds = [0.0]
        for i in range(1, len(df)):
            dt = time_deltas.iloc[i]
            if dt <= 0 or pd.isna(lat[i]) or pd.isna(lon[i]) or pd.isna(lat[i - 1]) or pd.isna(lon[i - 1]):
                speeds.append(0.0)
                continue
            dist = _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
            speeds.append(dist / dt)
        return pd.Series(speeds, index=df.index)

    return None


def _build_segments(df, time_deltas, n_segments=4):
    """
    Splits the ride into n_segments equal time chunks (by elapsed time)
    and summarizes each: elevation change, avg power, avg HR. This gives
    the model positional context — e.g. so a climb in the middle of the
    ride isn't mistaken for the character of the whole ride.
    """
    if len(df) < n_segments * 2:
        return None

    start, end = df.index[0], df.index[-1]
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return None

    edges = [start + pd.Timedelta(seconds=total_seconds * i / n_segments) for i in range(n_segments + 1)]
    segments = []
    for i in range(n_segments):
        chunk = df[(df.index >= edges[i]) & (df.index <= edges[i + 1])]
        if chunk.empty:
            continue
        seg = {
            "segment": f"{i + 1}/{n_segments}",
            "elapsed_range_min": (
                round((edges[i] - start).total_seconds() / 60, 1),
                round((edges[i + 1] - start).total_seconds() / 60, 1),
            ),
        }
        if "elevation" in chunk and chunk["elevation"].notna().any():
            elev = chunk["elevation"].dropna()
            seg["elevation_gain_m"] = round(elev.diff().clip(lower=0).sum(), 0)
        if "power" in chunk and chunk["power"].notna().any():
            seg["avg_power_w"] = round(chunk["power"].dropna().mean())
        if "heart_rate" in chunk and chunk["heart_rate"].notna().any():
            seg["avg_hr"] = round(chunk["heart_rate"].dropna().mean())
        segments.append(seg)
    return segments


def compute_metrics(records: list[dict], ftp: int | None = None) -> dict:
    """
    Turns raw per-sample records into the summary metrics used by the
    AI coach prompt. Any metric whose required data isn't present in
    the file is simply omitted (set to None) rather than guessed.
    """
    df = pd.DataFrame(records)
    if df.empty or df["timestamp"].isna().all():
        raise ValueError("No usable timestamped records found in this file.")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")

    elapsed_min = round((df.index[-1] - df.index[0]).total_seconds() / 60, 1)
    time_deltas = df.index.to_series().diff().dt.total_seconds().fillna(0).reset_index(drop=True)

    # --- Moving vs. elapsed time ---
    # If the head unit didn't auto-pause, elapsed time can include hours of
    # stops (rest, food, traffic) that would otherwise make the ride look
    # far more intense (or far longer) than it actually was.
    speed_series = _derive_speed_series(df.reset_index(drop=True), time_deltas)
    moving_min = None
    if speed_series is not None:
        moving_mask = speed_series.reset_index(drop=True) > STOPPED_SPEED_MPS
        moving_min = round(time_deltas[moving_mask].sum() / 60, 1)

    metrics = {
        "elapsed_time_min": elapsed_min,
        "moving_time_min": moving_min,
        "ftp_w": ftp,
        "warnings": [],
    }
    if moving_min is not None and elapsed_min - moving_min > 15:
        metrics["stopped_time_min"] = round(elapsed_min - moving_min, 1)
        metrics["warnings"].append(
            f"Elapsed time ({elapsed_min} min) is significantly longer than moving "
            f"time ({moving_min} min) — the recording likely included stops. "
            f"Analysis should be based on moving time, not elapsed time."
        )
    elif speed_series is None:
        metrics["warnings"].append(
            "Could not determine moving time (no speed or GPS data) — elapsed "
            "time is used as a fallback and may include unrecorded stops."
        )

    # Use moving time as the "duration" for training-load math when we have
    # it; fall back to elapsed time only if moving time couldn't be derived.
    effective_duration_min = moving_min if moving_min is not None else elapsed_min

    # --- Elevation ---
    if "elevation" in df and df["elevation"].notna().any():
        elev = df["elevation"].dropna()
        gain = elev.diff().clip(lower=0).sum()
        metrics["elevation_gain_m"] = round(gain, 0)
    else:
        metrics["warnings"].append("No elevation data found.")

    # --- Heart rate ---
    if "heart_rate" in df and df["heart_rate"].notna().any():
        hr = df["heart_rate"].dropna()
        metrics["avg_hr"] = round(hr.mean())
        metrics["max_hr"] = int(hr.max())

        midpoint = hr.index[0] + (hr.index[-1] - hr.index[0]) / 2
        first_half = hr[hr.index <= midpoint]
        second_half = hr[hr.index > midpoint]
        if len(first_half) > 0 and len(second_half) > 0 and first_half.mean() > 0:
            drift = (second_half.mean() - first_half.mean()) / first_half.mean() * 100
            metrics["hr_drift_pct"] = round(drift, 1)
    else:
        metrics["warnings"].append("No heart rate data found.")

    # --- Power (only present in most FIT files, and GPX files that
    # explicitly include a power meter extension) ---
    if "power" in df and df["power"].notna().any():
        power = df["power"].dropna()
        # Resample to 1-second resolution so rolling windows are meaningful
        # even if the file's recording interval is irregular
        power_1s = power.resample("1s").mean().interpolate()

        metrics["avg_power_w"] = round(power.mean())
        # Normalized Power: 30s rolling avg, raised to 4th power, mean, 4th root
        rolling_30s = power_1s.rolling(30, min_periods=1).mean()
        np_val = (rolling_30s ** 4).mean() ** 0.25
        metrics["normalized_power_w"] = round(np_val)

        if ftp:
            intensity_factor = np_val / ftp
            metrics["intensity_factor"] = round(intensity_factor, 2)
            # Use effective (moving) duration, not raw elapsed time, so long
            # stops don't inflate TSS.
            tss = (effective_duration_min / 60) * intensity_factor**2 * 100
            metrics["tss"] = round(tss)

        for label, window_s in [("best_5s_w", 5), ("best_1min_w", 60),
                                  ("best_5min_w", 300), ("best_20min_w", 1200)]:
            if len(power_1s) >= window_s:
                metrics[label] = round(power_1s.rolling(window_s).mean().max())
    else:
        metrics["warnings"].append(
            "No power data found — this is common for GPX files without a "
            "power meter extension. Power-based metrics (NP, IF, TSS, power "
            "curve) will be unavailable for this ride."
        )

    # --- Segment timeline (positional context, e.g. "climb was in the middle") ---
    segments = _build_segments(df, time_deltas)
    if segments:
        metrics["segments"] = segments

    return metrics


if __name__ == "__main__":
    # Example usage — replace with a real file path
    file_path = "example_ride.fit"  # or "example_ride.gpx"
    ftp = 250  # set to the rider's known FTP, or None if unknown

    records = load_ride(file_path)
    metrics = compute_metrics(records, ftp=ftp)

    for warning in metrics.pop("warnings"):
        print(f"Note: {warning}")
    print(metrics)