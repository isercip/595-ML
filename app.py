from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "PhysioState"
PROFILE_FILE = Path(__file__).parent / "physiostate_profile.json"
DATA_DIR = Path(__file__).parent / "data"
EXAMPLE_FILE = DATA_DIR / "example_session.csv"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=f"{APP_TITLE}: Stress/Overload Estimation",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — modern, high-contrast, fixes BaseWeb dropdown text
# IMPORTANT: do NOT globally color all divs (breaks dropdowns).
# ============================================================
st.markdown(
    """
<style>
/* App background */
[data-testid="stAppViewContainer"]{
  background:linear-gradient(180deg,#f9fbff 0%,#eef4ff 35%,#ffffff 100%);
}
[data-testid="stSidebar"]{
  background:#ffffff;
  border-right:1px solid #e5e7eb;
  box-shadow:6px 0 18px rgba(15,23,42,0.04);
}

/* Typography */
[data-testid="stMarkdownContainer"], label, p, li, h1,h2,h3,h4{
  color:#0f172a !important;
}

/* Top disclaimer bar */
.disclaimer {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 10px 12px;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
  display:flex;
  align-items:center;
  gap:10px;
}
.badge {
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  font-weight:700;
  font-size:12px;
  background:#e0f2fe;
  color:#075985;
  border:1px solid #bae6fd;
}
.badge-red {
  background:#fee2e2;
  color:#991b1b;
  border:1px solid #fecaca;
}
.badge-amber {
  background:#ffedd5;
  color:#9a3412;
  border:1px solid #fed7aa;
}
.panel {
  background:#ffffff;
  border:1px solid #e5e7eb;
  border-radius:18px;
  padding:14px 16px;
  box-shadow:0 8px 20px rgba(15,23,42,0.05);
  margin-bottom:12px;
}
.kpi {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 10px 12px;
  box-shadow: 0 6px 16px rgba(15,23,42,0.05);
}
.kpi .label{ font-size:12px; color:#475569; margin-bottom:4px; font-weight:700; }
.kpi .value{ font-size:26px; color:#0f172a; font-weight:800; line-height:1.1; }
.kpi .sub{ font-size:12px; color:#64748b; margin-top:4px; }

/* ---------- SELECTBOX VISIBILITY FIX ---------- */
.stSelectbox [data-baseweb="select"] > div{
  background-color:#f8fafc !important;
  border-color:#22d3ee !important;
  box-shadow:0 0 0 1px rgba(34,211,238,0.25), 0 10px 24px rgba(15,23,42,0.06);
  transition:border-color 0.12s ease, box-shadow 0.12s ease, background-color 0.12s ease;
}
.stSelectbox [data-baseweb="select"]{
  background-color:#ffffff !important;
  color:#0f172a !important;
}
.stSelectbox [data-baseweb="select"] *{
  background-color:transparent !important;
  color:#0f172a !important;
}
.stSelectbox [data-baseweb="select"] [role="button"],
.stSelectbox [data-baseweb="select"] [role="combobox"],
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div[aria-label]{
  background-color:#ffffff !important;
  color:#0f172a !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within{
  border-color:#0ea5e9 !important;
  box-shadow:0 0 0 3px rgba(34,211,238,0.22) !important;
  background-color:#ecfeff !important;
}
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] *,
[data-baseweb="select"],
[data-baseweb="select"] *,
div[data-baseweb="popover"],
div[data-baseweb="popover"] *,
div[role="listbox"],
div[role="listbox"] *{
  background-color:#ffffff !important;
  color:#0f172a !important;
}
.stSelectbox [data-baseweb="select"] input,
.stSelectbox [data-baseweb="select"] div{
  color:#0f172a !important;
}
.stSelectbox [data-baseweb="select"] svg{ fill:#0f172a !important; }
div[data-baseweb="popover"]{ background-color:#ffffff !important; }
div[role="listbox"]{ background-color:#ffffff !important; }
div[role="option"]{
  background-color:#f8fafc !important;
  color:#0f172a !important;
}
div[role="option"]:hover{ background-color:#e0f2fe !important; }
div[role="option"][aria-selected="true"]{
  background-color:#bae6fd !important;
  color:#0f172a !important;
}

/* Buttons */
.stButton>button{
  border-radius:14px;
  padding:0.7rem 1rem;
  font-weight:800;
  color:#f8fafc;
  text-shadow:0 1px 2px rgba(0,0,0,0.25);
  background:linear-gradient(135deg,#06b6d4 0%,#38bdf8 35%,#7c3aed 100%);
  border:1px solid #22d3ee;
  box-shadow:0 10px 22px rgba(56,189,248,0.25), 0 0 0 1px rgba(124,58,237,0.22), inset 0 1px 0 rgba(255,255,255,0.10);
  letter-spacing:0.02em;
  transition:transform 0.1s ease, box-shadow 0.2s ease, background 0.2s ease;
}
.stButton>button:hover{
  transform:translateY(-1px);
  background:linear-gradient(135deg,#0ea5e9 0%,#22d3ee 45%,#8b5cf6 100%);
  box-shadow:0 12px 28px rgba(56,189,248,0.32), 0 0 0 3px rgba(124,58,237,0.16);
}
.stButton>button:active{
  transform:translateY(0);
  box-shadow:0 6px 18px rgba(56,189,248,0.25), inset 0 1px 0 rgba(0,0,0,0.18);
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# UTILITIES
# ============================================================
def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def infer_sampling_rate_hz(t_sec: np.ndarray) -> Optional[float]:
    if t_sec is None or len(t_sec) < 3:
        return None
    dt = np.diff(t_sec)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    return 1.0 / med


def missing_pct(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 100.0
    return float(series.isna().mean() * 100.0)


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def toast(msg: str, icon: str = "✅"):
    # Streamlit toast is available in newer versions; fall back gracefully.
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.info(f"{icon} {msg}")


# ============================================================
# PHASE SCHEDULE
# ============================================================
PHASE_PRESETS = {
    "Rest -> City -> Highway": [("Rest", 0.33), ("City", 0.33), ("Highway", 0.34)],
    "Calm -> Busy -> Calm": [("Calm", 0.33), ("Busy", 0.34), ("Calm", 0.33)],
    "Constant High": [("High", 0.34), ("High", 0.33), ("High", 0.33)],
}

PHASE_INTENSITY = {
    "Rest": 0.10,
    "Calm": 0.20,
    "Work": 0.45,
    "City": 0.55,
    "Busy": 0.75,
    "Highway": 0.70,
    "High": 0.80,
}


def default_phase_segments(total_min: int, preset: str) -> List[Tuple[str, int]]:
    fracs = PHASE_PRESETS.get(preset, PHASE_PRESETS["Rest -> City -> Highway"])
    mins = [max(1, int(round(total_min * f))) for _, f in fracs]
    # fix rounding to sum exactly total_min
    mins[-1] = max(1, total_min - mins[0] - mins[1])
    labels = [name for name, _ in fracs]
    return [(labels[0], mins[0]), (labels[1], mins[1]), (labels[2], mins[2])]


def normalize_phase_segments(total_min: int, segs: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    # force 3 segments, durations sum to total_min, min 1 minute each
    labels = [segs[i][0] if i < len(segs) else f"Phase {i+1}" for i in range(3)]
    durs = [int(segs[i][1]) if i < len(segs) else 1 for i in range(3)]
    durs = [max(1, d) for d in durs]
    # adjust last to match total
    durs[2] = max(1, total_min - durs[0] - durs[1])
    # if still off (total too small), fix progressively
    while durs[0] + durs[1] + durs[2] > total_min and durs[2] > 1:
        durs[2] -= 1
    while durs[0] + durs[1] + durs[2] < total_min:
        durs[2] += 1
    return [(labels[0], durs[0]), (labels[1], durs[1]), (labels[2], durs[2])]


def phase_segments_to_bounds(segs: List[Tuple[str, int]]) -> List[Dict]:
    # returns list with start_sec/end_sec/label
    bounds = []
    t0 = 0
    for label, dur_min in segs:
        dur_sec = int(dur_min) * 60
        bounds.append({"label": label, "start_sec": t0, "end_sec": t0 + dur_sec})
        t0 += dur_sec
    return bounds


def plot_phase_timeline(segs: List[Tuple[str, int]], total_min: int, key: Optional[str] = None):
    bounds = phase_segments_to_bounds(segs)
    # simple stacked horizontal bar
    fig = go.Figure()
    colors = ["#22c55e", "#f59e0b", "#ef4444"]  # green/amber/red-ish
    left = 0
    for i, b in enumerate(bounds):
        width = (b["end_sec"] - b["start_sec"]) / 60.0
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[""],
                orientation="h",
                marker=dict(color=colors[i % len(colors)]),
                name=f'{b["label"]} ({int(width)} min)',
                hovertemplate=f'{b["label"]}: {int(width)} min<extra></extra>',
            )
        )
    fig.update_layout(
        barmode="stack",
        height=120,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Minutes", range=[0, total_min], showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        legend=dict(orientation="h", y=1.25),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ============================================================
# SIGNAL GENERATION (deterministic, plausible, transparent)
# ============================================================
TARGET_HZ = {"hr_bpm": 1, "resp_bpm": 1, "eda": 4, "emg": 20}


@dataclass
class SessionInputs:
    L_min: int
    seed: int
    context: str
    preset: str
    segments: List[Tuple[str, int]]
    latent_stress_0_100: int

    # required "confounders toggles" (friendly labels)
    conf_caffeine: bool
    conf_poor_sleep: bool
    conf_exercise: bool

    # richer user-friendly detail (used for notes + confidence)
    caffeine_mg: int
    sleep_hours: float
    exercise_type: str
    exercise_min: int
    exercise_intensity: str

    # quality + explicit missing channel toggles
    quality_mode: str  # Good / Noisy / Missing channel
    missing_hr: bool
    missing_eda: bool
    missing_emg: bool
    missing_resp: bool

    # safety symptoms
    sym_chest: bool
    sym_dizzy: bool
    sym_breath: bool


def build_stress_trajectory(t_sec: np.ndarray, phase_bounds: List[Dict], slider_0_100: int, sleep_hours: float, rng: np.random.Generator) -> np.ndarray:
    slider = float(slider_0_100) / 100.0
    sleep_deficit = clamp((7.0 - float(sleep_hours)) / 4.0, 0.0, 1.0)

    phase_level = np.zeros_like(t_sec, dtype=float)
    for b in phase_bounds:
        label = b["label"]
        lvl = PHASE_INTENSITY.get(label, 0.45)
        mask = (t_sec >= b["start_sec"]) & (t_sec < b["end_sec"])
        phase_level[mask] = lvl

    # add slow, deterministic drift (low-frequency noise)
    n = len(t_sec)
    drift = rng.normal(0, 0.02, size=n)
    drift = np.cumsum(drift)
    drift = (drift - drift.min()) / (drift.max() - drift.min() + 1e-9)
    drift = (drift - 0.5) * 0.10  # +/- 0.05

    stress = 0.55 * phase_level + 0.35 * slider + 0.10 * sleep_deficit + drift
    return np.clip(stress, 0.0, 1.0)


def generate_synthetic_session(inp: SessionInputs) -> Dict:
    rng = np.random.default_rng(int(inp.seed))
    total_sec = int(inp.L_min) * 60
    bounds = phase_segments_to_bounds(inp.segments)

    # Time bases
    t_hr = np.arange(0, total_sec, 1.0)                       # 1 Hz
    t_resp = np.arange(0, total_sec, 1.0)                     # 1 Hz
    t_eda = np.arange(0, total_sec, 1.0 / TARGET_HZ["eda"])   # 4 Hz
    t_emg = np.arange(0, total_sec, 1.0 / TARGET_HZ["emg"])   # 20 Hz

    # Stress trajectories per channel time base
    stress_hr = build_stress_trajectory(t_hr, bounds, inp.latent_stress_0_100, inp.sleep_hours, rng)
    stress_resp = build_stress_trajectory(t_resp, bounds, inp.latent_stress_0_100, inp.sleep_hours, rng)
    stress_eda = build_stress_trajectory(t_eda, bounds, inp.latent_stress_0_100, inp.sleep_hours, rng)
    stress_emg = build_stress_trajectory(t_emg, bounds, inp.latent_stress_0_100, inp.sleep_hours, rng)

    # Confounder factors (used for physiology + confidence)
    caffeine_factor = clamp(inp.caffeine_mg / 300.0, 0.0, 1.0) if inp.conf_caffeine else 0.0
    sleep_deficit = clamp((7.0 - inp.sleep_hours) / 4.0, 0.0, 1.0) if inp.conf_poor_sleep else 0.0

    intensity_map = {"Low": 0.4, "Moderate": 0.7, "High": 1.0, "None": 0.0}
    exercise_factor = 0.0
    if inp.conf_exercise and inp.exercise_min > 0:
        base = clamp(inp.exercise_min / 60.0, 0.0, 1.0)
        exercise_factor = base * intensity_map.get(inp.exercise_intensity, 0.7)

    # HR (bpm): baseline + stress + exercise; caffeine increases variability
    hr_base = 60.0 + 18.0 * exercise_factor + 2.0 * caffeine_factor + 2.0 * sleep_deficit
    hr_noise_sd = 1.3 + 1.5 * caffeine_factor + (0.8 if inp.quality_mode == "Noisy" else 0.0)
    hr = hr_base + 35.0 * stress_hr + rng.normal(0, hr_noise_sd, size=len(t_hr))
    # add stress-dependent variability
    hr += rng.normal(0, 2.0 * stress_hr, size=len(t_hr))

    # Resp (breaths/min): baseline + stress + exercise
    resp_base = 12.0 + 2.0 * exercise_factor + 1.0 * sleep_deficit
    resp = resp_base + 10.0 * stress_resp + rng.normal(0, 0.8 + 0.6 * stress_resp, size=len(t_resp))

    # EDA (µS-ish): tonic + phasic spikes
    eda_tonic = 1.5 + 6.0 * stress_eda + 0.6 * caffeine_factor + 0.5 * sleep_deficit
    # stress-dependent spike probability (time-varying) — use rand < p (NOT np.choice with vector p)
    spike_prob = 0.0015 + 0.03 * stress_eda + 0.008 * caffeine_factor
    spike_prob = np.clip(spike_prob, 0.0, 0.20)
    spikes = (rng.random(len(t_eda)) < spike_prob).astype(float)

    # SCR kernel (rise then decay)
    scr_kernel = np.concatenate([np.linspace(0, 1, 12), np.exp(-np.linspace(0, 5, 50))])
    scr = np.convolve(spikes, scr_kernel, mode="same")
    scr_amp = 1.0 + 2.0 * stress_eda
    eda = eda_tonic + 2.8 * scr * scr_amp + rng.normal(0, 0.04 + 0.03 * stress_eda, size=len(t_eda))

    # EMG (a.u.): tension envelope
    emg_base = 0.18 + 0.10 * exercise_factor
    emg = np.abs(emg_base + 0.70 * stress_emg + rng.normal(0, 0.08 + 0.12 * stress_emg, size=len(t_emg)))

    # Quality mode: deterministic dropout bursts
    if inp.quality_mode == "Noisy":
        hr += rng.normal(0, 6.0, size=len(hr))
        resp += rng.normal(0, 1.2, size=len(resp))
        eda += rng.normal(0, 0.25, size=len(eda))
        emg += np.abs(rng.normal(0, 0.25, size=len(emg)))

    elif inp.quality_mode == "Missing channel":
        # choose one channel to "drop out" for a window (deterministic by seed)
        ch = rng.choice(["eda", "emg", "hr", "resp"])
        start = int(0.35 * total_sec)
        end = int(0.55 * total_sec)
        if ch == "eda":
            mask = (t_eda >= start) & (t_eda < end)
            eda[mask] = np.nan
        elif ch == "emg":
            mask = (t_emg >= start) & (t_emg < end)
            emg[mask] = np.nan
        elif ch == "hr":
            mask = (t_hr >= start) & (t_hr < end)
            hr[mask] = np.nan
        else:
            mask = (t_resp >= start) & (t_resp < end)
            resp[mask] = np.nan

    # Explicit channel missing toggles (dominate)
    if inp.missing_hr:
        hr[:] = np.nan
    if inp.missing_resp:
        resp[:] = np.nan
    if inp.missing_eda:
        eda[:] = np.nan
    if inp.missing_emg:
        emg[:] = np.nan

    # Build native per-channel DataFrames
    native = {
        "hr": pd.DataFrame({"t_sec": t_hr, "hr_bpm": hr}),
        "resp": pd.DataFrame({"t_sec": t_resp, "resp_bpm": resp}),
        "eda": pd.DataFrame({"t_sec": t_eda, "eda": eda}),
        "emg": pd.DataFrame({"t_sec": t_emg, "emg": emg}),
    }

    # Canonical 1 Hz DataFrame for pipeline/export
    df1 = pd.DataFrame({"t_sec": t_hr})
    df1 = df1.merge(native["hr"], on="t_sec", how="left")
    df1 = df1.merge(native["resp"], on="t_sec", how="left")

    # EDA: mean per second
    if native["eda"]["eda"].notna().any():
        eda_1hz = native["eda"].copy()
        eda_1hz["t_sec_int"] = eda_1hz["t_sec"].astype(int)
        eda_1hz = eda_1hz.groupby("t_sec_int", as_index=False)["eda"].mean().rename(columns={"t_sec_int": "t_sec"})
        df1 = df1.merge(eda_1hz, on="t_sec", how="left")
    else:
        df1["eda"] = np.nan

    # EMG: RMS per second
    if native["emg"]["emg"].notna().any():
        emg_1hz = native["emg"].copy()
        emg_1hz["t_sec_int"] = emg_1hz["t_sec"].astype(int)
        emg_1hz = (
            emg_1hz.groupby("t_sec_int", as_index=False)["emg"]
            .apply(lambda x: float(np.sqrt(np.nanmean(np.square(x.values)))) if np.isfinite(x.values).any() else np.nan)
            .reset_index()
            .rename(columns={"t_sec_int": "t_sec", "emg": "emg"})
        )
        df1 = df1.merge(emg_1hz, on="t_sec", how="left")
    else:
        df1["emg"] = np.nan

    # Phase at 1 Hz
    phase = []
    for t in df1["t_sec"].values:
        lab = "Unknown"
        for b in bounds:
            if b["start_sec"] <= t < b["end_sec"]:
                lab = b["label"]
                break
        phase.append(lab)
    df1["phase"] = phase

    meta = {
        "source": "synthetic",
        "seed": inp.seed,
        "L_min": inp.L_min,
        "context": inp.context,
        "preset": inp.preset,
        "segments": inp.segments,
        "latent_stress_input": inp.latent_stress_0_100,
        "confounders": {
            "caffeine": inp.conf_caffeine,
            "poor_sleep": inp.conf_poor_sleep,
            "exercise": inp.conf_exercise,
        },
        "details": {
            "caffeine_mg": inp.caffeine_mg,
            "sleep_hours": inp.sleep_hours,
            "exercise_type": inp.exercise_type,
            "exercise_min": inp.exercise_min,
            "exercise_intensity": inp.exercise_intensity,
        },
        "quality_mode": inp.quality_mode,
        "missing_toggles": {
            "hr": inp.missing_hr,
            "eda": inp.missing_eda,
            "emg": inp.missing_emg,
            "resp": inp.missing_resp,
        },
        "phase_bounds": bounds,
    }
    return {"meta": meta, "native": native, "canonical": df1}


# ============================================================
# UPLOAD: schema mapping + validation + resampling
# ============================================================
CANON_COLS = ["t_sec", "hr_bpm", "eda", "emg", "resp_bpm", "phase"]


def coerce_time_to_seconds(col: pd.Series) -> Optional[pd.Series]:
    # Accept numeric seconds or datetime; return float seconds from start.
    if col is None:
        return None
    s = col.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        t0 = s.min()
        return (s - t0).dt.total_seconds().astype(float)
    # try parse datetime strings
    if s.dtype == object:
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() >= max(3, int(0.5 * len(s))):
                t0 = dt.min()
                return (dt - t0).dt.total_seconds().astype(float)
        except Exception:
            pass
    # numeric
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() < 3:
        return None
    # normalize to start at 0
    s_num = s_num.astype(float)
    s_num = s_num - float(s_num.min())
    return s_num


def resample_to_uniform(t_sec: np.ndarray, y: np.ndarray, target_hz: int, method: str = "linear") -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns (t_uniform, y_uniform, did_resample).
    If sampling already close to target, still return a uniform grid for downstream stability.
    """
    if t_sec is None or y is None or len(t_sec) < 3:
        t_u = np.arange(0, 60, 1.0 / target_hz)
        return t_u, np.full_like(t_u, np.nan, dtype=float), True

    t_sec = np.asarray(t_sec, dtype=float)
    y = np.asarray(y, dtype=float)
    # sort by time
    idx = np.argsort(t_sec)
    t_sec = t_sec[idx]
    y = y[idx]

    # remove duplicates
    _, unique_idx = np.unique(t_sec, return_index=True)
    t_sec = t_sec[unique_idx]
    y = y[unique_idx]

    t_max = float(np.nanmax(t_sec)) if np.isfinite(t_sec).any() else 0.0
    dt = 1.0 / float(target_hz)
    t_u = np.arange(0, t_max + 1e-9, dt)

    # if too few finite y, return NaNs
    finite = np.isfinite(y) & np.isfinite(t_sec)
    if finite.sum() < 3:
        return t_u, np.full_like(t_u, np.nan, dtype=float), True

    # interpolate
    if method == "nearest":
        # nearest neighbor via searchsorted
        pos = np.searchsorted(t_sec[finite], t_u, side="left")
        pos = np.clip(pos, 0, finite.sum() - 1)
        y_u = y[finite][pos]
    else:
        y_u = np.interp(t_u, t_sec[finite], y[finite], left=np.nan, right=np.nan)

    did_resample = True
    return t_u, y_u.astype(float), did_resample


def apply_upload_mapping(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> Tuple[Dict, Dict]:
    """
    mapping keys: t_sec, hr_bpm, eda, emg, resp_bpm, phase
    Returns: session dict (like synthetic) + validation dict
    """
    validation = {"checks": [], "details": {}}

    # time
    t_col = mapping.get("t_sec")
    if not t_col or t_col == "None":
        raise ValueError("Time mapping is required.")
    t_sec = coerce_time_to_seconds(df[t_col])
    if t_sec is None:
        raise ValueError("Could not parse time column into seconds.")
    t_sec = t_sec.astype(float)

    # monotone check
    mono = bool((t_sec.diff().fillna(0) >= 0).all())
    validation["checks"].append(("Monotone time", mono))

    # inferred sampling
    fs = infer_sampling_rate_hz(t_sec.values)
    validation["details"]["inferred_sampling_hz"] = fs

    # channels
    native = {}
    resample_notes = {}

    for key, hz in TARGET_HZ.items():
        colname = mapping.get(key, "None")
        if not colname or colname == "None":
            t_u = np.arange(0, float(t_sec.max()) + 1e-9, 1.0 / hz)
            y_u = np.full_like(t_u, np.nan, dtype=float)
            native_key = key.split("_")[0] if key.endswith("_bpm") else key
            native[native_key] = pd.DataFrame({"t_sec": t_u, key: y_u})
            resample_notes[key] = {"mapped": None, "resampled": True, "method": "n/a"}
            continue

        y = pd.to_numeric(df[colname], errors="coerce").astype(float)
        t_u, y_u, did = resample_to_uniform(t_sec.values, y.values, target_hz=hz, method="linear")
        native_key = key.split("_")[0] if key.endswith("_bpm") else key
        native[native_key] = pd.DataFrame({"t_sec": t_u, key: y_u})
        resample_notes[key] = {"mapped": colname, "resampled": did, "method": "linear"}

    # phase (optional)
    phase_col = mapping.get("phase", "None")
    if phase_col and phase_col != "None":
        phase_raw = df[phase_col].astype(str)
        # align phase to 1 Hz time base by nearest time
        t_1 = native["hr"]["t_sec"].values if "hr" in native else np.arange(0, float(t_sec.max()) + 1e-9, 1.0)
        # nearest match
        idx = np.searchsorted(t_sec.values, t_1, side="left")
        idx = np.clip(idx, 0, len(phase_raw) - 1)
        phase_1 = phase_raw.values[idx]
    else:
        phase_1 = np.array(["Unknown"] * len(native["hr"]), dtype=object)

    # Build canonical 1 Hz
    df1 = pd.DataFrame({"t_sec": native["hr"]["t_sec"].values})
    for col in ["hr_bpm", "resp_bpm", "eda"]:
        k = col.split("_")[0] if col.endswith("_bpm") else col
        df1[col] = native[k][col].values if col in native[k].columns else np.nan

    # EMG store as 'emg' already resampled; canonical keeps as emg (a.u.)
    df1["emg"] = native["emg"]["emg"].values if "emg" in native and "emg" in native["emg"].columns else np.nan
    df1["phase"] = phase_1

    # missingness
    miss = {
        "hr_bpm": float(np.mean(~np.isfinite(df1["hr_bpm"].values)) * 100.0),
        "eda": float(np.mean(~np.isfinite(df1["eda"].values)) * 100.0),
        "emg": float(np.mean(~np.isfinite(df1["emg"].values)) * 100.0),
        "resp_bpm": float(np.mean(~np.isfinite(df1["resp_bpm"].values)) * 100.0),
    }
    validation["details"]["missingness_pct"] = miss
    validation["checks"].append(("Has at least one mapped channel", any(v < 100.0 for v in miss.values())))
    validation["details"]["resampling"] = resample_notes

    meta = {
        "source": "upload",
        "seed": None,
        "L_min": int(np.ceil(df1["t_sec"].max() / 60.0)) if len(df1) else None,
        "context": "Unknown",
        "preset": "Uploaded",
        "segments": [],
        "latent_stress_input": None,
        "confounders": {},
        "details": {},
        "quality_mode": "User Provided",
        "missing_toggles": {},
        "phase_bounds": [],
    }
    return {"meta": meta, "native": native, "canonical": df1}, validation


# ============================================================
# FEATURES
# ============================================================
FEATURE_DEFS = [
    ("HR_mean", "Mean heart rate", "bpm"),
    ("HR_std", "HR variability (std)", "bpm"),
    ("HRV_proxy", "RMSSD-like proxy from successive HR diffs", "bpm"),
    ("EDA_tonic_level", "EDA tonic level (low-frequency level)", "a.u."),
    ("EDA_spike_count", "EDA spike count per minute", "#/min"),
    ("EDA_spike_amplitude", "Mean EDA spike amplitude above tonic", "a.u."),
    ("Resp_rate_mean", "Mean respiration rate", "breaths/min"),
    ("Resp_rate_std", "Respiration variability (std)", "breaths/min"),
    ("EMG_RMS", "EMG root-mean-square amplitude", "a.u."),
    ("Data_quality_score", "Overall quality score", "0–1"),
    ("Conf_caffeine", "Caffeine flag", "0/1"),
    ("Conf_poor_sleep", "Poor sleep flag", "0/1"),
    ("Conf_exercise", "Recent exercise flag", "0/1"),
    ("Missing_channels", "Count of missing channels", "0–4"),
]

# scales for standardization in scoring model (defensible, simple)
SCALES = {
    "HR_mean": 10.0,
    "HR_std": 6.0,
    "HRV_proxy": 4.0,
    "EDA_tonic_level": 1.0,
    "EDA_spike_count": 3.0,
    "EDA_spike_amplitude": 0.8,
    "Resp_rate_mean": 3.0,
    "Resp_rate_std": 2.0,
    "EMG_RMS": 0.25,
}

# scoring model weights (transparent); sign rationale in UI
SCORE_WEIGHTS = {
    "HR_mean": 0.22,
    "HR_std": 0.08,
    "HRV_proxy": 0.10,
    "EDA_tonic_level": 0.18,
    "EDA_spike_count": 0.14,
    "EDA_spike_amplitude": 0.06,
    "Resp_rate_mean": 0.12,
    "Resp_rate_std": 0.05,
    "EMG_RMS": 0.10,
}
SCORE_WEIGHT_RATIONALE = {
    "HR_mean": "Higher arousal often elevates mean HR (but can be confounded by exercise).",
    "HR_std": "Stress can increase short-term HR variability in noisy settings.",
    "HRV_proxy": "RMSSD-like proxy changes under load; interpreted cautiously.",
    "EDA_tonic_level": "Sympathetic activation increases tonic conductance.",
    "EDA_spike_count": "More frequent SCRs suggest higher arousal/reactivity.",
    "EDA_spike_amplitude": "Larger phasic responses indicate stronger reactivity.",
    "Resp_rate_mean": "Arousal often increases breathing rate.",
    "Resp_rate_std": "Breathing becomes more variable under load/anxiety.",
    "EMG_RMS": "Tension elevates EMG envelope (jaw/shoulder/forearm).",
}


def compute_eda_spikes(eda_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns: tonic_level, spike_count_per_min, spike_amp_mean
    If insufficient or missing data => (None, None, None)
    """
    if eda_df is None or "eda" not in eda_df.columns or eda_df["eda"].notna().sum() < 20:
        return None, None, None

    x = eda_df["eda"].astype(float).values
    t = eda_df["t_sec"].astype(float).values
    finite = np.isfinite(x) & np.isfinite(t)
    if finite.sum() < 20:
        return None, None, None

    x = x[finite]
    t = t[finite]
    duration_min = max((t.max() - t.min()) / 60.0, 1e-6)

    # tonic via rolling median (approx)
    s = pd.Series(x)
    tonic = float(s.rolling(window=40, min_periods=10, center=True).median().median())
    tonic = float(np.nanmedian(s.values)) if not np.isfinite(tonic) else tonic

    # phasic detection via derivative threshold + refractory
    dx = np.diff(x, prepend=x[0])
    thr = 0.08  # a.u.; tuned for our generator scale
    candidates = np.where(dx > thr)[0]
    if len(candidates) == 0:
        return tonic, 0.0, 0.0

    refractory = int(0.8 * TARGET_HZ["eda"])  # ~0.8s
    spikes_idx = []
    last = -10**9
    for i in candidates:
        if i - last >= refractory:
            spikes_idx.append(i)
            last = i

    spike_count_per_min = float(len(spikes_idx) / duration_min)

    # amplitude: peak in 2s window above tonic
    amps = []
    win = int(2.0 * TARGET_HZ["eda"])
    for i in spikes_idx:
        j = min(len(x) - 1, i + win)
        peak = float(np.nanmax(x[i:j + 1]))
        amp = max(0.0, peak - tonic)
        amps.append(amp)
    spike_amp_mean = float(np.mean(amps)) if len(amps) else 0.0
    return tonic, spike_count_per_min, spike_amp_mean


def compute_emg_rms(emg_df: pd.DataFrame) -> Optional[float]:
    if emg_df is None or "emg" not in emg_df.columns or emg_df["emg"].notna().sum() < 40:
        return None
    x = emg_df["emg"].astype(float).values
    finite = np.isfinite(x)
    if finite.sum() < 40:
        return None
    return float(np.sqrt(np.nanmean(np.square(x[finite]))))


def compute_features(session: Dict, inp: Optional[SessionInputs]) -> Dict[str, Optional[float]]:
    df1 = session["canonical"]

    feats: Dict[str, Optional[float]] = {}

    # HR
    if "hr_bpm" in df1 and df1["hr_bpm"].notna().sum() >= 10:
        hr = df1["hr_bpm"].astype(float)
        feats["HR_mean"] = float(hr.mean())
        feats["HR_std"] = float(hr.std(ddof=1))
        d = np.diff(hr.values)
        d = d[np.isfinite(d)]
        feats["HRV_proxy"] = float(np.sqrt(np.mean(d**2))) if len(d) >= 5 else None
    else:
        feats["HR_mean"] = feats["HR_std"] = feats["HRV_proxy"] = None

    # Resp
    if "resp_bpm" in df1 and df1["resp_bpm"].notna().sum() >= 10:
        r = df1["resp_bpm"].astype(float)
        feats["Resp_rate_mean"] = float(r.mean())
        feats["Resp_rate_std"] = float(r.std(ddof=1))
    else:
        feats["Resp_rate_mean"] = feats["Resp_rate_std"] = None

    # EDA (use native 4 Hz)
    eda_df = session["native"].get("eda", None)
    tonic, scount, samp = compute_eda_spikes(eda_df)
    feats["EDA_tonic_level"] = tonic
    feats["EDA_spike_count"] = scount
    feats["EDA_spike_amplitude"] = samp

    # EMG (native 20 Hz)
    emg_df = session["native"].get("emg", None)
    feats["EMG_RMS"] = compute_emg_rms(emg_df)

    # Confounder flags
    if inp is not None:
        feats["Conf_caffeine"] = 1.0 if inp.conf_caffeine else 0.0
        feats["Conf_poor_sleep"] = 1.0 if inp.conf_poor_sleep else 0.0
        feats["Conf_exercise"] = 1.0 if inp.conf_exercise else 0.0
    else:
        feats["Conf_caffeine"] = feats["Conf_poor_sleep"] = feats["Conf_exercise"] = None

    # Missing channels count
    miss = 0
    for ch, col in [("hr", "hr_bpm"), ("resp", "resp_bpm"), ("eda", "eda"), ("emg", "emg")]:
        if col not in df1 or df1[col].notna().sum() < 5:
            miss += 1
    feats["Missing_channels"] = float(miss)

    # Quality score q in [0,1]
    base_q = 1.0
    if inp is not None:
        if inp.quality_mode == "Noisy":
            base_q = 0.70
        elif inp.quality_mode == "Missing channel":
            base_q = 0.55
        else:
            base_q = 1.0

        # penalize explicit missing toggles
        explicit_missing = sum([inp.missing_hr, inp.missing_eda, inp.missing_emg, inp.missing_resp])
        base_q -= 0.12 * explicit_missing

        # confounders reduce confidence (not necessarily p_stress)
        base_q -= 0.06 * int(inp.conf_caffeine)
        base_q -= 0.08 * int(inp.conf_exercise)
        base_q -= 0.10 * int(inp.conf_poor_sleep)

    # penalize observed missingness (canonical)
    miss_frac = 0.0
    for col in ["hr_bpm", "resp_bpm", "eda", "emg"]:
        if col in df1:
            miss_frac += df1[col].isna().mean()
        else:
            miss_frac += 1.0
    miss_frac /= 4.0
    base_q -= 0.55 * float(miss_frac)

    feats["Data_quality_score"] = float(clamp(base_q, 0.0, 1.0))
    return feats


# ============================================================
# PERSONALIZATION: baseline + online bias update
# ============================================================
DEFAULT_BASELINE = {
    "HR_mean": 70.0,
    "EDA_spike_count": 1.0,
    "Resp_rate_mean": 13.0,
    "EMG_RMS": 0.28,
    "EDA_tonic_level": 2.2,
    "EDA_spike_amplitude": 0.20,
    "HR_std": 4.0,
    "HRV_proxy": 3.0,
    "Resp_rate_std": 1.4,
}


def load_profile() -> Dict:
    if PROFILE_FILE.exists():
        try:
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_profile(profile: Dict) -> None:
    try:
        PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_profile_state() -> Dict:
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    prof = st.session_state.profile
    prof.setdefault("baseline_features", {})
    prof.setdefault("calibration_bias", 0.0)
    prof.setdefault("feedback_history", [])
    return prof


def compute_baseline_from_first_60s(session: Dict) -> Dict[str, float]:
    df1 = session["canonical"]
    df_60 = df1[df1["t_sec"] < 60].copy()
    # create a temporary session object with canonical truncated (reuse native for EDA/EMG)
    tmp = {"canonical": df_60, "native": session["native"], "meta": session["meta"]}
    feats = compute_features(tmp, inp=None)
    # keep only numeric stress-relevant features
    baseline = {}
    for k in DEFAULT_BASELINE.keys():
        v = feats.get(k, None)
        if v is not None and np.isfinite(v):
            baseline[k] = float(v)
    # fill missing with defaults
    for k, v in DEFAULT_BASELINE.items():
        baseline.setdefault(k, float(v))
    return baseline


def update_bias_from_feedback(current_p: float, user_rating_0_10: int, lr: float = 0.12) -> float:
    # online shift of bias term b to better match user self-report (transparent)
    y = clamp(float(user_rating_0_10) / 10.0, 0.0, 1.0)
    err = y - float(current_p)
    return float(lr * err)


# ============================================================
# MODEL MODE 1: Transparent scoring
# ============================================================
def scoring_model(
    feats: Dict[str, Optional[float]],
    baseline: Dict[str, float],
    calibration_bias: float,
) -> Tuple[float, str, Dict[str, float], Dict[str, float]]:
    """
    Returns:
      p_stress, label, standardized z used, contributions (w*z)
    Excludes missing features and renormalizes weights for fairness.
    """
    # build z from available features relative to baseline
    z = {}
    available = []
    for k in SCORE_WEIGHTS.keys():
        fv = feats.get(k, None)
        if fv is None or not np.isfinite(fv):
            continue
        b = float(baseline.get(k, DEFAULT_BASELINE.get(k, 0.0)))
        scale = float(SCALES.get(k, 1.0))
        z[k] = float((float(fv) - b) / (scale if scale > 1e-9 else 1.0))
        available.append(k)

    # if nothing available, return neutral
    if len(available) == 0:
        return 0.50, "Moderate", {}, {}

    # renormalize weights among available features
    w_sum = sum(abs(SCORE_WEIGHTS[k]) for k in available)
    w_eff = {k: (SCORE_WEIGHTS[k] / w_sum) for k in available} if w_sum > 0 else {k: 0.0 for k in available}

    # linear score + bias
    lin = sum(w_eff[k] * z[k] for k in available) + float(calibration_bias)

    p = float(sigmoid(1.6 * lin))  # scale factor for a reasonable range
    p = clamp(p, 0.0, 1.0)

    # label thresholds
    if p < 0.35:
        label = "Low"
    elif p < 0.75:
        label = "Moderate"
    else:
        label = "High"

    contrib = {k: float(w_eff[k] * z[k]) for k in available}
    return p, label, z, contrib


# ============================================================
# CONFIDENCE MODEL (separate from p_stress)
# ============================================================
def confidence_model(feats: Dict[str, Optional[float]], inp: Optional[SessionInputs]) -> Tuple[str, List[str]]:
    reasons = []
    q = feats.get("Data_quality_score", 0.0)
    q = float(q) if q is not None else 0.0

    missing = int(feats.get("Missing_channels", 4.0) or 4.0)
    if missing >= 2:
        reasons.append(f"{missing} channels missing")

    if inp is not None:
        if inp.conf_exercise:
            reasons.append("recent exercise can elevate HR/resp")
        if inp.conf_caffeine:
            reasons.append("caffeine can increase variability")
        if inp.conf_poor_sleep:
            reasons.append("poor sleep elevates baseline arousal")
        if inp.quality_mode == "Noisy":
            reasons.append("noisy signals")
        if inp.quality_mode == "Missing channel":
            reasons.append("dropout observed")

    # map q to discrete confidence
    if q >= 0.78 and missing <= 1:
        level = "High"
    elif q >= 0.52 and missing <= 2:
        level = "Medium"
    else:
        level = "Low"

    # keep reasons concise
    if len(reasons) > 4:
        reasons = reasons[:4]
    if len(reasons) == 0:
        reasons = ["no major data-quality flags"]
    return level, reasons


# ============================================================
# EXPLAINABILITY
# Mode 1: contrib_i = w_i * (z_i - z_i_baseline) ; z_baseline = 0 by construction
# ============================================================
def top_drivers_from_contrib(contrib: Dict[str, float], top_k: int = 6) -> pd.DataFrame:
    if not contrib:
        return pd.DataFrame(columns=["Driver", "Contribution"])
    items = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return pd.DataFrame({"Driver": [k for k, _ in items], "Contribution": [v for _, v in items]})


# ============================================================
# MODE 2: Train-at-runtime classifier
# ============================================================
TRAIN_FEATURES = [
    "HR_mean", "HR_std", "HRV_proxy",
    "EDA_tonic_level", "EDA_spike_count", "EDA_spike_amplitude",
    "Resp_rate_mean", "Resp_rate_std",
    "EMG_RMS",
    "Data_quality_score",
    "Conf_caffeine", "Conf_poor_sleep", "Conf_exercise",
    "Missing_channels",
]


def synthesize_training_dataset(N: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    # create a mild imbalance: more low/moderate than high
    for i in range(N):
        s = float(rng.choice([rng.uniform(0, 55), rng.uniform(40, 80), rng.uniform(70, 100)], p=[0.50, 0.35, 0.15]))
        L = int(rng.integers(6, 11))
        preset = rng.choice(list(PHASE_PRESETS.keys()))
        segs = default_phase_segments(L, preset)

        conf_caf = bool(rng.random() < 0.35)
        conf_slp = bool(rng.random() < 0.25)
        conf_ex = bool(rng.random() < 0.30)

        caffeine_mg = int(rng.integers(60, 250)) if conf_caf else 0
        sleep_hours = float(rng.uniform(4.5, 8.5))
        exercise_type = str(rng.choice(["None", "Strength", "Cardio", "HIIT", "Mixed"]))
        if not conf_ex:
            exercise_type = "None"
        exercise_min = int(rng.integers(10, 70)) if conf_ex else 0
        exercise_intensity = str(rng.choice(["Low", "Moderate", "High"])) if conf_ex else "None"

        quality = str(rng.choice(["Good", "Noisy", "Missing channel"], p=[0.65, 0.25, 0.10]))
        missing_toggles = {
            "missing_hr": bool(rng.random() < 0.06),
            "missing_eda": bool(rng.random() < 0.08),
            "missing_emg": bool(rng.random() < 0.06),
            "missing_resp": bool(rng.random() < 0.06),
        }

        inp = SessionInputs(
            L_min=L,
            seed=int(rng.integers(1, 10**9)),
            context=str(rng.choice(["Driving", "Rest", "Work focus", "Exercise", "Commuting", "Recovery", "Sleep"])),
            preset=preset,
            segments=segs,
            latent_stress_0_100=int(s),
            conf_caffeine=conf_caf,
            conf_poor_sleep=conf_slp,
            conf_exercise=conf_ex,
            caffeine_mg=caffeine_mg,
            sleep_hours=sleep_hours,
            exercise_type=exercise_type,
            exercise_min=exercise_min,
            exercise_intensity=exercise_intensity,
            quality_mode=quality,
            missing_hr=missing_toggles["missing_hr"],
            missing_eda=missing_toggles["missing_eda"],
            missing_emg=missing_toggles["missing_emg"],
            missing_resp=missing_toggles["missing_resp"],
            sym_chest=False,
            sym_dizzy=False,
            sym_breath=False,
        )

        sess = generate_synthetic_session(inp)
        feats = compute_features(sess, inp)

        # label from latent stress with noise (illustrative, not medical)
        latent = s / 100.0
        noise = rng.normal(0, 0.10)  # label noise
        y = clamp(latent + noise, 0.0, 1.0)

        if y < 0.35:
            label = "Low"
        elif y < 0.75:
            label = "Moderate"
        else:
            label = "High"

        row = {k: feats.get(k, None) for k in TRAIN_FEATURES}
        row["y_label"] = label
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def train_runtime_model(df_train: pd.DataFrame, seed: int):
    X = df_train[TRAIN_FEATURES].copy()
    y = df_train["y_label"].copy()

    # impute missing numeric features; keep it lightweight
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.25, random_state=int(seed), stratify=y
    )

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)

    acc = accuracy_score(y_test, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_hat, labels=["Low", "Moderate", "High"])
    report = classification_report(y_test, y_hat, output_dict=True, zero_division=0)

    # permutation importance (lightweight)
    try:
        perm = permutation_importance(clf, X_test, y_test, n_repeats=3, random_state=int(seed), scoring="accuracy")
        imp_vals = perm.importances_mean
    except Exception:
        imp_vals = np.zeros(len(TRAIN_FEATURES), dtype=float)

    importance = pd.DataFrame({"Feature": TRAIN_FEATURES, "Importance": imp_vals}).sort_values("Importance", ascending=False)

    return {
        "imputer": imp,
        "model": clf,
        "metrics": {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1},
        "cm": cm,
        "report": report,
        "importance": importance,
        "labels": ["Low", "Moderate", "High"],
    }


# ============================================================
# RECOMMENDATIONS + INTERVENTION
# ============================================================
def make_recommendations(
    context: str,
    label: str,
    confidence: str,
    reasons: List[str],
    deltas: Dict[str, float],
    top_drivers: List[str],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Return dict with sections:
      do_now, next_30_60, long_term
    Each item includes action + rationale.
    """
    do_now = []
    next_ = []
    long_ = []

    # driver-based rationales
    driver_text = ", ".join(top_drivers[:2]) if top_drivers else "the strongest signals"
    conf_text = f"Confidence is {confidence} due to {', '.join(reasons[:2])}."

    # context-gated actions
    if context == "Driving":
        if label == "High":
            do_now.append({
                "action": "Reduce sensory load (mute audio, simplify the cabin).",
                "rationale": f"Your estimate is high; primary drivers are {driver_text}. {conf_text} Keep attention on safe driving."
            })
            do_now.append({
                "action": "Plan a stop at the next safe location; do paced breathing only when stopped.",
                "rationale": "Breathing exercises can lower arousal, but should never be done while actively driving."
            })
            next_.append({
                "action": "Take a 3–5 minute break when safe; hydrate.",
                "rationale": "Short breaks reduce overload and can stabilize HR/resp."
            })
            long_.append({
                "action": "If this pattern repeats, review sleep and caffeine timing before driving.",
                "rationale": "Poor sleep and stimulants can amplify physiological arousal during routine stress."
            })
        elif label == "Moderate":
            do_now.append({
                "action": "Increase following distance; avoid unnecessary lane changes.",
                "rationale": f"Moderate load with drivers {driver_text}. Safer spacing reduces cognitive demand."
            })
            next_.append({
                "action": "Schedule a short stop within 30–60 minutes if possible.",
                "rationale": "A short pause can prevent escalation from moderate to high load."
            })
            long_.append({
                "action": "Build a pre-drive baseline routine (2 minutes of calm breathing before starting).",
                "rationale": "A consistent baseline improves ‘You vs You’ interpretation and reduces false alarms."
            })
        else:
            do_now.append({
                "action": "Maintain steady pace; keep hydration available.",
                "rationale": "Current estimate is low; continue monitoring without overreacting."
            })
            long_.append({
                "action": "Calibrate baseline during a calm minute to improve personalization.",
                "rationale": "Baseline calibration strengthens self-relevant comparisons and confidence."
            })

    else:  # Work / Rest
        if label == "High":
            do_now.append({
                "action": "Step away for 60–120 seconds; stand and reset posture.",
                "rationale": f"High load driven by {driver_text}. A micro-break reduces physiological momentum."
            })
            do_now.append({
                "action": "Start the 60-second paced breathing intervention.",
                "rationale": "Short, low-effort interventions improve perceived control and can reduce arousal."
            })
            next_.append({
                "action": "Hydrate and do a 3–5 minute walk within the next hour.",
                "rationale": "Movement helps downshift stress physiology, especially after sustained sitting."
            })
            long_.append({
                "action": "If sleep is <6h frequently, prioritize a consistent sleep window for one week.",
                "rationale": "Sleep debt raises baseline arousal and reduces confidence in stress inference."
            })
        elif label == "Moderate":
            do_now.append({
                "action": "Do a 30-second micro-reset (shoulders down, slow exhale).",
                "rationale": f"Moderate load; {driver_text} suggests rising arousal. Quick resets prevent escalation."
            })
            next_.append({
                "action": "Triage tasks: pick one next action; postpone low-value tasks.",
                "rationale": "Cognitive overload often escalates physiological load; task simplification helps."
            })
            long_.append({
                "action": "Use daily baseline calibration (first calm minute) for better personalization.",
                "rationale": "Calibration improves ‘You vs baseline’ deltas and reduces uncertainty."
            })
        else:
            do_now.append({
                "action": "Maintain routine; consider a brief stretch every 30–60 minutes.",
                "rationale": "Low load does not require aggressive interventions."
            })
            long_.append({
                "action": "Calibrate baseline to establish ‘You vs You’ reference.",
                "rationale": "Baseline deltas make the app’s feedback loop meaningful."
            })

    return {"do_now": do_now, "next_30_60": next_, "long_term": long_}


def intervention_effect(session: Dict, seed: int) -> Dict:
    """
    Deterministic, modest improvements applied to signals.
    Returns a NEW session dict with adjusted native + canonical.
    """
    rng = np.random.default_rng(int(seed) + 99991)

    sess2 = {
        "meta": dict(session["meta"]),
        "native": {k: v.copy() for k, v in session["native"].items()},
        "canonical": session["canonical"].copy(),
    }
    sess2["meta"]["intervention"] = {"type": "paced_breathing_60s", "applied": True}

    # modest shifts (do not create large jumps)
    hr_shift = float(rng.uniform(2.5, 4.5))
    resp_shift = float(rng.uniform(0.6, 1.4))
    eda_scale = float(rng.uniform(0.92, 0.97))
    emg_scale = float(rng.uniform(0.92, 0.98))

    # apply across entire session for prototype simplicity
    if "hr" in sess2["native"] and "hr_bpm" in sess2["native"]["hr"]:
        sess2["native"]["hr"]["hr_bpm"] = sess2["native"]["hr"]["hr_bpm"] - hr_shift
    if "resp" in sess2["native"] and "resp_bpm" in sess2["native"]["resp"]:
        sess2["native"]["resp"]["resp_bpm"] = sess2["native"]["resp"]["resp_bpm"] - resp_shift
    if "eda" in sess2["native"] and "eda" in sess2["native"]["eda"]:
        sess2["native"]["eda"]["eda"] = sess2["native"]["eda"]["eda"] * eda_scale
    if "emg" in sess2["native"] and "emg" in sess2["native"]["emg"]:
        sess2["native"]["emg"]["emg"] = sess2["native"]["emg"]["emg"] * emg_scale

    # rebuild canonical from adjusted native (reuse logic: 1 Hz merge)
    # (minimal reconstruction; consistent with generator structure)
    # t_sec base from hr
    if "hr" in sess2["native"]:
        t_hr = sess2["native"]["hr"]["t_sec"].values
    else:
        t_hr = sess2["canonical"]["t_sec"].values

    df1 = pd.DataFrame({"t_sec": t_hr})
    # hr + resp
    if "hr" in sess2["native"]:
        df1 = df1.merge(sess2["native"]["hr"], on="t_sec", how="left")
    if "resp" in sess2["native"]:
        df1 = df1.merge(sess2["native"]["resp"], on="t_sec", how="left")

    # EDA 1Hz mean
    if "eda" in sess2["native"] and sess2["native"]["eda"]["eda"].notna().any():
        eda_1hz = sess2["native"]["eda"].copy()
        eda_1hz["t_sec_int"] = eda_1hz["t_sec"].astype(int)
        eda_1hz = eda_1hz.groupby("t_sec_int", as_index=False)["eda"].mean().rename(columns={"t_sec_int": "t_sec"})
        df1 = df1.merge(eda_1hz, on="t_sec", how="left")
    else:
        df1["eda"] = np.nan

    # EMG 1Hz RMS
    if "emg" in sess2["native"] and sess2["native"]["emg"]["emg"].notna().any():
        emg_1hz = sess2["native"]["emg"].copy()
        emg_1hz["t_sec_int"] = emg_1hz["t_sec"].astype(int)
        emg_1hz = (
            emg_1hz.groupby("t_sec_int", as_index=False)["emg"]
            .apply(lambda x: float(np.sqrt(np.nanmean(np.square(x.values)))) if np.isfinite(x.values).any() else np.nan)
            .reset_index()
            .rename(columns={"t_sec_int": "t_sec", "emg": "emg"})
        )
        df1 = df1.merge(emg_1hz, on="t_sec", how="left")
    else:
        df1["emg"] = np.nan

    # phase carry
    if "phase" in sess2["canonical"].columns:
        df1["phase"] = sess2["canonical"]["phase"].values[: len(df1)]
    else:
        df1["phase"] = "Unknown"

    sess2["canonical"] = df1
    return sess2


# ============================================================
# PLOTS
# ============================================================
def add_phase_overlays(fig: go.Figure, phase_bounds: List[Dict]):
    colors = ["rgba(34,197,94,0.10)", "rgba(245,158,11,0.10)", "rgba(239,68,68,0.10)"]
    for i, b in enumerate(phase_bounds[:3]):
        fig.add_vrect(
            x0=b["start_sec"],
            x1=b["end_sec"],
            fillcolor=colors[i % len(colors)],
            opacity=0.25,
            line_width=0,
            layer="below",
            annotation_text=b["label"],
            annotation_position="top left",
        )


def plot_channel(df: pd.DataFrame, x: str, y: str, title: str, phase_bounds: List[Dict]):
    color_map = {
        "hr_bpm": "#ef4444",  # red
        "resp_bpm": "#2563eb",  # blue
        "eda": "#0ea5e9",  # cyan
        "emg": "#8b5cf6",  # purple
    }
    line_color = color_map.get(y, "#111827")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", line=dict(width=2, color=line_color), name=title))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="t (sec)",
        yaxis_title=y,
        template="plotly_white",
    )
    if phase_bounds:
        add_phase_overlays(fig, phase_bounds)
    st.plotly_chart(fig, use_container_width=True)


def plot_progress_ring(pct: float, label: str):
    pct = clamp(pct, 0.0, 1.0)
    fig = go.Figure(
        go.Pie(
            values=[pct, 1 - pct],
            hole=0.78,
            sort=False,
            direction="clockwise",
            textinfo="none",
            marker=dict(colors=["#2563eb", "rgba(148,163,184,0.25)"]),
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        annotations=[
            dict(text=f"<b>{int(pct*60):d}</b><br><span style='font-size:12px;color:#64748b'>{label}</span>",
                 x=0.5, y=0.5, showarrow=False)
        ],
    )
    return fig


# ============================================================
# SAFETY OVERRIDE
# ============================================================
def safety_override(inp: SessionInputs) -> bool:
    return bool(inp.sym_chest or inp.sym_dizzy or inp.sym_breath)


def render_safety_panel(inp: SessionInputs):
    flagged = []
    if inp.sym_chest:
        flagged.append("chest pain/pressure")
    if inp.sym_dizzy:
        flagged.append("fainting/dizziness")
    if inp.sym_breath:
        flagged.append("severe shortness of breath")

    st.markdown(
        f"""
<div class="panel">
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="badge badge-red">Safety override</span>
    <div style="font-weight:900;font-size:18px;">Urgent safety message</div>
  </div>
  <div style="margin-top:8px;color:#334155;">
    <b>Symptoms flagged:</b> {", ".join(flagged)}<br><br>
    This app is informational only and cannot assess emergencies. If these symptoms are present:
    <ol>
      <li>Stop activity immediately.</li>
      <li>If driving, pull over safely.</li>
      <li>Seek urgent medical evaluation / emergency services.</li>
    </ol>
    Normal recommendations and chat are suppressed during safety events.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE INIT
# ============================================================
def ss_init():
    st.session_state.setdefault("page", "Home (Story)")
    st.session_state.setdefault("developer_mode", False)
    st.session_state.setdefault("intervention_done", False)
    st.session_state.setdefault("intervention_session", None)
    st.session_state.setdefault("upload_df", None)
    st.session_state.setdefault("upload_validation", None)
    st.session_state.setdefault("upload_mapping_applied", False)
    st.session_state.setdefault("data_source_mode", "Synthetic session")
    st.session_state.setdefault("model_mode", "Mode 1 — Transparent scoring")
    st.session_state.setdefault("train_state", None)  # store trained artifacts
    st.session_state.setdefault("chat", {"history": [], "asked": 0})
    st.session_state.setdefault("timeline_version", 0)


ss_init()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
PAGES = [
    "Home (Story)",
    "Inputs",
    "Data Source",
    "Signals",
    "Features",
    "Model",
    "Explainability",
    "Recommendations",
    "Coach Chat",
    "Export",
]

with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.caption("Offline-first prototype · end-to-end ML pipeline on-screen.")

    st.session_state.developer_mode = st.toggle(
        "Developer mode (show reproducibility controls)", value=st.session_state.developer_mode
    )

    st.session_state.page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page))
    st.divider()
    st.caption("Persistent disclaimer:")
    st.markdown("**Informational only; not medical advice.**")
    st.caption("Driving is the default demo; switch context to reuse the same pipeline for any activity.")


# ============================================================
# INPUTS (single source of truth)
# ============================================================
def read_inputs_from_ui() -> SessionInputs:
    # defaults
    L_default = 12
    seed_default = 42

    # keep some inputs in session state for stability across pages
    st.session_state.setdefault("L_min", L_default)
    st.session_state.setdefault("seed", seed_default)
    st.session_state.setdefault("context", "Driving")
    st.session_state.setdefault("preset", "Rest -> City -> Highway")
    st.session_state.setdefault("latent_stress", 40)

    segments_default = default_phase_segments(int(L_default), st.session_state["preset"])
    st.session_state.setdefault("seg_labels", [s[0] for s in segments_default])
    st.session_state.setdefault("seg_durs", [s[1] for s in segments_default])

    st.session_state.setdefault("conf_caffeine", True)
    st.session_state.setdefault("conf_poor_sleep", False)
    st.session_state.setdefault("conf_exercise", False)

    st.session_state.setdefault("caffeine_mg", 120)
    st.session_state.setdefault("sleep_hours", 7.0)
    st.session_state.setdefault("exercise_type", "None")
    st.session_state.setdefault("exercise_min", 0)
    st.session_state.setdefault("exercise_intensity", "Moderate")

    st.session_state.setdefault("quality_mode", "Good")
    st.session_state.setdefault("missing_hr", False)
    st.session_state.setdefault("missing_eda", False)
    st.session_state.setdefault("missing_emg", False)
    st.session_state.setdefault("missing_resp", False)

    st.session_state.setdefault("sym_chest", False)
    st.session_state.setdefault("sym_dizzy", False)
    st.session_state.setdefault("sym_breath", False)

    # values are edited in Inputs page; here we just read current state
    segs = list(zip(st.session_state.seg_labels, st.session_state.seg_durs))
    segs = normalize_phase_segments(int(st.session_state.L_min), segs)

    return SessionInputs(
        L_min=int(st.session_state.L_min),
        seed=int(st.session_state.seed),
        context=str(st.session_state.context),
        preset=str(st.session_state.preset),
        segments=segs,
        latent_stress_0_100=int(st.session_state.latent_stress),

        conf_caffeine=bool(st.session_state.conf_caffeine),
        conf_poor_sleep=bool(st.session_state.conf_poor_sleep),
        conf_exercise=bool(st.session_state.conf_exercise),

        caffeine_mg=int(st.session_state.caffeine_mg),
        sleep_hours=float(st.session_state.sleep_hours),
        exercise_type=str(st.session_state.exercise_type),
        exercise_min=int(st.session_state.exercise_min),
        exercise_intensity=str(st.session_state.exercise_intensity),

        quality_mode=str(st.session_state.quality_mode),
        missing_hr=bool(st.session_state.missing_hr),
        missing_eda=bool(st.session_state.missing_eda),
        missing_emg=bool(st.session_state.missing_emg),
        missing_resp=bool(st.session_state.missing_resp),

        sym_chest=bool(st.session_state.sym_chest),
        sym_dizzy=bool(st.session_state.sym_dizzy),
        sym_breath=bool(st.session_state.sym_breath),
    )


inp = read_inputs_from_ui()


# ============================================================
# DATA SOURCE: decide current session
# ============================================================
@st.cache_data(show_spinner=False)
def cached_synth_session(inp_tuple: Tuple) -> Dict:
    # inp_tuple holds only JSON-serializable items for caching
    inp_obj = SessionInputs(*inp_tuple)  # type: ignore
    return generate_synthetic_session(inp_obj)


def inp_to_tuple(inp: SessionInputs) -> Tuple:
    # match SessionInputs field order
    return (
        inp.L_min, inp.seed, inp.context, inp.preset, inp.segments, inp.latent_stress_0_100,
        inp.conf_caffeine, inp.conf_poor_sleep, inp.conf_exercise,
        inp.caffeine_mg, inp.sleep_hours, inp.exercise_type, inp.exercise_min, inp.exercise_intensity,
        inp.quality_mode, inp.missing_hr, inp.missing_eda, inp.missing_emg, inp.missing_resp,
        inp.sym_chest, inp.sym_dizzy, inp.sym_breath
    )


def load_or_make_example():
    ensure_data_dir()
    if EXAMPLE_FILE.exists():
        return
    # create a deterministic example from generator (ships by creation on first run)
    ex_inp = SessionInputs(
        L_min=8, seed=42, context="Driving", preset="Rest -> City -> Highway",
        segments=default_phase_segments(8, "Rest -> City -> Highway"),
        latent_stress_0_100=45,
        conf_caffeine=True, conf_poor_sleep=False, conf_exercise=False,
        caffeine_mg=120, sleep_hours=7.0, exercise_type="None", exercise_min=0, exercise_intensity="None",
        quality_mode="Good",
        missing_hr=False, missing_eda=False, missing_emg=False, missing_resp=False,
        sym_chest=False, sym_dizzy=False, sym_breath=False,
    )
    sess = generate_synthetic_session(ex_inp)
    # save canonical only (schema)
    df = sess["canonical"][CANON_COLS].copy()
    df.to_csv(EXAMPLE_FILE, index=False)


def get_active_session(inp: SessionInputs) -> Tuple[Dict, Optional[Dict]]:
    mode = st.session_state.data_source_mode

    if mode == "Synthetic session":
        sess = cached_synth_session(inp_to_tuple(inp))
        return sess, None

    if mode == "Load example session":
        load_or_make_example()
        df = pd.read_csv(EXAMPLE_FILE)
        # treat as upload with mapping already aligned to canonical
        mapping = {c: c for c in CANON_COLS}
        sess, val = apply_upload_mapping(df, mapping)
        sess["meta"]["source"] = "example"
        sess["meta"]["context"] = "Driving"
        return sess, val

    # Upload: if applied, use stored mapped session; else fallback to synthetic
    if mode == "Upload CSV/JSON":
        if st.session_state.upload_mapping_applied and "upload_session" in st.session_state:
            return st.session_state.upload_session, st.session_state.upload_validation
        # fallback
        sess = cached_synth_session(inp_to_tuple(inp))
        return sess, None

    sess = cached_synth_session(inp_to_tuple(inp))
    return sess, None


session, upload_validation = get_active_session(inp)


# Intervention session (post) if completed
if st.session_state.intervention_done and st.session_state.intervention_session is not None:
    active_session = st.session_state.intervention_session
else:
    active_session = session


# ============================================================
# PIPELINE: features, personalization, model outputs
# ============================================================
profile = get_profile_state()

# baseline selection: saved profile baseline if present else defaults
baseline = dict(DEFAULT_BASELINE)
baseline.update(profile.get("baseline_features", {}) or {})
calibration_bias = float(profile.get("calibration_bias", 0.0))

# compute features on active session
feats = compute_features(active_session, inp if active_session["meta"]["source"] != "upload" else None)

# deltas (You vs You)
deltas = {}
for k in ["HR_mean", "EDA_spike_count", "Resp_rate_mean", "EMG_RMS"]:
    fv = feats.get(k, None)
    if fv is None or not np.isfinite(fv):
        continue
    deltas[k] = float(fv - float(baseline.get(k, DEFAULT_BASELINE.get(k, 0.0))))

# mode 1 outputs (always available)
p1, label1, z1, contrib1 = scoring_model(feats, baseline, calibration_bias)
conf_level, conf_reasons = confidence_model(feats, inp if active_session["meta"]["source"] != "upload" else None)
drivers_df = top_drivers_from_contrib(contrib1, top_k=7)
top_driver_list = drivers_df["Driver"].tolist() if len(drivers_df) else []


# ============================================================
# HEADER (persistent)
# ============================================================
# status badge
state_badge = "Low" if label1 == "Low" else "Moderate" if label1 == "Moderate" else "High"
badge_class = "badge" if state_badge != "High" else "badge badge-amber"
if state_badge == "High":
    badge_class = "badge badge-red"

st.markdown(
    f"""
<div class="disclaimer">
  <span class="{badge_class}">State: {label1}</span>
  <span class="badge">Confidence: {conf_level}</span>
  <span class="badge">Offline-first</span>
  <div style="margin-left:auto;color:#475569;font-weight:700;">
    Informational only; not medical advice.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Safety override dominates everything
if safety_override(inp):
    render_safety_panel(inp)
    st.stop()


# ============================================================
# HOME (STORY) — pipeline stepper, clickable
# ============================================================
def goto(page: str):
    st.session_state.page = page
    st.rerun()


if st.session_state.page == "Home (Story)":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## On-screen pipeline (click a step)")

    cols = st.columns(8)
    steps = [
        ("1) Inputs", "Inputs"),
        ("2) Signals", "Signals"),
        ("3) Features", "Features"),
        ("4) Model", "Model"),
        ("5) Confidence", "Explainability"),
        ("6) Explanation", "Explainability"),
        ("7) Actions", "Recommendations"),
        ("8) Safety", "Inputs"),
    ]
    for i, (txt, page) in enumerate(steps):
        with cols[i]:
            if st.button(txt, use_container_width=True):
                goto(page)

    st.divider()

    # two-minute trace panel
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
<div class="kpi">
  <div class="label">Stress probability (Mode 1)</div>
  <div class="value">{p1*100:.1f}%</div>
  <div class="sub">Thresholds: Low<35%, Moderate<75%, High≥75%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
<div class="kpi">
  <div class="label">Primary drivers</div>
  <div class="value">{(top_driver_list[0] if top_driver_list else "N/A")}</div>
  <div class="sub">Top 2: {", ".join(top_driver_list[:2]) if top_driver_list else "N/A"}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
<div class="kpi">
  <div class="label">You vs baseline</div>
  <div class="value">{deltas.get("HR_mean", 0.0):+.1f} bpm</div>
  <div class="sub">ΔHR_mean shown; full deltas in Features</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
<div class="kpi">
  <div class="label">Data source</div>
  <div class="value">{active_session["meta"]["source"].title()}</div>
  <div class="sub">Synthetic / upload / example</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### What this prototype demonstrates")
    st.markdown(
        """
- **Immediate feedback loop:** change inputs → signals/features/model update.
- **Personalization:** baseline calibration + “You vs You” deltas + transparent bias update from user feedback.
- **ML course pipeline:** Mode 1 (white-box scoring) + Mode 2 (train-at-runtime classifier with evaluation).
- **Safety:** explicit override suppresses actions/chat when severe symptoms are flagged.
- **Missing channels:** handled explicitly (N/A features, weight renormalization, lower confidence).
- **Driving demo by default; swap context to reuse the same pipeline for work, rest, exercise, recovery, or sleep.**
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INPUTS
# ============================================================
if st.session_state.page == "Inputs":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Inputs (explicit, with units and tooltips)")

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("A) Session controls")
        st.session_state.L_min = st.slider("Session length L (minutes)", 5, 30, int(st.session_state.L_min), help="Controls total duration.")
        if st.session_state.developer_mode:
            st.session_state.seed = st.number_input("Random seed (int)", value=int(st.session_state.seed), step=1, help="Same seed + same inputs → identical outputs.")
        else:
            st.caption("Seed is hidden (Developer mode only) to keep consumer UI clean.")

        st.subheader("B) Context mode")
        st.session_state.context = st.selectbox("Context", ["Driving", "Work", "Rest"], index=["Driving","Rest","Work focus","Exercise","Commuting","Recovery","Sleep"].index(st.session_state.context))

        st.subheader("C) Phase schedule")
        prev_L = int(st.session_state.L_min)
        pc1, pc2 = st.columns([3,1])
        with pc1:
            st.session_state.preset = st.selectbox(
                "Preset (auto-fills the editor; you can still edit)",
                list(PHASE_PRESETS.keys()),
                index=list(PHASE_PRESETS.keys()).index(st.session_state.preset),
            )
        with pc2:
            if st.button("Apply preset", use_container_width=True):
                segs = default_phase_segments(int(st.session_state.L_min), st.session_state.preset)
                st.session_state.seg_labels = [s[0] for s in segs]
                st.session_state.seg_durs = [s[1] for s in segs]
                toast("Phase preset applied.", "dY-")
                st.rerun()

        # quick visual of current 3 phases
        chips = []
        chip_colors = ["#06b6d4", "#6366f1", "#f97316"]
        for i, (lab, dur) in enumerate(zip(st.session_state.seg_labels, st.session_state.seg_durs)):
            c = chip_colors[i % len(chip_colors)]
            chips.append(f"<span style='background:{c}1A;color:{c};border:1px solid {c};border-radius:12px;padding:4px 10px;font-weight:800;'>{lab} - {dur}m</span>")
        st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 2px 0;'>{''.join(chips)}</div>", unsafe_allow_html=True)

        # editor inputs
        prev_durs = tuple(st.session_state.seg_durs)
        prev_labels = tuple(st.session_state.seg_labels)
        l1, l2, l3 = st.columns([1,1,1])
        with l1:
            st.session_state.seg_labels[0] = st.text_input("Segment 1 label", st.session_state.seg_labels[0])
            st.session_state.seg_durs[0] = st.number_input("Segment 1 duration (min)", min_value=1, max_value=60, value=int(st.session_state.seg_durs[0]), step=1)
        with l2:
            st.session_state.seg_labels[1] = st.text_input("Segment 2 label", st.session_state.seg_labels[1])
            st.session_state.seg_durs[1] = st.number_input("Segment 2 duration (min)", min_value=1, max_value=60, value=int(st.session_state.seg_durs[1]), step=1)
        with l3:
            st.session_state.seg_labels[2] = st.text_input("Segment 3 label", st.session_state.seg_labels[2])
            # auto-adjusted; still editable but will be normalized
            st.session_state.seg_durs[2] = st.number_input("Segment 3 duration (min)", min_value=1, max_value=60, value=int(st.session_state.seg_durs[2]), step=1)

        # normalize and display timeline
        segs = normalize_phase_segments(int(st.session_state.L_min), list(zip(st.session_state.seg_labels, st.session_state.seg_durs)))
        st.session_state.seg_labels = [s[0] for s in segs]
        st.session_state.seg_durs = [s[1] for s in segs]
        if (tuple(st.session_state.seg_durs) != prev_durs) or (tuple(st.session_state.seg_labels) != prev_labels) or int(st.session_state.L_min) != prev_L:
            st.session_state.timeline_version = int(st.session_state.timeline_version) + 1
        st.caption("Timeline:")
        plot_phase_timeline(segs, int(st.session_state.L_min), key=f"phase_timeline_{st.session_state.timeline_version}")

    with colB:
        st.subheader("D) Latent stress")
        st.session_state.latent_stress = st.slider("Latent stress s ∈ [0,100]", 0, 100, int(st.session_state.latent_stress),
                                                   help="Synthetic generator only: in real use, stress is inferred, not set.")

        st.subheader("E) Lifestyle confounders (explicit toggles + details)")
        # toggles required by spec
        st.session_state.conf_caffeine = st.checkbox("Caffeine (toggle)", value=bool(st.session_state.conf_caffeine), help="Flag that caffeine is present today.")
        st.session_state.conf_poor_sleep = st.checkbox("Poor sleep (toggle)", value=bool(st.session_state.conf_poor_sleep), help="Flag that sleep was insufficient.")
        st.session_state.conf_exercise = st.checkbox("Recent exercise (toggle)", value=bool(st.session_state.conf_exercise), help="Flag that exercise occurred recently.")

        st.caption("Details (used to make signals more realistic + confidence reasons):")
        st.session_state.caffeine_mg = st.slider("Caffeine amount (mg)", 0, 400, int(st.session_state.caffeine_mg), step=10)
        st.session_state.sleep_hours = st.slider("Hours of sleep last night", 3.0, 10.0, float(st.session_state.sleep_hours), step=0.5)
        st.session_state.exercise_type = st.selectbox("Exercise type", ["None","Strength","Cardio","HIIT","Mixed"],
                                                      index=["None","Strength","Cardio","HIIT","Mixed"].index(st.session_state.exercise_type))
        st.session_state.exercise_min = st.slider("Exercise duration (min)", 0, 120, int(st.session_state.exercise_min), step=5)
        st.session_state.exercise_intensity = st.selectbox("Exercise intensity", ["Low","Moderate","High"],
                                                           index=["Low","Moderate","High"].index(st.session_state.exercise_intensity))

        st.subheader("F) Signal quality + missing channels")
        st.session_state.quality_mode = st.selectbox("Quality ∈ {Good, Noisy, Missing channel}",
                                                     ["Good","Noisy","Missing channel"],
                                                     index=["Good","Noisy","Missing channel"].index(st.session_state.quality_mode))
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.missing_hr = st.checkbox("HR missing", value=bool(st.session_state.missing_hr))
            st.session_state.missing_eda = st.checkbox("EDA missing", value=bool(st.session_state.missing_eda))
        with c2:
            st.session_state.missing_emg = st.checkbox("EMG missing", value=bool(st.session_state.missing_emg))
            st.session_state.missing_resp = st.checkbox("Resp missing", value=bool(st.session_state.missing_resp))

        st.subheader("G) Safety symptoms (override)")
        st.session_state.sym_chest = st.checkbox("Chest pain/pressure", value=bool(st.session_state.sym_chest))
        st.session_state.sym_dizzy = st.checkbox("Fainting/dizziness", value=bool(st.session_state.sym_dizzy))
        st.session_state.sym_breath = st.checkbox("Severe shortness of breath", value=bool(st.session_state.sym_breath))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Signal model notes")
    with st.expander("Open notes"):
        st.markdown(
            """
- HR (1 Hz): baseline + stress component; exercise raises baseline; caffeine increases variability.
- Resp (1 Hz): baseline + stress; exercise raises baseline slightly.
- EDA (4 Hz): tonic rises with stress; phasic spikes occur more often under stress (time-varying probability).
- EMG (20 Hz): envelope scales with tension; modeled as a positive-valued noisy process.
- Quality “Noisy”: adds artifacts; “Missing channel”: introduces dropout window; explicit missing toggles dominate.
- All randomness is **seeded** to ensure reproducibility (Developer mode exposes the seed).
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DATA SOURCE
# ============================================================
if st.session_state.page == "Data Source":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Data Source (pluggable)")

    st.session_state.data_source_mode = st.selectbox(
        "Choose a source",
        ["Synthetic session", "Upload CSV/JSON", "Load example session"],
        index=["Synthetic session","Upload CSV/JSON","Load example session"].index(st.session_state.data_source_mode),
        help="Default is synthetic; upload is bring-your-own-data; example ships from generator (created on first run).",
    )

    if st.session_state.data_source_mode == "Synthetic session":
        st.info("Using deterministic synthetic signals. Adjust Inputs to see immediate changes.")

    elif st.session_state.data_source_mode == "Load example session":
        load_or_make_example()
        st.success("Example session available (generated by the app’s own generator).")

    else:
        st.markdown("### Upload flow (product-grade)")
        up = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if up is not None:
            try:
                if up.name.lower().endswith(".json"):
                    raw = json.load(up)
                    dfu = pd.DataFrame(raw)
                else:
                    dfu = pd.read_csv(up)
                st.session_state.upload_df = dfu
                st.session_state.upload_mapping_applied = False
                toast("Upload successful. Map your schema below.", "📥")
            except Exception as e:
                st.error(f"Upload failed: {e}")

        dfu = st.session_state.upload_df
        if dfu is not None:
            st.markdown("#### Preview")
            st.dataframe(dfu.head(12), use_container_width=True)
            st.caption(f"Columns: {list(dfu.columns)}")

            st.markdown("#### Schema mapper")
            cols = ["None"] + list(dfu.columns)
            m = {}
            c1, c2, c3 = st.columns(3)
            with c1:
                m["t_sec"] = st.selectbox("Time (required)", cols, index=(cols.index("t_sec") if "t_sec" in cols else 0))
                m["hr_bpm"] = st.selectbox("HR (bpm)", cols, index=(cols.index("hr_bpm") if "hr_bpm" in cols else 0))
            with c2:
                m["eda"] = st.selectbox("EDA", cols, index=(cols.index("eda") if "eda" in cols else 0))
                m["emg"] = st.selectbox("EMG", cols, index=(cols.index("emg") if "emg" in cols else 0))
            with c3:
                m["resp_bpm"] = st.selectbox("Resp (breaths/min)", cols, index=(cols.index("resp_bpm") if "resp_bpm" in cols else 0))
                m["phase"] = st.selectbox("Phase (optional)", cols, index=(cols.index("phase") if "phase" in cols else 0))

            st.markdown("#### Validation checklist")
            if st.button("Apply mapping"):
                try:
                    sess_u, val = apply_upload_mapping(dfu, m)
                    st.session_state.upload_session = sess_u
                    st.session_state.upload_validation = val
                    st.session_state.upload_mapping_applied = True
                    toast("Mapping applied. Downstream pipeline recomputed.", "✅")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

            if st.session_state.upload_mapping_applied and st.session_state.upload_validation is not None:
                val = st.session_state.upload_validation
                for name, ok in val["checks"]:
                    st.write(("✅ " if ok else "❌ ") + name)
                st.caption(f"Inferred sampling (time column): {val['details'].get('inferred_sampling_hz')}")
                st.caption(f"Missingness (%): {val['details'].get('missingness_pct')}")
                with st.expander("Resampling details"):
                    st.json(val["details"].get("resampling", {}))

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SIGNALS
# ============================================================
if st.session_state.page == "Signals":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Signals (Plotly + phase overlays)")

    bounds = active_session["meta"].get("phase_bounds", [])
    n_hr = active_session["native"].get("hr", pd.DataFrame())
    n_eda = active_session["native"].get("eda", pd.DataFrame())
    n_emg = active_session["native"].get("emg", pd.DataFrame())
    n_resp = active_session["native"].get("resp", pd.DataFrame())

    # summary cards
    df1 = active_session["canonical"]
    c1, c2, c3, c4 = st.columns(4)
    hr_m = feats.get("HR_mean", None)
    eda_t = feats.get("EDA_tonic_level", None)
    resp_m = feats.get("Resp_rate_mean", None)
    emg_r = feats.get("EMG_RMS", None)

    def card(col, label, value, sub):
        with col:
            st.markdown(
                f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    card(c1, "HR mean", f"{hr_m:.1f} bpm" if hr_m is not None else "N/A", f"Missing: {missing_pct(df1.get('hr_bpm', pd.Series(dtype=float))):.0f}%")
    card(c2, "EDA tonic", f"{eda_t:.2f}" if eda_t is not None else "N/A", f"Missing: {missing_pct(df1.get('eda', pd.Series(dtype=float))):.0f}%")
    card(c3, "Resp mean", f"{resp_m:.1f} br/min" if resp_m is not None else "N/A", f"Missing: {missing_pct(df1.get('resp_bpm', pd.Series(dtype=float))):.0f}%")
    card(c4, "EMG RMS", f"{emg_r:.3f}" if emg_r is not None else "N/A", f"Missing: {missing_pct(df1.get('emg', pd.Series(dtype=float))):.0f}%")

    st.divider()

    if not n_hr.empty and "hr_bpm" in n_hr.columns:
        plot_channel(n_hr, "t_sec", "hr_bpm", "Heart Rate (1 Hz)", bounds)
    if not n_resp.empty and "resp_bpm" in n_resp.columns:
        plot_channel(n_resp, "t_sec", "resp_bpm", "Respiration (1 Hz)", bounds)
    if not n_eda.empty and "eda" in n_eda.columns:
        plot_channel(n_eda, "t_sec", "eda", "EDA (4 Hz, tonic + phasic)", bounds)
    if not n_emg.empty and "emg" in n_emg.columns:
        # For EMG, show a downsampled view for performance
        emg_plot = n_emg.iloc[::5].copy()
        plot_channel(emg_plot, "t_sec", "emg", "EMG (20 Hz, amplitude envelope)", bounds)

    with st.expander("Missingness + quality notes"):
        st.write(f"Data quality score q = **{feats.get('Data_quality_score', 0.0):.2f}**")
        st.write(f"Confidence = **{conf_level}** because: {', '.join(conf_reasons)}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FEATURES
# ============================================================
if st.session_state.page == "Features":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Features (intermediate computations)")

    rows = []
    for k, desc, unit in FEATURE_DEFS:
        v = feats.get(k, None)
        rows.append({"Feature": k, "Value": (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)), "Unit": unit, "Definition": desc})

    feat_table = pd.DataFrame(rows)
    st.dataframe(feat_table, use_container_width=True)

    st.divider()
    st.markdown("### Personalization: baseline + You vs baseline deltas")
    b = baseline
    delta_rows = []
    for k in ["HR_mean", "EDA_spike_count", "Resp_rate_mean", "EMG_RMS"]:
        fv = feats.get(k, None)
        if fv is None or not np.isfinite(fv):
            delta_rows.append({"Metric": k, "Current": "N/A", "Baseline": f"{b.get(k, DEFAULT_BASELINE.get(k, 0.0)):.3f}", "Delta": "N/A"})
        else:
            basev = float(b.get(k, DEFAULT_BASELINE.get(k, 0.0)))
            delta_rows.append({"Metric": k, "Current": f"{float(fv):.3f}", "Baseline": f"{basev:.3f}", "Delta": f"{float(fv)-basev:+.3f}"})
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)

    st.divider()
    st.markdown("### Baseline calibration")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Calibrate baseline from first 60 seconds"):
            new_base = compute_baseline_from_first_60s(active_session)
            profile["baseline_features"] = new_base
            st.session_state.profile = profile
            save_profile(profile)
            toast("Baseline calibrated and saved to profile.json.", "🧠")
            st.rerun()
    with col2:
        if st.button("Reset baseline to defaults"):
            profile["baseline_features"] = {}
            profile["calibration_bias"] = 0.0
            profile["feedback_history"] = []
            st.session_state.profile = profile
            save_profile(profile)
            toast("Profile reset (baseline + bias).", "🧽")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
if st.session_state.page == "Model":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Model")

    st.session_state.model_mode = st.selectbox(
        "Choose a mode",
        ["Mode 1 — Transparent scoring", "Mode 2 — Train-at-runtime classifier"],
        index=["Mode 1 — Transparent scoring", "Mode 2 — Train-at-runtime classifier"].index(st.session_state.model_mode),
    )

    st.markdown(
        """
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
  <span style="background:#ecfeff;color:#0e7490;border:1px solid #22d3ee;border-radius:10px;padding:4px 10px;font-weight:800;">Mode 1: Transparent</span>
  <span style="background:#eef2ff;color:#4338ca;border:1px solid #6366f1;border-radius:10px;padding:4px 10px;font-weight:800;">Mode 2: Runtime classifier</span>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.model_mode.startswith("Mode 1"):
        st.markdown("### Mode 1: White-box scoring model")
        st.latex(r"p_{stress}=\sigma(\mathbf{w}^\top \mathbf{z} + b)")
        st.caption("z = standardized feature deltas (current − baseline) / scale. Missing features are excluded; weights renormalize.")

        # weights table
        w_rows = []
        for k, w in SCORE_WEIGHTS.items():
            w_rows.append({
                "Feature": k,
                "Weight (raw)": w,
                "Sign": "↑ increases p" if w >= 0 else "↓ decreases p",
                "Rationale": SCORE_WEIGHT_RATIONALE.get(k, ""),
                "Used?": "Yes" if k in z1 else "No (missing)",
                "z (std delta)": (z1.get(k, None)),
                "Contribution": (contrib1.get(k, None)),
            })
        w_df = pd.DataFrame(w_rows)
        st.dataframe(w_df, use_container_width=True)

        st.divider()
        st.markdown("### Output")
        c1, c2, c3 = st.columns(3)
        c1.metric("p_stress", f"{p1:.3f}")
        c2.metric("Label", label1)
        c3.metric("Confidence", conf_level)

        st.markdown("### Feedback loop (transparent online adaptation)")
        st.caption("User label (0–10) updates a small bias term b. This is the personalization ‘learning’ mechanism.")
        rating = st.slider("How stressed did you feel? (0–10)", 0, 10, 4)
        if st.button("Submit feedback"):
            db = update_bias_from_feedback(p1, rating, lr=0.12)
            profile["calibration_bias"] = float(profile.get("calibration_bias", 0.0)) + float(db)
            profile.setdefault("feedback_history", []).append({"p": p1, "rating": rating, "delta_bias": db})
            st.session_state.profile = profile
            save_profile(profile)
            toast(f"Bias updated by {db:+.3f} (saved).", "🧪")
            st.rerun()

        with st.expander("Show calibration state"):
            st.write(f"Calibration bias b = **{float(profile.get('calibration_bias',0.0)):.3f}**")
            st.write(f"Feedback count = {len(profile.get('feedback_history', []))}")

    else:
        st.markdown("### Mode 2: Train-at-runtime ML pipeline (illustrative)")
        st.caption("Generates a synthetic training set (no proprietary data), trains a classifier, and shows evaluation + explainability.")
        c1, c2, c3 = st.columns(3)
        with c1:
            N = st.slider("Synthetic training sessions N", 500, 5000, 1200, step=100)
        with c2:
            seed_train = st.number_input("Training seed", value=int(inp.seed), step=1)
        with c3:
            st.write("")
            st.write("")

        if st.button("Generate + Train"):
            with st.spinner("Generating synthetic training set..."):
                df_train = synthesize_training_dataset(int(N), int(seed_train))
            with st.spinner("Training classifier..."):
                artifacts = train_runtime_model(df_train, int(seed_train))
            st.session_state.train_state = {"artifacts": artifacts, "df_train": df_train, "seed": int(seed_train)}
            toast("Training complete. Evaluation updated.", "✅")
            st.rerun()

        if st.session_state.train_state is None:
            st.info("Train to display evaluation and explainability for Mode 2.")
        else:
            art = st.session_state.train_state["artifacts"]
            m = art["metrics"]
            st.markdown("#### Evaluation (train/test split)")
            a, b_, c_ = st.columns(3)
            a.metric("Accuracy", f"{m['accuracy']:.3f}")
            b_.metric("Weighted F1", f"{m['f1']:.3f}")
            c_.metric("Weighted Precision/Recall", f"{m['precision']:.3f} / {m['recall']:.3f}")

            st.markdown("#### Confusion matrix")
            cm = art["cm"]
            labels = art["labels"]
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=False,
                hovertemplate="True %{y}<br>Pred %{x}<br>Count %{z}<extra></extra>",
            ))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Explainability (permutation importance)")
            imp_df = art["importance"].head(10).copy()
            fig2 = go.Figure(go.Bar(x=imp_df["Importance"], y=imp_df["Feature"], orientation="h"))
            fig2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Classification report (details)"):
                st.json(art["report"])

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# EXPLAINABILITY
# ============================================================
if st.session_state.page == "Explainability":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Explainability (legible drivers)")

    st.markdown("### Top drivers (Mode 1 scoring)")
    if drivers_df.empty:
        st.info("No drivers available (all features missing).")
    else:
        fig = go.Figure(go.Bar(
            x=drivers_df["Contribution"],
            y=drivers_df["Driver"],
            orientation="h",
            marker=dict(color="#ef4444"),
            hovertemplate="%{y}<br>Contribution %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    top2 = top_driver_list[:2] if top_driver_list else ["N/A"]
    sentence = f"Stress increased primarily due to **{', '.join(top2)}**; confidence is **{conf_level}** because **{', '.join(conf_reasons)}**."
    st.markdown(f"**Auto-summary:** {sentence}")

    st.divider()
    st.markdown("### You vs baseline deltas (self-relevance)")
    delta_show = []
    for k, v in deltas.items():
        delta_show.append({"Metric": k, "Delta": v})
    if delta_show:
        ddf = pd.DataFrame(delta_show).sort_values("Delta", key=lambda s: s.abs(), ascending=False)
        figd = go.Figure(go.Bar(x=ddf["Delta"], y=ddf["Metric"], orientation="h"))
        figd.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
        st.plotly_chart(figd, use_container_width=True)
    else:
        st.info("Deltas unavailable (missing baseline-sensitive features).")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RECOMMENDATIONS + INTERVENTION
# ============================================================
if st.session_state.page == "Recommendations":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Recommendations (context-gated, safe, actionable)")

    recs = make_recommendations(
        context=inp.context if active_session["meta"]["source"] != "upload" else "Work",
        label=label1,
        confidence=conf_level,
        reasons=conf_reasons,
        deltas=deltas,
        top_drivers=top_driver_list,
    )

    def render_section(title: str, items: List[Dict[str, str]]):
        st.markdown(f"### {title}")
        if not items:
            st.write("—")
            return
        for it in items:
            st.markdown(
                f"""
<div class="panel" style="box-shadow:none;">
  <div style="font-weight:900;font-size:16px;">{it["action"]}</div>
  <div style="color:#475569;margin-top:6px;">{it["rationale"]}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    render_section("1) Do now (1–3 minutes)", recs["do_now"])
    render_section("2) Next 30–60 minutes", recs["next_30_60"])
    render_section("3) Long-term pattern", recs["long_term"])

    st.divider()
    st.markdown("## 60-second paced breathing (progress ring + before/after)")

    before_p = p1
    before_label = label1

    colL, colR = st.columns([1, 1])

    with colL:
        st.markdown("### Start intervention")
        if st.button("Start 60-second paced breathing", use_container_width=True):
            # show ring animation
            ring = st.empty()
            status = st.empty()
            for i in range(61):
                pct = i / 60.0
                phase = "Inhale…" if (i % 10) < 4 else "Exhale…"
                ring.plotly_chart(plot_progress_ring(pct, phase), use_container_width=True)
                status.markdown(f"**{phase}** · {60-i:02d}s remaining")
                time.sleep(0.02)

            # apply deterministic effect to signals and recompute downstream
            post = intervention_effect(session, seed=inp.seed)
            st.session_state.intervention_done = True
            st.session_state.intervention_session = post
            toast("Intervention complete. Signals and outputs updated.", "🫁")
            st.rerun()

    with colR:
        st.markdown("### Before / after (small, plausible change)")
        if st.session_state.intervention_done and st.session_state.intervention_session is not None:
            post_sess = st.session_state.intervention_session
            post_feats = compute_features(post_sess, inp if post_sess["meta"]["source"] != "upload" else None)
            post_p, post_label, _, _ = scoring_model(post_feats, baseline, calibration_bias)

            st.markdown(
                f"""
<div class="panel">
  <div style="display:flex;gap:10px;align-items:center;">
    <span class="badge">Before</span>
    <div style="font-weight:900;">p={before_p:.3f} · {before_label}</div>
  </div>
  <div style="margin-top:10px;display:flex;gap:10px;align-items:center;">
    <span class="badge">After</span>
    <div style="font-weight:900;">p={post_p:.3f} · {post_label}</div>
  </div>
  <div style="margin-top:10px;color:#475569;">
    Applied a modest, seeded improvement to HR/Resp/EDA/EMG to reflect a short downshift in arousal.
    This is illustrative, not a medical claim.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Run the intervention to see the before/after card and updated score.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# COACH CHAT (offline template; optional LLM if OPENAI_API_KEY present)
# ============================================================
def template_coach_reply(user_text: str, context: str, p: float, label: str, conf: str, reasons: List[str], drivers: List[str], asked: int) -> Tuple[str, Optional[str]]:
    """
    Returns (reply, optional_question)
    Must ask at most two clarifying questions total.
    """
    summary = (
        f"Right now I estimate **{label}** load (p={p:.2f}) with **{conf}** confidence. "
        f"Top drivers: {', '.join(drivers[:2]) if drivers else 'N/A'}. "
        f"Confidence reasons: {', '.join(reasons[:2])}."
    )

    # at most two questions
    question = None
    if asked < 2:
        if context == "Driving" and label in ["High", "Moderate"]:
            question = "Are you currently in a safe place to pull over within the next few minutes? (yes/no)"
        elif conf == "Low":
            question = "Which signals do you have available right now (HR/EDA/Resp/EMG), if any?"
        else:
            question = "Do you want a 60-second action plan or a 30–60 minute plan? (short/long)"

    # deterministic plan (3 bullets)
    plan = []
    if context == "Driving":
        plan = [
            "Reduce stimulation (mute audio; simplify decisions).",
            "Plan a safe stop; do paced breathing only when stopped.",
            "If repeated, calibrate baseline before driving and review sleep/caffeine timing.",
        ]
    elif context == "Work":
        plan = [
            "Do a 60–120s break: stand, shoulders down, slow exhale.",
            "Try 60s paced breathing; then reassess.",
            "Triage: choose one next action; postpone low-value tasks for 30 minutes.",
        ]
    else:
        plan = [
            "Hydrate and do a brief posture reset.",
            "Try 60s paced breathing if you feel escalating tension.",
            "If sleep is low, treat today’s signals as higher uncertainty; focus on recovery.",
        ]

    reply = summary + "\n\n**Plan:**\n- " + "\n- ".join(plan)
    return reply, question


def llm_reply_if_available(prompt: str) -> Optional[str]:
    """
    Optional: if OPENAI_API_KEY is set and openai package installed.
    Offline-first: if not available, return None.
    Guardrails: no diagnosis language; no unsafe driving advice; no fear language.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)
    system = (
        "You are a calm, safety-first health coach inside a stress-estimation app. "
        "Do NOT diagnose, do NOT make medical claims. "
        "Avoid fear language. If driving: never suggest unsafe actions; breathing only when stopped. "
        "If confidence is low: say so and ask at most two clarifying questions. "
        "Return a concise response: 2–3 sentence summary + 3 bullets. "
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


if st.session_state.page == "Coach Chat":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Coach Chat (offline-first)")

    context_for_chat = inp.context if active_session["meta"]["source"] != "upload" else "Work"

    chat_state = st.session_state.chat
    hist = chat_state["history"]
    asked = int(chat_state.get("asked", 0))

    if len(hist) == 0:
        intro = (
            f"I can help you act on the current estimate. "
            f"State: {label1} (p={p1:.2f}), confidence: {conf_level}. "
            f"Top drivers: {', '.join(top_driver_list[:2]) if top_driver_list else 'N/A'}."
        )
        hist.append(("Coach", intro))

    for who, msg in hist:
        badge = "badge" if who == "Coach" else "badge badge-amber"
        st.markdown(f"<div class='panel' style='box-shadow:none;'><span class='{badge}'>{who}</span><div style='margin-top:8px;'>{msg}</div></div>", unsafe_allow_html=True)

    user_text = st.text_input("Type a message (offline template unless OPENAI_API_KEY is set)", key="chat_input")
    if st.button("Send", key="chat_send"):
        user_text = (user_text or "").strip()
        if user_text:
            hist.append(("You", user_text))

            # build prompt context
            prompt = (
                f"Context={context_for_chat}\n"
                f"p_stress={p1:.2f}\nlabel={label1}\nconfidence={conf_level}\n"
                f"reasons={conf_reasons}\ndrivers={top_driver_list[:4]}\n"
                f"user_message={user_text}\n"
            )

            # optional LLM; otherwise template
            llm = llm_reply_if_available(prompt)
            if llm is not None:
                hist.append(("Coach", llm))
            else:
                reply, q = template_coach_reply(user_text, context_for_chat, p1, label1, conf_level, conf_reasons, top_driver_list, asked)
                hist.append(("Coach", reply))
                if q is not None and asked < 2:
                    hist.append(("Coach", q))
                    chat_state["asked"] = asked + 1

            st.session_state.chat = chat_state
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# EXPORT
# ============================================================
if st.session_state.page == "Export":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("## Export (optional)")

    summary = {
        "app": APP_TITLE,
        "data_source": active_session["meta"].get("source"),
        "seed": inp.seed if st.session_state.developer_mode else None,
        "inputs": {
            "L_min": inp.L_min,
            "context": inp.context,
            "phase_segments": inp.segments,
            "latent_stress_input": inp.latent_stress_0_100,
            "confounders_toggles": {
                "caffeine": inp.conf_caffeine,
                "poor_sleep": inp.conf_poor_sleep,
                "exercise": inp.conf_exercise,
            },
            "quality_mode": inp.quality_mode,
            "missing_toggles": {
                "hr": inp.missing_hr, "eda": inp.missing_eda, "emg": inp.missing_emg, "resp": inp.missing_resp
            },
        },
        "features": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)) for k, v in feats.items()},
        "baseline": baseline,
        "deltas": deltas,
        "mode1": {
            "p_stress": p1,
            "label": label1,
            "confidence": conf_level,
            "confidence_reasons": conf_reasons,
            "top_drivers": top_driver_list[:5],
        },
        "intervention_applied": bool(st.session_state.intervention_done),
    }

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "Download session summary (JSON)",
            data=json.dumps(summary, indent=2).encode("utf-8"),
            file_name="physiostate_summary.json",
            mime="application/json",
        )
    with col2:
        df_export = active_session["canonical"][CANON_COLS].copy()
        st.download_button(
            "Download canonical data (CSV)",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name="physiostate_canonical.csv",
            mime="text/csv",
        )

    st.caption("Canonical schema: t_sec (required), hr_bpm/eda/emg/resp_bpm/phase (optional). Missing channels are allowed.")
    st.markdown("</div>", unsafe_allow_html=True)

