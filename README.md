# PhysioState  
### Wellness-Oriented Stress Load Estimation from Wearable Sensor Data

PhysioState is an interactive **Streamlit-based wellness application** that demonstrates how multi-sensor wearable data can be transformed into an interpretable **stress load estimate** using a transparent machine-learning pipeline.

The app is designed as a **research and educational prototype**, emphasizing clarity, interpretability, and system-level understanding rather than clinical diagnosis or treatment.

> ⚠️ **Disclaimer**  
> This application provides **wellness-level insights only**.  
> It is **not a medical device**, does not diagnose stress-related conditions, and does not provide medical advice.

---

## 1. Motivation

Wearable devices can measure physiological signals related to autonomic activation—such as heart rate, electrodermal activity, respiration, and motion—but raw sensor data are difficult to interpret without a structured processing pipeline.

Most existing tools either:
- focus on algorithmic performance without transparency, or  
- visualize signals without explaining how predictions are generated.

PhysioState addresses this gap by providing a **clear, end-to-end workflow** that shows:
1. how raw wearable signals are processed,
2. how features are extracted,
3. how a baseline machine-learning model estimates stress load, and
4. how results and limitations are communicated to users.

---

## 2. Scope and Design Philosophy

- **Wellness-focused**: estimates physiological *stress load*, not clinical stress.
- **Context-aware**: incorporates activity or driving phase to reduce false positives.
- **Interpretable by design**: simple models with feature-level explanations.
- **Demo-ready**: works with real CSV data or internally generated synthetic sessions.

Driving scenarios are used as a **default structured example** (rest → city → highway), but the same pipeline generalizes to other daily activities.

---

## 3. What the App Does

### Inputs
- Multi-sensor time-series data (real or synthetic)
- Optional context markers (e.g., driving phase or activity)
- Optional session metadata (sleep, caffeine, hydration, mood)

### Processing
- Signal preprocessing and imputation
- Window-based feature extraction
- Baseline machine-learning classification
- Reliability and data-quality checks

### Outputs
- Stress load score and state (Low / Medium / High)
- Confidence indicator based on signal quality
- Feature-driven explanations
- Wellness-safe recommendations

---

## 4. Data Format

### Supported CSV Columns (any subset is allowed)

**Time**
- `t` (seconds), or  
- `time` (seconds), or  
- `timestamp` (any parseable datetime)

**Physiological signals**
- `hr` – Heart rate (bpm)
- `eda` – Electrodermal activity (µS)
- `resp` – Respiration rate (breaths/min)
- `emg` – Muscle activity (a.u.)
- `acc` – Acceleration / activity proxy (a.u.)
- `skin_temp` – Skin temperature (°C)

**Context**
- `context` – categorical marker (e.g., rest / city / highway)

**Label (optional)**
- `label` – binary indicator (0/1) for high stress load

Missing channels are handled through imputation, with reduced reliability reflected in the output.

### Minimal example
```csv
t,hr,eda,resp,acc,context,label
0,72,1.1,14,0.02,rest,0
1,73,1.2,14,0.03,rest,0
2,90,2.0,18,0.15,city,1
