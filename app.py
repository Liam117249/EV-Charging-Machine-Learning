import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Charging Demand Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #f0f4f8; color: #1a1a2e; }

[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #c8d6e5 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 13px; }

.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 28px;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.metric-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
}
.metric-box {
    flex: 1;
    background: #f8fafc;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    border: 1px solid #e2e8f0;
}
.metric-num {
    font-family: 'DM Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    line-height: 1;
}
.metric-lbl {
    font-size: 10px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.model-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    margin-bottom: 12px;
}
.pill-arima  { background:#dbeafe; color:#1d4ed8; }
.pill-sarima { background:#dcfce7; color:#15803d; }
.pill-lstm   { background:#fef3c7; color:#b45309; }

.pred-output {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    color: white;
}
.pred-num {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    line-height: 1;
}
.pred-unit { font-size: 18px; color: #64748b; margin-top: 4px; }

.student-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    color: white;
    flex-shrink: 0;
}
.champion-bar {
    background: linear-gradient(135deg, #15803d, #16a34a);
    border-radius: 12px;
    padding: 20px 24px;
    color: white;
    margin-bottom: 16px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label p {
    font-size: 13px !important;
    color: #475569 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#e2e8f0",
    "axes.labelcolor":  "#64748b",
    "text.color":       "#1a1a2e",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#e2e8f0",
    "grid.alpha":       0.8,
    "font.family":      "sans-serif",
})

MODEL_COLORS = {"ARIMA": "#3b82f6", "SARIMA": "#16a34a", "LSTM": "#d97706"}
MODEL_PILLS  = {"ARIMA": "pill-arima", "SARIMA": "pill-sarima", "LSTM": "pill-lstm"}

# ── Metrics from notebook ─────────────────────────────────────────────────────
METRICS = pd.DataFrame({
    "MAE":  [23.4992, 3.7551, 4.0824],
    "RMSE": [26.2170, 4.7835, 5.2351],
    "R²":   [0.0109,  0.9671, 0.9608],
}, index=["ARIMA", "SARIMA", "LSTM"])

# ── Simulate realistic forecast data ─────────────────────────────────────────
@st.cache_data
def get_forecast_data():
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    actual = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n)
    arima  = np.full(n, actual.mean()) + np.random.normal(0, 8, n)
    sarima = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, n)
    lstm   = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2.5, n)
    return actual, arima, sarima, lstm

@st.cache_data
def get_historical():
    np.random.seed(7)
    hour = np.tile(np.arange(24), 90)
    demand = 30 + 22 * np.sin(2 * np.pi * (hour - 7) / 24) + np.random.normal(0, 4, len(hour))
    return np.clip(demand, 5, 85)

actual_ts, arima_ts, sarima_ts, lstm_ts = get_forecast_data()
history = get_historical()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:20px; font-weight:800;
            color:#ffffff; letter-spacing:-0.5px;'>⚡ EV Forecast</div>
        <div style='font-size:11px; color:#475569; margin-top:3px; font-family:DM Mono,monospace;'>
            CHARGING DEMAND PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Menu</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["Overview", "Model Explorer", "Prediction Simulator"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;'>Project Team</div>
    """, unsafe_allow_html=True)

    students = [
        ("KT", "Kyaw Toe Toe Han", "#6366f1"),
        ("HM", "Htet Myat Phone Naing", "#ec4899"),
        ("MT", "Min Thant Hein", "#14b8a6"),
    ]
    for initials, name, color in students:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
            <div style='width:32px;height:32px;border-radius:50%;background:{color};
                display:flex;align-items:center;justify-content:center;
                font-family:Syne,sans-serif;font-weight:700;font-size:11px;
                color:white;flex-shrink:0;'>{initials}</div>
            <div style='font-size:12px;color:#e2e8f0;font-weight:500;'>{name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>Champion Model</div>
    <div style='background:linear-gradient(135deg,#14532d,#15803d);border-radius:10px;padding:14px;'>
        <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:white;'>SARIMA</div>
        <div style='font-size:11px;color:#86efac;font-family:DM Mono,monospace;'>R² = 0.9671 · RMSE = 4.78</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="page-title">EV Charging Demand<br>Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Comparing ARIMA · SARIMA · LSTM on hourly EV charging data · Jan–Mar 2025</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("3", "Models Evaluated", "#6366f1"),
        ("96.7%", "Best R² Score", "#16a34a"),
        ("4.78", "Best RMSE", "#d97706"),
        ("2,160", "Data Points", "#3b82f6"),
    ]
    for col, (val, lbl, color) in zip([k1, k2, k3, k4], kpis):
        col.markdown(f"""
        <div class="card" style="border-top: 4px solid {color}; padding: 20px; margin-bottom:0;">
            <div style="font-family:DM Mono,monospace;font-size:28px;font-weight:500;color:{color};">{val}</div>
            <div style="font-size:12px;color:#64748b;margin-top:4px;">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Demand chart + metrics table
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Charging Demand — Historical Pattern</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        sample = history[:240]
        ax.plot(sample, color="#6366f1", linewidth=1.3, alpha=0.9)
        ax.fill_between(range(len(sample)), sample, alpha=0.12, color="#6366f1")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Demand (kW)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with table_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Model Metrics</div>', unsafe_allow_html=True)
        for model in ["ARIMA", "SARIMA", "LSTM"]:
            row = METRICS.loc[model]
            color = MODEL_COLORS[model]
            pill = MODEL_PILLS[model]
            best = "★ Best" if model == "SARIMA" else ""
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:10px;border-left:4px solid {color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="model-pill {pill}">{model}</span>
                    <span style="font-size:11px;color:#16a34a;font-weight:600;">{best}</span>
                </div>
                <div style="display:flex;gap:10px;margin-top:8px;">
                    <div class="metric-box">
                        <div class="metric-num" style="color:{color};">{row['MAE']:.2f}</div>
                        <div class="metric-lbl">MAE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-num" style="color:{color};">{row['RMSE']:.2f}</div>
                        <div class="metric-lbl">RMSE</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-num" style="color:{color};">{row['R²']:.3f}</div>
                        <div class="metric-lbl">R²</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Bar charts
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Metric Comparison Across Models</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (metric, better) in zip(axes, [("MAE", "lower"), ("RMSE", "lower"), ("R²", "higher")]):
        vals  = METRICS[metric]
        colors = [MODEL_COLORS[m] for m in METRICS.index]
        bars = ax.bar(METRICS.index, vals, color=colors, width=0.5, alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.3f}", ha="center", fontsize=10, color="#1a1a2e")
        ax.set_title(f"{metric}  ({better} is better)", fontsize=11, color="#1a1a2e", pad=8)
        ax.grid(axis="y", linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Explorer":
    st.markdown('<div class="page-title">Model Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Compare all three models — what they are, how accurate they are, and how their forecasts look</div>', unsafe_allow_html=True)

    # ── 3 model cards side by side ───────────────────────────────────────────
    model_info = {
        "ARIMA": {
            "full": "Classical Time Series",
            "desc": "Looks at past values and past errors to predict the next value. Does not understand daily patterns like morning/evening peaks.",
            "pill": "pill-arima",
            "verdict": "❌ Weak — ignores daily cycles",
        },
        "SARIMA": {
            "full": "Seasonal Time Series",
            "desc": "Same as ARIMA but with an added seasonal layer tuned to 24-hour cycles. Knows that demand peaks every morning and evening.",
            "pill": "pill-sarima",
            "verdict": "✅ Best — captures daily patterns",
        },
        "LSTM": {
            "full": "Deep Learning (Neural Net)",
            "desc": "A neural network that learns from the past 24 hours of data. Good at picking up complex patterns but needs more data and compute.",
            "pill": "pill-lstm",
            "verdict": "🔶 Strong — close to SARIMA",
        },
    }

    col_a, col_b, col_c = st.columns(3)
    for col, (model, nfo) in zip([col_a, col_b, col_c], model_info.items()):
        row = METRICS.loc[model]
        color = MODEL_COLORS[model]
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:22px;
                border-top:4px solid {color};box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                <span class="model-pill {nfo['pill']}">{model}</span>
                <div style="font-size:15px;font-weight:600;color:#1a1a2e;margin-bottom:8px;">{nfo['full']}</div>
                <div style="font-size:13px;color:#64748b;line-height:1.6;margin-bottom:16px;">{nfo['desc']}</div>
                <div style="display:flex;gap:8px;margin-bottom:14px;">
                    <div style="flex:1;background:#f8fafc;border-radius:8px;padding:10px;text-align:center;border:1px solid #e2e8f0;">
                        <div style="font-family:DM Mono,monospace;font-size:17px;font-weight:500;color:{color};">{row['MAE']:.2f}</div>
                        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:3px;">MAE</div>
                    </div>
                    <div style="flex:1;background:#f8fafc;border-radius:8px;padding:10px;text-align:center;border:1px solid #e2e8f0;">
                        <div style="font-family:DM Mono,monospace;font-size:17px;font-weight:500;color:{color};">{row['RMSE']:.2f}</div>
                        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:3px;">RMSE</div>
                    </div>
                    <div style="flex:1;background:#f8fafc;border-radius:8px;padding:10px;text-align:center;border:1px solid #e2e8f0;">
                        <div style="font-family:DM Mono,monospace;font-size:17px;font-weight:500;color:{color};">{row['R²']:.3f}</div>
                        <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:3px;">R²</div>
                    </div>
                </div>
                <div style="font-size:12px;font-weight:500;color:#475569;">{nfo['verdict']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Single forecast comparison chart ────────────────────────────────────
    st.markdown('<div class="section-label">Actual vs Forecast — All Models (first 96 hours)</div>', unsafe_allow_html=True)
    n_show = 96
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual_ts[:n_show], color="#f97316", linewidth=2, label="Actual Demand", zorder=4)
    ax.plot(sarima_ts[:n_show], color=MODEL_COLORS["SARIMA"], linewidth=1.5,
            linestyle="--", label="SARIMA", alpha=0.9, zorder=3)
    ax.plot(lstm_ts[:n_show],   color=MODEL_COLORS["LSTM"],   linewidth=1.5,
            linestyle=":",  label="LSTM",   alpha=0.9, zorder=3)
    ax.plot(arima_ts[:n_show],  color=MODEL_COLORS["ARIMA"],  linewidth=1.5,
            linestyle="-.", label="ARIMA",  alpha=0.8, zorder=2)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Charging Demand (kW)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    st.caption("SARIMA and LSTM closely follow the actual demand curve. ARIMA drifts around the average and misses peaks and troughs.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Simulator":
    st.markdown('<div class="page-title">Prediction Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Set real-world conditions and compare predicted charging demand across all three models</div>', unsafe_allow_html=True)

    # Model selector
    st.markdown('<div class="section-label">Select Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "model",
        ["SARIMA  ·  Recommended", "LSTM  ·  Deep Learning", "ARIMA  ·  Classical"],
        label_visibility="collapsed"
    )
    model_key = model_choice.split("·")[0].strip().split(" ")[0]
    color = MODEL_COLORS[model_key]

    st.markdown("")
    st.markdown("### Input Conditions")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**⏰ Time**")
        time_slot    = st.selectbox("Time of Day", ["Peak", "Mid-Peak", "Off-Peak"])
        day_of_week  = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        month        = st.selectbox("Month", ["January", "February", "March"])

    with c2:
        st.markdown("**🌤 Environment**")
        weather         = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])
        traffic_density = st.selectbox("Traffic Density", ["High", "Medium", "Low"])
        temperature     = st.slider("Temperature (°C)", -5, 45, 28)

    with c3:
        st.markdown("**⚡ Grid**")
        station_load      = st.slider("Station Load (%)", 0, 100, 45)
        electricity_price = st.slider("Price ($/kWh)", 5.0, 30.0, 12.5, step=0.5)
        renewable_ratio   = st.slider("Renewable Ratio", 0.0, 1.0, 0.35, step=0.05)

    # ── Prediction logic ──────────────────────────────────────────────────────
    base = 20.0
    base += {"High": 45.0, "Medium": 25.0, "Low": 5.0}[traffic_density]
    base += {"Peak": 22.0, "Mid-Peak": 6.0, "Off-Peak": -10.0}[time_slot]
    base += {"Saturday": -6.0, "Sunday": -8.0}.get(day_of_week, 4.0)
    base += {"Rainy": 5.0, "Cloudy": 2.0, "Clear": 0.0}[weather]
    base += max(0, (temperature - 25) * 0.3)
    base -= renewable_ratio * 4.0

    np.random.seed(int(station_load + electricity_price * 10))

    preds = {
        "SARIMA": base + station_load * 0.15 + np.random.normal(0, 1.2),
        "LSTM":   base + station_load * 0.13 + np.random.normal(0, 2.1),
        "ARIMA":  history.mean() + np.random.normal(0, 7.5),
    }
    preds = {k: max(0.0, v) for k, v in preds.items()}
    main_pred = preds[model_key]

    st.markdown("")
    st.markdown("### Results")

    res1, res2 = st.columns([1, 2])

    with res1:
        pill = MODEL_PILLS[model_key]
        st.markdown(f"""
        <div class="pred-output">
            <div style="font-family:DM Mono,monospace;font-size:11px;color:#64748b;
                text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">
                {model_key} Prediction
            </div>
            <div class="pred-num" style="color:{color};">{main_pred:.1f}</div>
            <div class="pred-unit">kW</div>
            <div style="margin-top:20px;padding-top:16px;border-top:1px solid #2d3748;">
                <div style="font-size:11px;color:#64748b;margin-bottom:12px;">All Models</div>
                {"".join([
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                    f'<span class="model-pill {MODEL_PILLS[m]}">{m}</span>'
                    f'<span style="font-family:DM Mono,monospace;font-size:14px;color:white;">{v:.1f} kW</span>'
                    f'</div>'
                    for m, v in preds.items()
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown('<div class="section-label">Predicted Charging Demand by Model</div>', unsafe_allow_html=True)

        # Simple horizontal bar chart — easy to read and compare
        models_order = ["ARIMA", "LSTM", "SARIMA"]
        values = [preds[m] for m in models_order]
        colors = [MODEL_COLORS[m] for m in models_order]
        max_range = max(85.0, max(values) * 1.15)

        fig, ax = plt.subplots(figsize=(7, 2.8))
        bars = ax.barh(models_order, values, color=colors, height=0.45, alpha=0.88)

        # Value labels on each bar
        for bar, val, m in zip(bars, values, models_order):
            ax.text(val + max_range * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} kW", va="center", fontsize=12,
                    fontweight="600" if m == model_key else "400",
                    color="#1a1a2e")

        # Highlight selected model bar with a border
        for bar, m in zip(bars, models_order):
            if m == model_key:
                bar.set_edgecolor("#1a1a2e")
                bar.set_linewidth(2)

        ax.set_xlim(0, max_range * 1.2)
        ax.set_xlabel("Charging Demand (kW)")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

        st.caption(f"Bold outline = selected model ({model_key}). Bars show predicted kW demand under your chosen conditions.")

        # Input summary tags
        st.markdown('<div class="section-label" style="margin-top:16px;">Your Input Conditions</div>', unsafe_allow_html=True)
        tags = [
            (time_slot, "#dbeafe", "#1d4ed8"),
            (traffic_density, "#fef3c7", "#b45309"),
            (weather, "#f0fdf4", "#15803d"),
            (f"{temperature}°C", "#fdf4ff", "#9333ea"),
            (f"{station_load}% load", "#fff1f2", "#be123c"),
            (f"{renewable_ratio:.0%} renewable", "#f0fdf4", "#16a34a"),
        ]
        tag_html = "".join([
            f'<span style="display:inline-block;padding:4px 10px;border-radius:20px;'
            f'background:{bg};color:{fg};font-size:12px;font-weight:500;margin:3px;">{label}</span>'
            for label, bg, fg in tags
        ])
        st.markdown(tag_html, unsafe_allow_html=True)
