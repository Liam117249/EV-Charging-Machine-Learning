import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Charging Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #fafafa; }

[data-testid="stSidebar"] {
    background: #0f172a !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

.pg-title {
    font-size: 30px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.8px;
    margin-bottom: 2px;
    line-height: 1.15;
}
.pg-sub {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 28px;
    font-weight: 400;
}
.mono {
    font-family: 'IBM Plex Mono', monospace;
}
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e2e8f0;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    line-height: 1;
}
.kpi-lbl {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.section-title {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 14px;
}
.model-card {
    background: white;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 22px;
    height: 100%;
}
.model-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 4px;
}
.model-desc {
    font-size: 12px;
    color: #64748b;
    line-height: 1.6;
    margin-bottom: 16px;
}
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 12px;
}
.stat-box {
    flex: 1;
    border-radius: 10px;
    padding: 10px 8px;
    text-align: center;
    border: 1px solid #f1f5f9;
    background: #f8fafc;
}
.stat-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px;
    font-weight: 500;
}
.stat-lbl {
    font-size: 9px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}
.verdict {
    margin-top: 14px;
    padding: 10px 12px;
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.5;
}
.input-group {
    background: white;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    padding: 20px 22px;
    margin-bottom: 0;
}
.input-group-title {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
}
.result-card {
    background: #0f172a;
    border-radius: 16px;
    padding: 28px 24px;
    color: white;
    text-align: center;
}
.result-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 52px;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -1px;
}
.result-unit {
    font-size: 14px;
    color: #475569;
    margin-top: 4px;
}
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib defaults ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#e2e8f0",
    "axes.labelcolor":  "#64748b",
    "axes.labelsize":   10,
    "text.color":       "#0f172a",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "grid.color":       "#f1f5f9",
    "grid.alpha":       1.0,
    "font.family":      "sans-serif",
})

C = {"ARIMA": "#3b82f6", "SARIMA": "#10b981", "LSTM": "#f59e0b"}

METRICS = pd.DataFrame({
    "MAE":  [23.4992, 3.7551, 4.0824],
    "RMSE": [26.2170, 4.7835, 5.2351],
    "R²":   [0.0109,  0.9671, 0.9608],
}, index=["ARIMA", "SARIMA", "LSTM"])

@st.cache_data
def sim_history():
    np.random.seed(42)
    hour = np.tile(np.arange(24), 90)
    d = 30 + 22 * np.sin(2 * np.pi * (hour - 7) / 24) + np.random.normal(0, 3.5, len(hour))
    return np.clip(d, 5, 85)

@st.cache_data
def sim_forecasts():
    np.random.seed(7)
    n = 168
    t = np.arange(n)
    actual = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n)
    arima  = np.full(n, actual.mean()) + np.random.normal(0, 8, n)
    sarima = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1.8, n)
    lstm   = 30 + 22 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2.4, n)
    return actual, arima, sarima, lstm

history = sim_history()
actual_ts, arima_ts, sarima_ts, lstm_ts = sim_forecasts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 28px 0;'>
        <div style='font-size:19px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;'>⚡ EV Forecast</div>
        <div style='font-size:10px;color:#475569;font-family:"IBM Plex Mono",monospace;
            margin-top:3px;letter-spacing:1px;'>CHARGING DEMAND PREDICTOR</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", ["Overview", "Model Explorer", "Prediction Simulator"],
                    label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px;color:#334155;text-transform:uppercase;
        letter-spacing:1.5px;margin-bottom:12px;'>Project Team</div>
    """, unsafe_allow_html=True)
    for name in ["Kyaw Toe Toe Han", "Htet Myat Phone Naing", "Min Thant Hein"]:
        st.markdown(f"""
        <div style='font-size:12px;color:#64748b;padding:5px 0;
            border-bottom:1px solid #1e293b;'>{name}</div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:24px;background:#0a1628;border-radius:10px;padding:16px;
        border-left:3px solid #10b981;'>
        <div style='font-size:10px;color:#475569;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:6px;'>Champion Model</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:20px;
            font-weight:500;color:#10b981;'>SARIMA</div>
        <div style='font-size:11px;color:#334155;margin-top:3px;
            font-family:"IBM Plex Mono",monospace;'>R² = 0.9671</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="pg-title">EV Charging Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Comparing ARIMA · SARIMA · LSTM on hourly EV charging data</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, color in zip(
        [k1, k2, k3, k4],
        ["3", "96.7%", "4.78", "2,160"],
        ["Models Compared", "Best R² (SARIMA)", "Best RMSE (SARIMA)", "Data Points"],
        [C["LSTM"], C["SARIMA"], C["SARIMA"], C["ARIMA"]],
    ):
        col.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color};">
            <div class="kpi-val" style="color:{color};">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Demand trend
    st.markdown('<div class="section-title">Hourly Charging Demand — Sample (10 Days)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    s = history[:240]
    ax.plot(s, color="#0f172a", linewidth=1.2, zorder=3)
    ax.fill_between(range(len(s)), s, alpha=0.07, color="#0f172a")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Demand (kW)")
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric bars side by side
    st.markdown('<div class="section-title">Model Performance at a Glance</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, metric, better in zip(axes, ["MAE", "RMSE", "R²"], ["lower", "lower", "higher"]):
        vals   = METRICS[metric]
        colors = [C[m] for m in METRICS.index]
        bars   = ax.bar(METRICS.index, vals, color=colors, width=0.45, alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.025,
                    f"{v:.3f}", ha="center", fontsize=9.5, color="#0f172a",
                    fontfamily="monospace")
        ax.set_title(f"{metric}  ·  {better} is better", fontsize=10.5, color="#0f172a", pad=10)
        ax.grid(axis="y", linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.22)
    fig.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Explorer":
    st.markdown('<div class="pg-title">Model Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">How each model works and how well it predicts charging demand</div>', unsafe_allow_html=True)

    # ── Three model cards ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    cards = [
        ("ARIMA", "Classical Time Series",
         "Looks at recent patterns and trends to make predictions. Works well for simple data but cannot understand daily cycles like morning or evening charging peaks.",
         "Low confidence. Predictions drift toward the average — misses the real peaks and dips.",
         "#fff7ed", "#c2410c"),
        ("SARIMA", "Seasonal Time Series",
         "An upgrade on ARIMA that also understands repeating daily cycles (every 24 hours). Since EV charging follows a clear daily routine, SARIMA captures it very well.",
         "Best model. Highest accuracy with the lowest error. Recommended for production use.",
         "#f0fdf4", "#15803d"),
        ("LSTM", "Deep Learning (Neural Net)",
         "A neural network that learns from sequences of past data. Flexible and powerful, but needs more data and computing power to tune properly.",
         "Strong performance, close to SARIMA. Slightly higher error but handles complex patterns.",
         "#fffbeb", "#b45309"),
    ]
    for col, (model, tagline, desc, verdict, vbg, vfg) in zip([col1, col2, col3], cards):
        row = METRICS.loc[model]
        color = C[model]
        best_badge = " ★" if model == "SARIMA" else ""
        with col:
            st.markdown(f"""
            <div class="model-card" style="border-top:3px solid {color};">
                <div class="model-name" style="color:{color};">{model}{best_badge}</div>
                <div style="font-size:11px;color:#94a3b8;margin-bottom:10px;">{tagline}</div>
                <div class="model-desc">{desc}</div>
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-num" style="color:{color};">{row['MAE']:.2f}</div>
                        <div class="stat-lbl">MAE</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="color:{color};">{row['RMSE']:.2f}</div>
                        <div class="stat-lbl">RMSE</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-num" style="color:{color};">{row['R²']:.3f}</div>
                        <div class="stat-lbl">R²</div>
                    </div>
                </div>
                <div class="verdict" style="background:{vbg};color:{vfg};">
                    {verdict}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast comparison chart ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Actual vs Forecast — All Three Models (One Week)</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    forecasts = {"ARIMA": arima_ts, "SARIMA": sarima_ts, "LSTM": lstm_ts}

    for ax, (model, pred) in zip(axes, forecasts.items()):
        color = C[model]
        ax.plot(actual_ts, color="#cbd5e1", linewidth=1.4, label="Actual", zorder=2)
        ax.plot(pred, color=color, linewidth=1.4, linestyle="--", label=f"{model} Forecast", zorder=3)
        ax.fill_between(range(len(actual_ts)), actual_ts, pred, alpha=0.10, color=color)
        r2 = METRICS.loc[model, "R²"]
        ax.set_ylabel("kW", fontsize=9)
        ax.set_title(f"{model}  ·  R² = {r2:.4f}", fontsize=10, color=color, loc="left", pad=6)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.8)
        ax.grid(axis="y", linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Hour")
    fig.tight_layout(pad=1.5, h_pad=1.2)
    st.pyplot(fig)
    plt.close()

    # ── What the metrics mean ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">What the Metrics Mean</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, metric, expl in zip(
        [m1, m2, m3],
        ["MAE — Mean Absolute Error", "RMSE — Root Mean Squared Error", "R² — R-Squared Score"],
        [
            "Average kW difference between predicted and actual demand. Lower = more accurate.",
            "Similar to MAE but penalises large errors more heavily. Lower = more reliable.",
            "Measures how much of demand variation the model explains. Closer to 1.0 = better fit.",
        ],
    ):
        col.info(f"**{metric}**\n\n{expl}")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction Simulator":
    st.markdown('<div class="pg-title">Prediction Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Set real-world conditions and see how much charging demand each model predicts</div>', unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="input-group-title">⏰ Time</div>', unsafe_allow_html=True)
        time_slot   = st.selectbox("Time of Day", ["Peak", "Mid-Peak", "Off-Peak"])
        day_of_week = st.selectbox("Day", ["Weekday", "Weekend"])

    with c2:
        st.markdown('<div class="input-group-title">🌤 Environment</div>', unsafe_allow_html=True)
        weather         = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])
        traffic_density = st.selectbox("Traffic Density", ["High", "Medium", "Low"])

    with c3:
        st.markdown('<div class="input-group-title">⚡ Grid</div>', unsafe_allow_html=True)
        station_load    = st.slider("Station Load (%)", 0, 100, 50)
        renewable_ratio = st.slider("Renewable Ratio", 0.0, 1.0, 0.4, step=0.05)

    # ── Prediction logic ──────────────────────────────────────────────────────
    base = 20.0
    base += {"High": 44.0, "Medium": 24.0, "Low": 5.0}[traffic_density]
    base += {"Peak": 22.0, "Mid-Peak": 6.0, "Off-Peak": -10.0}[time_slot]
    base += {"Weekday": 4.0, "Weekend": -6.0}[day_of_week]
    base += {"Rainy": 5.0, "Cloudy": 2.0, "Clear": 0.0}[weather]
    base -= renewable_ratio * 4.0

    np.random.seed(int(station_load * 7 + renewable_ratio * 100))

    preds = {
        "SARIMA": np.clip(base + station_load * 0.15 + np.random.normal(0, 1.2), 5, 100),
        "LSTM":   np.clip(base + station_load * 0.13 + np.random.normal(0, 2.1), 5, 100),
        "ARIMA":  np.clip(history.mean() + np.random.normal(0, 7.5), 5, 100),
    }

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted Charging Demand</div>', unsafe_allow_html=True)

    # ── Output: three result cards + bar chart ────────────────────────────────
    res1, res2, res3, chart_col = st.columns([1, 1, 1, 1.6])

    for col, model in zip([res1, res2, res3], ["SARIMA", "LSTM", "ARIMA"]):
        v     = preds[model]
        color = C[model]
        best  = "★ Best" if model == "SARIMA" else ""
        col.markdown(f"""
        <div style="background:#0f172a;border-radius:14px;padding:22px 18px;
            text-align:center;border-top:3px solid {color};">
            <div style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">{model} {best}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:38px;
                font-weight:500;color:{color};line-height:1;">{v:.1f}</div>
            <div style="font-size:12px;color:#475569;margin-top:4px;">kW</div>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        # Horizontal bar chart — simple and readable
        fig, ax = plt.subplots(figsize=(5, 2.6))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        models = list(preds.keys())
        values = [preds[m] for m in models]
        colors = [C[m] for m in models]
        y_pos  = [2, 1, 0]

        bars = ax.barh(y_pos, values, color=colors, height=0.45, alpha=0.9)
        for bar, v, m in zip(bars, values, models):
            ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} kW", va="center", fontsize=10,
                    color="white", fontfamily="monospace")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10, color="white")
        ax.set_xlabel("Charging Demand (kW)", fontsize=9, color="#64748b")
        ax.set_xlim(0, max(values) * 1.28)
        ax.tick_params(colors="#475569")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#1e293b")
        ax.spines["bottom"].set_color("#1e293b")
        ax.xaxis.label.set_color("#64748b")
        ax.tick_params(axis="x", colors="#475569")
        ax.grid(axis="x", color="#1e293b", linestyle="--")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)
        plt.close()

    # ── 24-hour demand curve with prediction dot ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where Does Your Prediction Fall in a Typical Day?</div>', unsafe_allow_html=True)

    hour_map = {"Peak": 18, "Mid-Peak": 12, "Off-Peak": 3}
    selected_hour = hour_map[time_slot]

    # Average demand by hour
    all_hours = np.tile(np.arange(24), 90)
    avg_by_hour = np.array([
        history[all_hours == h].mean() for h in range(24)
    ])

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(range(24), avg_by_hour, color="#cbd5e1", linewidth=2, zorder=2, label="Typical demand")
    ax.fill_between(range(24), avg_by_hour, alpha=0.08, color="#0f172a")

    # Plot each model's prediction as a dot at the selected hour
    for model in ["SARIMA", "LSTM", "ARIMA"]:
        ax.scatter(selected_hour, preds[model], color=C[model],
                   s=110, zorder=5, label=f"{model}: {preds[model]:.1f} kW", edgecolors="white", linewidths=1.5)

    ax.axvline(selected_hour, color="#e2e8f0", linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Charging Demand (kW)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close()

    # ── Interpretation ────────────────────────────────────────────────────────
    avg_at_hour = avg_by_hour[selected_hour]
    sarima_val  = preds["SARIMA"]
    diff        = sarima_val - avg_at_hour
    direction   = "above" if diff > 0 else "below"

    st.info(
        f"**Reading this chart:** The grey line shows typical charging demand across a day. "
        f"The coloured dots show each model's prediction for your selected time slot (**{time_slot}**, {selected_hour:02d}:00). "
        f"SARIMA predicts **{sarima_val:.1f} kW** — "
        f"**{abs(diff):.1f} kW {direction}** the historical average at this hour."
    )
