import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras

WINDOW = 24

st.set_page_config(page_title="Smart Meter Anomaly Detection", layout="wide")
st.title("Smart Meter Anomaly Detection")
st.write(
    "An LSTM Autoencoder learns normal electricity usage. "
    "Hours it cannot reconstruct well are flagged as anomalies (red dots)."
)


@st.cache_resource
def load_assets():
    model = keras.models.load_model("data/lstm_ae.keras")
    scaler = joblib.load("data/scaler.pkl")
    threshold = float(np.load("data/threshold.npy"))
    return model, scaler, threshold


@st.cache_data
def load_test_data():
    d = np.load("data/processed.npz")
    err = np.load("data/test_err.npy")
    return d["test_raw"], d["test_index"], err


def make_windows(data, window):
    return np.array([data[i : i + window] for i in range(len(data) - window + 1)])


def draw(x, y, err, thr):
    flags = err > thr
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, color="steelblue", linewidth=1, label="Consumption (kW)")
    ax.scatter(x[flags], y[flags], color="red", s=25, label="Anomaly", zorder=3)
    ax.set_ylabel("kW")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, flags


model, scaler, base_threshold = load_assets()

# Sidebar controls
mode = st.sidebar.radio("Data source", ["Test data (from dataset)", "Upload your own CSV"])
factor = st.sidebar.slider("Threshold multiplier (higher = fewer anomalies)", 0.5, 2.0, 1.0, 0.1)
thr = base_threshold * factor
st.sidebar.write(f"Current threshold: {thr:.4f}")

if mode == "Test data (from dataset)":
    test_raw, test_index, test_err = load_test_data()
    y_all = test_raw[WINDOW - 1 :]
    n = len(test_err)

    days = st.slider("Days to show", 7, 90, 30)
    start_day = st.slider("Start from day number", 0, max(0, n // 24 - days), 0)
    show = days * 24
    start = start_day * 24
    sl = slice(start, start + show)
    st.caption(
        f"Showing from {pd.to_datetime(test_index[start]).date()} "
        f"to {pd.to_datetime(test_index[start + show - 1]).date()}"
    )

    fig, flags = draw(test_index[sl], y_all[sl], test_err[sl], thr)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Hours shown", show)
    c2.metric("Anomalies found", int(flags.sum()))
    c3.metric("Anomaly rate", f"{100 * flags.mean():.1f}%")

    if flags.any():
        table = pd.DataFrame(
            {
                "time": test_index[sl][flags],
                "consumption_kw": y_all[sl][flags],
                "error": test_err[sl][flags],
            }
        ).sort_values("error", ascending=False)
        st.subheader("Most suspicious hours")
        st.dataframe(table.head(20), use_container_width=True)

else:
    st.write(
        "Upload a CSV with one column of **hourly consumption in kW** "
        f"(at least {WINDOW + 1} rows)."
    )

    # Demo file with a few injected spikes
    test_raw, _, _ = load_test_data()
    sample = test_raw[:500].copy()
    for i in (100, 250, 400):
        sample[i] += 5
    st.download_button(
        "Download sample CSV (with 3 injected spikes)",
        pd.DataFrame({"consumption_kw": sample}).to_csv(index=False),
        file_name="sample_meter.csv",
    )

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        num = df.select_dtypes("number")
        if num.empty:
            st.error("No numeric column found in this file.")
        else:
            vals = num.iloc[:, 0].dropna().values.astype(float)
            if len(vals) < WINDOW + 1:
                st.error(f"Need at least {WINDOW + 1} rows.")
            else:
                scaled = scaler.transform(vals.reshape(-1, 1))
                X = make_windows(scaled, WINDOW)
                pred = model.predict(X, verbose=0)
                err = np.max(np.abs(pred - X), axis=(1, 2))

                x = np.arange(WINDOW - 1, len(vals))
                fig, flags = draw(x, vals[WINDOW - 1 :], err, thr)
                st.pyplot(fig)

                c1, c2 = st.columns(2)
                c1.metric("Hours checked", len(err))
                c2.metric("Anomalies found", int(flags.sum()))

                if flags.any():
                    table = pd.DataFrame(
                        {
                            "hour_number": x[flags],
                            "consumption_kw": vals[WINDOW - 1 :][flags],
                            "error": err[flags],
                        }
                    ).sort_values("error", ascending=False)
                    st.subheader("Most suspicious hours")
                    st.dataframe(table.head(20), use_container_width=True)