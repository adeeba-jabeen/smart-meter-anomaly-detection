import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score

WINDOW = 24
rng = np.random.default_rng(42)

model = keras.models.load_model("data/lstm_ae.keras")
scaler = joblib.load("data/scaler.pkl")
data = np.load("data/processed.npz")
X_train = data["X_train"]
test_raw = data["test_raw"].astype(float)
n = len(test_raw)


def windows(arr):
    return np.array([arr[i : i + WINDOW] for i in range(len(arr) - WINDOW + 1)])


def scores(X):
    pred = model.predict(X, verbose=0)
    abs_err = np.abs(pred - X)[:, :, 0]
    return abs_err.mean(axis=1), abs_err.max(axis=1)


# 1. Inject artificial anomalies (same as before: 30 spikes, 30 zero drops)
idx = rng.choice(np.arange(WINDOW, n - WINDOW), size=60, replace=False)
corrupted = test_raw.copy()
corrupted[idx[:30]] += rng.uniform(3, 6, size=30)
corrupted[idx[30:]] = 0.0

is_anomaly = np.zeros(n, dtype=bool)
is_anomaly[idx] = True
labels = np.array([is_anomaly[i : i + WINDOW].any() for i in range(n - WINDOW + 1)])

# 2. Compute both error scores
X_test = windows(scaler.transform(corrupted.reshape(-1, 1)))
train_mean, train_max = scores(X_train)
test_mean, test_max = scores(X_test)

# 3. Compare scoring methods and thresholds
lines = [f"Anomalous windows: {labels.sum()} of {len(labels)}", ""]
for name, tr, te in [("mean error", train_mean, test_mean), ("max error", train_max, test_max)]:
    for pct in (99, 97, 95, 90):
        thr = np.percentile(tr, pct)
        flags = te > thr
        p = precision_score(labels, flags, zero_division=0)
        r = recall_score(labels, flags, zero_division=0)
        f = f1_score(labels, flags, zero_division=0)
        lines.append(
            f"{name:11} | percentile {pct} | thr={thr:.4f} | "
            f"P={p:.3f} R={r:.3f} F1={f:.3f} | flagged={flags.sum()}"
        )

result = "\n".join(lines)
print(result)
with open("data/results_compare.txt", "w") as fh:
    fh.write(result)
print("\nSaved: data/results_compare.txt")

# ===================== BASELINES + COMPARISON TABLE =====================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

PCT = 95  # sab models ke liye same percentile, taake comparison fair ho


def evaluate(name, tr_score, te_score):
    thr = np.percentile(tr_score, PCT)
    flags = te_score > thr
    return {
        "Model": name,
        "Precision": round(precision_score(labels, flags, zero_division=0), 3),
        "Recall": round(recall_score(labels, flags, zero_division=0), 3),
        "F1": round(f1_score(labels, flags, zero_division=0), 3),
        "Flagged": int(flags.sum()),
    }


# 1) Z-score baseline: har window ka sabse bada z-score
mu = X_train[:, :, 0].mean()
sd = X_train[:, :, 0].std()
z_train = (np.abs(X_train[:, :, 0] - mu) / sd).max(axis=1)
z_test = (np.abs(X_test[:, :, 0] - mu) / sd).max(axis=1)

# 2) Isolation Forest baseline (flattened windows par)
iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
iso.fit(X_train.reshape(len(X_train), -1))
iso_train = -iso.score_samples(X_train.reshape(len(X_train), -1))
iso_test = -iso.score_samples(X_test.reshape(len(X_test), -1))

table = pd.DataFrame([
    evaluate("Z-score", z_train, z_test),
    evaluate("Isolation Forest", iso_train, iso_test),
    evaluate("LSTM AE (mean error)", train_mean, test_mean),
    evaluate("LSTM AE (max error)", train_max, test_max),
])
print("\n", table.to_string(index=False))
table.to_csv("data/comparison_table.csv", index=False)

ax = table.set_index("Model")[["Precision", "Recall", "F1"]].plot(kind="bar", figsize=(8, 4.5))
ax.set_title("Model Comparison on Injected Anomalies")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("data/comparison_chart.png", dpi=150)
print("Saved: data/comparison_table.csv, data/comparison_chart.png")