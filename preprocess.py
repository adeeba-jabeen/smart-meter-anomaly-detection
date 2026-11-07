import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/household_power_consumption.txt"
WINDOW = 24  # one sample = last 24 hours of data

# 1. Load data (fields are separated by ';' and missing values are '?')
print("Loading data...")
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values="?",
    low_memory=False,
)

# 2. Combine Date and Time into a single datetime index
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
)
df = df.set_index("datetime")

# 3. Use one column only: Global_active_power (kilowatts)
series = df["Global_active_power"].astype(float)
print("Missing values (before):", series.isna().sum())

# 4. Convert per-minute readings to hourly averages and fill gaps
series = series.resample("h").mean()
series = series.interpolate(limit_direction="both")
print("Total hours:", len(series))
print("Missing values (after):", series.isna().sum())

# 5. Chronological split: first 80% train, last 20% test
split = int(len(series) * 0.8)
train_raw = series.iloc[:split]
test_raw = series.iloc[split:]

# 6. Normalize to the 0-1 range (scaler is fitted on train data only)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw.values.reshape(-1, 1))
test_scaled = scaler.transform(test_raw.values.reshape(-1, 1))


# 7. Sliding window: each sample is 24 consecutive hours
def make_windows(data, window):
    out = []
    for i in range(len(data) - window + 1):
        out.append(data[i : i + window])
    return np.array(out)


X_train = make_windows(train_scaled, WINDOW)
X_test = make_windows(test_scaled, WINDOW)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 8. Save for train.py and app.py
np.savez(
    "data/processed.npz",
    X_train=X_train,
    X_test=X_test,
    test_index=test_raw.index.values[WINDOW - 1 :],
    test_raw=test_raw.values,
)
joblib.dump(scaler, "data/scaler.pkl")
print("Saved: data/processed.npz and data/scaler.pkl")