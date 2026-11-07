import numpy as np
from tensorflow import keras

PERCENTILE = 97

model = keras.models.load_model("data/lstm_ae.keras")
data = np.load("data/processed.npz")
X_train, X_test = data["X_train"], data["X_test"]


def max_error(X):
    pred = model.predict(X, verbose=0)
    return np.max(np.abs(pred - X), axis=(1, 2))


train_err = max_error(X_train)
test_err = max_error(X_test)
threshold = np.percentile(train_err, PERCENTILE)

np.save("data/threshold.npy", threshold)
np.save("data/test_err.npy", test_err)
print("New threshold:", threshold)
print("Anomalies in test data:", (test_err > threshold).sum(), "of", len(test_err))