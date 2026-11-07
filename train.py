import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

WINDOW = 24

# 1. Load the preprocessed data
data = np.load("data/processed.npz")
X_train = data["X_train"]
X_test = data["X_test"]
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# 2. Build the LSTM Autoencoder
#    Encoder compresses 24 hours into a small vector,
#    decoder tries to rebuild the same 24 hours from it.
inputs = keras.Input(shape=(WINDOW, 1))
x = layers.LSTM(32, return_sequences=True)(inputs)
x = layers.LSTM(16)(x)                          # bottleneck
x = layers.RepeatVector(WINDOW)(x)
x = layers.LSTM(16, return_sequences=True)(x)
x = layers.LSTM(32, return_sequences=True)(x)
outputs = layers.TimeDistributed(layers.Dense(1))(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mae")
model.summary()

# 3. Train (the model learns to reconstruct normal consumption)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    X_train,
    X_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# 4. Reconstruction error = how badly the model rebuilt each window
train_pred = model.predict(X_train, verbose=0)
train_err = np.max(np.abs(train_pred - X_train), axis=(1, 2))

test_pred = model.predict(X_test, verbose=0)
test_err = np.max(np.abs(test_pred - X_test), axis=(1, 2))

# 5. Threshold: windows with error above this are called anomalies
threshold = np.percentile(train_err, 97)
print("Threshold:", threshold)

anomalies = test_err > threshold
print("Anomalies in test data:", anomalies.sum(), "of", len(test_err))

# 6. Save everything for the app
model.save("data/lstm_ae.keras")
np.save("data/threshold.npy", threshold)
np.save("data/test_err.npy", test_err)

# 7. Save the loss curve (useful for the report)
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MAE loss")
plt.title("LSTM Autoencoder training")
plt.legend()
plt.tight_layout()
plt.savefig("data/loss_curve.png")
print("Saved: model, threshold, errors and loss curve in data/")