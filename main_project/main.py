import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import pandas as pd

import os
import sys
import joblib

from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ================= SETTINGS =================
PORT = "COM7"
BAUD = 115200
FS = 860
WINDOW_SIZE = 180
GRAPH_LEN = 500

MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "structured_dataset.csv"

PROB_THRESHOLD = 0.60
STRONG_CONFIRM_FRAMES = 3
SMOOTHING_WINDOW = 25

# ================= SERIAL =================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
except:
    print("❌ Cannot open serial port")
    sys.exit()

# ================= FILTER =================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

b, a = butter_bandpass(20, 350, FS)

def bandpass_filter(data):
    return lfilter(b, a, data)

# ================= FEATURE EXTRACTION =================
def extract_features(window):
    window = bandpass_filter(window)
    window = window - np.mean(window)

    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    var = np.var(window)
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)

    return [rms, mav, var, zc, ssc]

# ================= BASELINE =================
print("Calibrating baseline (5 sec)... Relax.")
baseline_vals = []
start = time.time()

while time.time() - start < 5:
    try:
        val = float(ser.readline().decode().strip())
        baseline_vals.append(val)
    except:
        continue

baseline_mean = np.mean(baseline_vals)
print("Baseline:", baseline_mean)

# ================= MODE SELECTION =================
print("\nSelect Mode:")
print("1 → Train / Improve Model")
print("2 → Load Existing Model")

choice = input("Enter choice: ")

X_session = []
y_session = []

# ================= TRAINING MODE =================
if choice == "1":

    raw_buffer = deque(maxlen=WINDOW_SIZE)

    def record_state(label, name):
        print(f"\nRecording {name} for 10 seconds...")
        start = time.time()

        while time.time() - start < 10:
            try:
                val = float(ser.readline().decode().strip()) - baseline_mean
                raw_buffer.append(val)

                if len(raw_buffer) == WINDOW_SIZE:
                    features = extract_features(np.array(raw_buffer))
                    X_session.append(features)
                    y_session.append(label)
            except:
                continue

    while True:
        record_state(0, "REST")
        record_state(1, "WEAK")
        record_state(2, "STRONG")

        repeat = input("Repeat another cycle? (y/n): ")
        if repeat.lower() != "y":
            break

    print("Total session samples:", len(X_session))

    # ================= APPEND TO DATASET =================
    df_new = pd.DataFrame(
        np.column_stack((X_session, y_session)),
        columns=["RMS","MAV","VAR","ZC","SSC","Label"]
    )

    if os.path.exists(DATASET_PATH):
        df_old = pd.read_csv(DATASET_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(DATASET_PATH, index=False)
    print("Dataset Permanently Updated.")

    # ================= RETRAIN FULL MODEL =================
    X_full = df_all.drop("Label", axis=1)
    y_full = df_all["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_scaled, y_full)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Model Retrained on Full Dataset and Saved.")

# ================= LOAD MODE =================
elif choice == "2":

    if not os.path.exists(MODEL_PATH):
        print("❌ No saved model found.")
        sys.exit()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model Loaded.")

else:
    print("Invalid choice.")
    sys.exit()

# ================= LIVE MODE =================
print("\n=== LIVE MODE STARTED ===")

raw_buffer = deque(maxlen=WINDOW_SIZE)
raw_graph = deque(maxlen=GRAPH_LEN)
state_graph = deque(maxlen=GRAPH_LEN)
state_buffer = deque(maxlen=SMOOTHING_WINDOW)

last_state = None
strong_counter = 0

fig, (ax1, ax2) = plt.subplots(2, 1)

raw_line, = ax1.plot([], [], lw=1)
state_line, = ax2.plot([], [], lw=2)

ax1.set_title("Raw EMG Signal")
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlim(0, GRAPH_LEN)

ax2.set_title("Predicted State")
ax2.set_ylim(-0.5, 2.5)
ax2.set_xlim(0, GRAPH_LEN)
ax2.set_yticks([0,1,2])
ax2.set_yticklabels(["REST","WEAK","STRONG"])

def update(frame):
    global last_state, strong_counter

    try:
        val = float(ser.readline().decode().strip()) - baseline_mean
        raw_buffer.append(val)
        raw_graph.append(val)
        raw_line.set_data(range(len(raw_graph)), raw_graph)

        if len(raw_buffer) < WINDOW_SIZE:
            return raw_line, state_line

        features = extract_features(np.array(raw_buffer))
        features_df = pd.DataFrame([features],columns=["RMS","MAV","VAR","ZC","SSC"])

        features_scaled = scaler.transform(features_df)

        probs = model.predict_proba(features_scaled)[0]
        pred = np.argmax(probs)

        # Probability threshold
        if max(probs) < PROB_THRESHOLD:
            pred = last_state if last_state is not None else 0

        # Strong confirmation
        if pred == 2:
            strong_counter += 1
        else:
            strong_counter = 0

        if strong_counter < STRONG_CONFIRM_FRAMES and pred == 2:
            pred = 1

        state_buffer.append(pred)
        state = max(set(state_buffer), key=state_buffer.count)

        if state != last_state:
            print("State →", ["REST","WEAK","STRONG"][state])
            last_state = state

        state_graph.append(state)
        state_line.set_data(range(len(state_graph)), state_graph)

    except:
        pass

    return raw_line, state_line

ani = animation.FuncAnimation(fig, update, interval=10)
plt.tight_layout()
plt.show()

print("\nSession Complete.")