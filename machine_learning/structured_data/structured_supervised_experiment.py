import serial
import time
import numpy as np
import pandas as pd
from collections import deque
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ================= SETTINGS =================
PORT = "COM7"
BAUD = 115200
FS = 860
WINDOW_SIZE = 300
RECORD_TIME = 20  # seconds per state

# ================= SERIAL CONNECTION =================
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
except:
    print("❌ Cannot open serial port")
    exit()

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
print("Calibrating baseline (5 sec)...")

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

# ================= DATA COLLECTION =================
raw_buffer = deque(maxlen=WINDOW_SIZE)
dataset = []

def record_state(label, name):
    print(f"\nRecording {name} for {RECORD_TIME} seconds...")
    start = time.time()

    while time.time() - start < RECORD_TIME:
        try:
            val = float(ser.readline().decode().strip()) - baseline_mean
            raw_buffer.append(val)

            if len(raw_buffer) == WINDOW_SIZE:
                features = extract_features(np.array(raw_buffer))
                dataset.append(features + [label])
        except:
            continue

while True:
    input("\nPress Enter for REST...")
    record_state(0, "REST")

    input("\nPress Enter for WEAK...")
    record_state(1, "WEAK")

    input("\nPress Enter for STRONG...")
    record_state(2, "STRONG")

    repeat = input("\nRepeat cycle? (y/n): ")
    if repeat.lower() != 'y':
        break

# ================= SAVE DATA =================
df = pd.DataFrame(dataset, columns=["RMS","MAV","VAR","ZC","SSC","Label"])
df.to_csv("structured_dataset.csv", index=False)
print("\nDataset saved as structured_dataset.csv")
print("Total samples collected:", len(df))

# ================= PREPARE DATA =================
X = df.drop("Label", axis=1)
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ================= MODELS =================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=3000),
    "SVM_RBF": SVC(kernel='rbf'),
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=12)
}

print("\n========== MODEL RESULTS ==========\n")

label_names = ["REST", "WEAK", "STRONG"]

best_model = None
best_accuracy = 0

for name, model in models.items():

    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)

    test_acc = accuracy_score(y_test, test_preds)
    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=5))

    cm = confusion_matrix(y_test, test_preds)

    print(name)
    print("Test Accuracy:", round(test_acc, 4))
    print("Cross Validation Accuracy:", round(cv_score, 4))

    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, 
                         index=label_names, 
                         columns=label_names)
    print(cm_df)

    # Save confusion matrix
    cm_df.to_csv(f"confusion_matrix_{name}.csv")

    print("----------------------------------")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = name

print("\nBest Model:", best_model)
print("Best Accuracy:", round(best_accuracy, 4))