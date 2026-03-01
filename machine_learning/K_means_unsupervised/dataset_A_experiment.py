import serial
import time
import numpy as np
import pandas as pd
from collections import deque
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ================= SETTINGS =================
PORT = "COM7"
BAUD = 115200
FS = 860
WINDOW_SIZE = 300
RECORD_TIME = 120  # 2 minutes

# ================= SERIAL =================
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
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(20, 350, FS)

def bandpass_filter(data):
    return lfilter(b, a, data)

def extract_features(window):
    window = np.array(window)
    window = bandpass_filter(window)
    window = window - np.mean(window)

    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    var = np.var(window)
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)

    return [rms, mav, var, zc, ssc]

# ================= SAFE BASELINE =================
print("Waiting for serial data...")

while True:
    try:
        line = ser.readline().decode().strip()
        float(line)
        break
    except:
        continue

print("Calibrating baseline (5 sec)...")

baseline_vals = []
start = time.time()

while time.time() - start < 5:
    try:
        line = ser.readline().decode().strip()
        baseline_vals.append(float(line))
    except:
        continue

if len(baseline_vals) == 0:
    print("❌ No baseline data received.")
    exit()

baseline_mean = np.mean(baseline_vals)
print("Baseline:", baseline_mean)

# ================= RECORD DATASET A =================
print("\nRecording natural EMG for 2 minutes...")
print("Relax and flex randomly.\n")

raw_buffer = deque(maxlen=WINDOW_SIZE)
features_dataset = []

start = time.time()

while time.time() - start < RECORD_TIME:

    try:
        line = ser.readline().decode().strip()
        value = float(line) - baseline_mean
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:
            features = extract_features(raw_buffer)
            features_dataset.append(features)

    except:
        continue

print("Total feature samples collected:", len(features_dataset))

# Save raw features
df_features = pd.DataFrame(
    features_dataset,
    columns=["RMS","MAV","VAR","ZC","SSC"]
)
df_features.to_csv("dataset_A_natural_features.csv", index=False)
print("dataset_A_natural_features.csv saved.")

# ================= KMEANS =================
print("\nApplying KMeans clustering...")

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(features_dataset)

sil_score = silhouette_score(features_dataset, cluster_labels)
print("Silhouette Score:", sil_score)

# Save clustered dataset
df_clusters = df_features.copy()
df_clusters["Cluster"] = cluster_labels
df_clusters.to_csv("dataset_A_with_clusters.csv", index=False)
print("dataset_A_with_clusters.csv saved.")

# ================= PREPARE DATA =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_dataset)
y = cluster_labels

# ================= MODELS =================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=3000),
    "SVM_RBF": SVC(kernel='rbf'),
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=12)
}

results = []

print("\n================ MODEL COMPARISON ================\n")

for name, model in models.items():

    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)

    train_acc = accuracy_score(y, preds)
    cv_score = np.mean(cross_val_score(model, X_scaled, y, cv=5))

    print(f"{name}")
    print("Training Accuracy:", round(train_acc,4))
    print("Cross-Validation Accuracy:", round(cv_score,4))

    cm = confusion_matrix(y, preds)
    print("Confusion Matrix:\n", cm)
    print("--------------------------------------------------")

    # Save confusion matrix
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(f"confusion_matrix_{name}.csv", index=False)

    results.append({
        "Model": name,
        "Training_Accuracy": train_acc,
        "CrossVal_Accuracy": cv_score,
        "Silhouette_Score": sil_score
    })

# Save model comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)

print("\nAll experiment results saved successfully.")