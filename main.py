import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from sklearn.svm import SVC
import pyautogui
import keyboard
import sys

PORT = "COM7"
BAUD = 115200
WINDOW_SIZE = 50
GRAPH_LEN = 300
CLICK_COOLDOWN = 1.2

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
except:
    print("❌ ERROR: Cannot open serial port")
    print("Check COM port and close Arduino Serial Monitor")
    sys.exit()

raw_buffer = deque(maxlen=WINDOW_SIZE)
graph_buffer = deque(maxlen=GRAPH_LEN)

X_train, y_train = [], []

last_click_time = 0
last_state = "REST"
cursor_x, cursor_y = pyautogui.position()
direction = 1


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(window):
    window = np.array(window)
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    var = np.var(window)
    return [rms, mav, var]


# ================= TRAINING MODE =================
print("\nPress T to start Training Mode...")

while not keyboard.is_pressed('t'):
    pass

print("\n=== TRAINING MODE ===")
print("Press:")
print(" R → REST")
print(" W → WEAK FLEX")
print(" S → STRONG FLEX")
print(" Q → Finish Training\n")

# ----- Baseline Calibration -----
print("Calibrating REST baseline (5 seconds)... Relax muscle.")
baseline_vals = []
start = time.time()

while time.time() - start < 5:
    try:
        line = ser.readline().decode().strip()
        if line:
            baseline_vals.append(abs(float(line)))
    except:
        pass

if len(baseline_vals) == 0:
    print("No baseline data collected.")
    sys.exit()

baseline_mean = np.mean(baseline_vals)
print("Baseline RMS reference:", baseline_mean)

current_label = None

# ----- Manual Training -----
while True:

    if keyboard.is_pressed('r'):
        current_label = 0
        print(">> REST")
        time.sleep(0.4)

    elif keyboard.is_pressed('w'):
        current_label = 1
        print(">> WEAK")
        time.sleep(0.4)

    elif keyboard.is_pressed('s'):
        current_label = 2
        print(">> STRONG")
        time.sleep(0.4)

    elif keyboard.is_pressed('q'):
        print("\nTraining Finished.")
        break

    try:
        line = ser.readline().decode().strip()
        if not line or current_label is None:
            continue

        value = float(line)
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:
            features = extract_features(raw_buffer)

            # Normalize RMS feature
            features[0] = (features[0] - baseline_mean) / baseline_mean

            X_train.append(features)
            y_train.append(current_label)

    except:
        pass

if len(X_train) == 0:
    print("❌ No training data collected")
    sys.exit()

print("✅ Training samples:", len(X_train))

# ----- Train Model -----
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("✅ SVC Model Trained Successfully")
print("=== SIMULATION MODE STARTED ===\n")


# ================= CLASSIFICATION =================
def classify(pred):
    if pred == 0:
        return "REST"
    elif pred == 1:
        return "WEAK"
    else:
        return "STRONG"


# ================= LIVE PLOTTING =================
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], lw=2)
ax.set_ylim(0, 2.5)
ax.set_title("Live EMG State")
ax.set_ylabel("State")
ax.set_xlabel("Samples")


def update(frame):
    global last_click_time, last_state, cursor_x, cursor_y, direction

    try:
        line = ser.readline().decode().strip()
        if not line:
            return line_plot,

        value = float(line)
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:
            features = extract_features(raw_buffer)

            # Normalize RMS
            features[0] = (features[0] - baseline_mean) / baseline_mean

            pred = model.predict([features])[0]
            graph_buffer.append(pred)

            state = classify(pred)
            print(f"State → {state}")

            now = time.time()

            # ----- REST: Move cursor -----
            if state == "REST":
                cursor_x += 3 * direction
                if cursor_x < 100 or cursor_x > 1200:
                    direction *= -1
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.02)

            # ----- STRONG: Click -----
            elif state == "STRONG":
                if last_state != "STRONG" and (now - last_click_time) > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = now
                    print("🖱 CLICK")

            last_state = state

            line_plot.set_data(range(len(graph_buffer)), graph_buffer)
            ax.set_xlim(0, len(graph_buffer))

    except:
        pass

    return line_plot,


ani = animation.FuncAnimation(fig, update, interval=20)
plt.show()