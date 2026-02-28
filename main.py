import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from sklearn.svm import SVC
import keyboard
import sys

PORT = "COM7"
BAUD = 115200
WINDOW_SIZE = 80
GRAPH_LEN = 300
DOUBLE_STRONG_WINDOW = 0.6
CLICK_COOLDOWN = 1.2

# ---------------- SERIAL ----------------
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
except:
    print("❌ Cannot open serial port")
    sys.exit()

# ---------------- BUFFERS ----------------
raw_buffer = deque(maxlen=WINDOW_SIZE)
graph_buffer = deque(maxlen=GRAPH_LEN)
state_buffer = deque(maxlen=10)

X_train, y_train = [], []

axis = "X"
direction = 1
last_state = None
last_strong_time = 0
last_click_time = 0

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(window):
    window = np.array(window)
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    var = np.var(window)
    return [rms, mav, var]

# ================= TRAINING =================
print("\nPress T to start training...")
while not keyboard.is_pressed('t'):
    pass

print("\n=== TRAINING MODE ===")
print("R → REST")
print("W → WEAK")
print("S → STRONG")
print("Q → Finish\n")

# Baseline calibration
print("Calibrating REST baseline (5 sec)... Relax.")
baseline_vals = []
start = time.time()

while time.time() - start < 5:
    try:
        line = ser.readline().decode().strip()
        if line:
            baseline_vals.append(abs(float(line)))
    except:
        pass

baseline_mean = np.mean(baseline_vals)
print("Baseline:", baseline_mean)

current_label = None

while True:

    if keyboard.is_pressed('r'):
        current_label = 0
        print(">> Collecting REST")
        time.sleep(0.5)

    elif keyboard.is_pressed('w'):
        current_label = 1
        print(">> Collecting WEAK")
        time.sleep(0.5)

    elif keyboard.is_pressed('s'):
        current_label = 2
        print(">> Collecting STRONG")
        time.sleep(0.5)

    elif keyboard.is_pressed('q'):
        break

    try:
        line = ser.readline().decode().strip()
        if not line or current_label is None:
            continue

        value = float(line)
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:
            features = extract_features(raw_buffer)
            features[0] = (features[0] - baseline_mean) / baseline_mean
            X_train.append(features)
            y_train.append(current_label)

    except:
        pass

print("Training samples:", len(X_train))

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("✅ Model trained")
print("=== SIMULATION MODE ===\n")

state_buffer.clear()

# ---------------- CLASSIFY ----------------
def classify(pred):
    return ["REST", "WEAK", "STRONG"][pred]

# ---------------- LIVE UPDATE ----------------
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], lw=2)
ax.set_ylim(0, 2.5)
ax.set_title("Live EMG State")

def update(frame):
    global axis, direction
    global last_state, last_strong_time, last_click_time

    try:
        line = ser.readline().decode().strip()
        if not line:
            return line_plot,

        value = float(line)
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:

            features = extract_features(raw_buffer)
            features[0] = (features[0] - baseline_mean) / baseline_mean

            pred = model.predict([features])[0]
            graph_buffer.append(pred)

            raw_state = classify(pred)
            state_buffer.append(raw_state)
            state = max(set(state_buffer), key=state_buffer.count)

            now = time.time()

            # Print only when state changes
            if state != last_state:

                print(f"\nState → {state}")

                if state == "REST":
                    print(f"Action → MOVE_{axis} (Direction {direction})")

                elif state == "WEAK":
                    direction *= -1
                    print("Action → REVERSE_DIRECTION")

                elif state == "STRONG":

                    if (now - last_strong_time) < DOUBLE_STRONG_WINDOW and \
                       (now - last_click_time) > CLICK_COOLDOWN:

                        print("Action → CLICK")
                        last_click_time = now
                        last_strong_time = 0

                    else:
                        axis = "Y" if axis == "X" else "X"
                        print(f"Action → SWITCH_AXIS → {axis}")
                        last_strong_time = now

                last_state = state

            line_plot.set_data(range(len(graph_buffer)), graph_buffer)
            ax.set_xlim(0, len(graph_buffer))

    except:
        pass

    return line_plot,

ani = animation.FuncAnimation(fig, update, interval=20)
plt.show()