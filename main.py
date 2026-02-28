import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from sklearn.linear_model import LinearRegression
import pyautogui
import sys

PORT = "COM7" 
BAUD = 115200
TRAIN_TIME = 60 
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

print("\n=== EMG ML TRAINING MODE ===")
print("0–20 sec  : REST")
print("20–40 sec : WEAK FLEX")
print("40–60 sec : STRONG FLEX\n")

start = time.time()

def extract_features(window):
    window = np.array(window)
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    var = np.var(window)
    return [rms, mav, var]

while time.time() - start < TRAIN_TIME:
    try:
        line = ser.readline().decode().strip()
        if not line:
            continue

        value = float(line)
        raw_buffer.append(value)

        if len(raw_buffer) == WINDOW_SIZE:
            features = extract_features(raw_buffer)
            elapsed = time.time() - start

            if elapsed < 20:
                label = 0.0
            elif elapsed < 40:
                label = 0.5
            else:
                label = 1.0

            X_train.append(features)
            y_train.append(label)

    except:
        pass

if len(X_train) == 0:
    print("❌ ERROR: No training data received")
    print("Close Arduino Serial Monitor and retry")
    sys.exit()

print("✅ Training samples:", len(X_train))

model = LinearRegression()
model.fit(X_train, y_train)
print("✅ ML model trained\n")

def classify(pred):
    if pred < 0.3:
        return "REST"
    elif pred < 0.7:
        return "WEAK"
    else:
        return "STRONG"

fig, ax = plt.subplots()
line_plot, = ax.plot([], [], lw=2)
ax.set_ylim(0, 1.1)
ax.set_title("Live EMG ML Output")
ax.set_ylabel("Intensity")
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
            pred = model.predict([features])[0]
            graph_buffer.append(pred)

            state = classify(pred)
            print(f"EMG: {pred:.2f} → {state}")

            now = time.time()

            if state == "REST":
                cursor_x += 3 * direction
                if cursor_x < 100 or cursor_x > 1200:
                    direction *= -1
                pyautogui.moveTo(cursor_x, cursor_y, duration=0.02)

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