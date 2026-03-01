# bio_controller_train_run.py

import socket
import time
import numpy as np
import threading
import joblib
from collections import deque
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pynput import keyboard
STOP = False
def on_press(key):
    global STOP
    if key == keyboard.Key.esc:
        STOP = True
listener = keyboard.Listener(on_press=on_press)
listener.start()

HOST = '127.0.0.1'
PORT_CLEAN = 65433
SAMPLE_RATE = 500

buffer_eeg = deque(maxlen=SAMPLE_RATE*10)  
buffer_emg = deque(maxlen=SAMPLE_RATE*10)  
buffer_mode = deque(maxlen=SAMPLE_RATE*10)
recv_lock = threading.Lock()
running = True

def connection_reader():
    """Connects to cleaner server and reads cleaned lines into buffers"""
    global running
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            s.connect((HOST, PORT_CLEAN))
            print("[CTRL] Connected to cleaner at", HOST, PORT_CLEAN)
            break
        except ConnectionRefusedError:
            print("[CTRL] Cleaner not ready, retrying in 1s...")
            time.sleep(1.0)

    buf = b''
    try:
        while running and not STOP:
            data = s.recv(4096)
            if not data:
                break
            buf += data
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                try:
                    parts = line.decode().strip().split(',')
                    if len(parts) >= 3:
                        eeg = float(parts[0])
                        emg = float(parts[1])
                        mode = parts[2]
                        with recv_lock:
                            buffer_eeg.append(eeg)
                            buffer_emg.append(emg)
                            buffer_mode.append(mode)
                except Exception:
                    pass
    except Exception as e:
        print("[CTRL] Reader error:", e)
    finally:
        s.close()
        print("[CTRL] Connection closed.")

def extract_features_segment(eeg_seg, emg_seg, fs=SAMPLE_RATE):
    """
    Takes one-second segments (numpy arrays) and returns a 10-d feature vector:
    [eeg_mean, eeg_rms, eeg_peak, eeg_env,
     emg_mean, emg_rms, emg_mav, emg_zc,
     eeg_mf, emg_mf]
    Note: signals are centered inside this function.
    """
    # center signals
    eeg = eeg_seg - np.mean(eeg_seg)
    emg = emg_seg - np.mean(emg_seg)

    # time-domain features
    eeg_mean = np.mean(eeg)
    eeg_rms = np.sqrt(np.mean(eeg**2))
    eeg_peak = np.max(np.abs(eeg))
    eeg_env = np.mean(np.abs(eeg))

    emg_mean = np.mean(emg)
    emg_rms = np.sqrt(np.mean(emg**2))
    emg_mav = np.mean(np.abs(emg))
    emg_zc = np.sum(np.diff(np.sign(emg)) != 0) / len(emg)

    # spectral features
    n = len(eeg)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    EegF = np.abs(np.fft.rfft(eeg))
    EmgF = np.abs(np.fft.rfft(emg))

    def median_freq(spec, freqs):
        cumsum = np.cumsum(spec)
        idx = np.searchsorted(cumsum, cumsum[-1]/2.0)
        return freqs[idx] if idx < len(freqs) else freqs[-1]

    eeg_mf = median_freq(EegF, freqs) + np.random.normal(0, 0.2)
    emg_mf = median_freq(EmgF, freqs) + np.random.normal(0, 0.5)
    jitter = np.random.normal(0, 0.01, 10)

    feat = np.array([
        eeg_mean, eeg_rms, eeg_peak, eeg_env,
        emg_mean, emg_rms, emg_mav, emg_zc,
        eeg_mf, emg_mf
    ]) + jitter

    return feat

SEGMENT_SECONDS = 1.0
TOTAL_SEGMENTS = 100 

def collect_training_data():
    X, y_class, y_reg = [], [], []
    print("[CTRL] Waiting for cleaned stream to fill buffers...")
    while len(buffer_mode) < SAMPLE_RATE:
        time.sleep(0.5)
        if STOP:
            return X, y_class, y_reg

    collected = 0
    print(f"[CTRL] Collecting {TOTAL_SEGMENTS} segments for training...")

    while collected < TOTAL_SEGMENTS and not STOP:
        seg_len = int(SEGMENT_SECONDS * SAMPLE_RATE)
        with recv_lock:
            if len(buffer_eeg) < seg_len:
                pass
            else:
                eeg_seg = np.array(list(buffer_eeg)[-seg_len:])
                emg_seg = np.array(list(buffer_emg)[-seg_len:])
                mode_seg = list(buffer_mode)[-seg_len:]

                feat = extract_features_segment(eeg_seg, emg_seg)
                X.append(feat)
                if any(m in ("EMG_FLEX", "EMG_CLENCH") for m in mode_seg):
                    y_class.append(1)
                else:
                    y_class.append(0)
                y_reg.append(np.mean(np.abs(eeg_seg - np.mean(eeg_seg))) * (0.95 + 0.1*np.random.rand()))

                collected += 1
                print(f"[CTRL] Collected segment {collected}/{TOTAL_SEGMENTS}")

        time.sleep(0.05)

    return np.array(X), np.array(y_class), np.array(y_reg)
def train_models(X, y_class, y_reg):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SVC(kernel='rbf', probability=True)
    reg = SVR(kernel='rbf')
    print("[CTRL] Training SVM classifier...")
    clf.fit(Xs, y_class)
    print("[CTRL] Training SVR regressor...")
    reg.fit(Xs, y_reg)
    joblib.dump({'scaler': scaler, 'svm': clf, 'svr': reg}, "models_ctrl.pkl")
    print("[CTRL] Models saved to models_ctrl.pkl")
    return scaler, clf, reg

class DummyModel:
    def predict(self, X):
        return [1 if x[5] > 60.0 else 0 for x in X]

class DummyRegressor:
    def predict(self, X):
        return [float(x[3]) * 15.0 for x in X]


def inference_loop(scaler, svm, svr):
    print("[CTRL] Starting continuous inference. Press ESC to stop.")
    try:
        while not STOP:
            seg_len = int(1.0 * SAMPLE_RATE)
            with recv_lock:
                if len(buffer_eeg) < seg_len:
                    time.sleep(0.05)
                    continue
                eeg_seg = np.array(list(buffer_eeg)[-seg_len:])
                emg_seg = np.array(list(buffer_emg)[-seg_len:])

            feat = extract_features_segment(eeg_seg, emg_seg)
            Xs = np.array([feat])

            if scaler is not None:
                Xs_trans = scaler.transform(Xs)
                class_pred = svm.predict(Xs_trans)[0]
                reg_pred = svr.predict(Xs_trans)[0]
            else:
                class_pred = DummyModel().predict(Xs)[0]
                reg_pred = DummyRegressor().predict(Xs)[0]

            dx = int(np.clip(reg_pred * 15.0, -30, 30))

            if class_pred == 1:
                click_text = "CLICK"
            else:
                click_text = "NO CLICK"

            move_text = "MOVING" if abs(dx) > 3 else "NOT MOVING"

            print(f"[CTRL] {click_text} -> dx={dx} | {move_text}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("[CTRL] Stopped inference loop.")

if __name__ == "__main__":
    t = threading.Thread(target=connection_reader, daemon=True)
    t.start()
    time.sleep(2.0)

    X, y_class, y_reg = collect_training_data()
    print("[CTRL] Feature matrix shape:", X.shape)

    if len(np.unique(y_class)) >= 2:
        scaler, svm, svr = train_models(X, y_class, y_reg)
    else:
        print("[CTRL] Not enough classes for training. Using fallback dummy models.")
        scaler, svm, svr = None, None, None

    try:
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.title("Class labels")
        plt.bar(range(len(y_class)), y_class)
        plt.subplot(1, 2, 2)
        plt.title("Regression targets (EEG envelope)")
        plt.plot(y_reg, 'o-')
        plt.tight_layout()
        plt.show(block=False)
    except Exception:
        pass

    inference_loop(scaler, svm, svr)
    running = False
    print("[CTRL] Exiting.")
