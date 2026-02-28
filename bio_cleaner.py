# bio_cleaner.py
import socket
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
from collections import deque

HOST = '127.0.0.1'
PORT_SIM = 65432
PORT_CLEAN = 65433
SAMPLE_RATE = 500
WINDOW_SIZE = 1024
STRIDE = 256
DT = 1.0 / SAMPLE_RATE

WINDOW_PLOT = 2000
raw_eeg_buf = deque([2048]*WINDOW_PLOT, maxlen=WINDOW_PLOT)
raw_emg_buf = deque([2048]*WINDOW_PLOT, maxlen=WINDOW_PLOT)
clean_eeg_buf = deque([2048]*WINDOW_PLOT, maxlen=WINDOW_PLOT)
clean_emg_buf = deque([2048]*WINDOW_PLOT, maxlen=WINDOW_PLOT)

proc_eeg_window = deque(maxlen=WINDOW_SIZE)
proc_emg_window = deque(maxlen=WINDOW_SIZE)
proc_mode_window = deque(maxlen=WINDOW_SIZE)

buf_lock = threading.Lock()

def fft_mask_clean(win, lowcut, highcut, notch=[50], notch_w=1, fs=SAMPLE_RATE):
    n = len(win)
    X = np.fft.rfft(win)
    freqs = np.fft.rfftfreq(n, 1/fs)

    bp = (freqs >= lowcut) & (freqs <= highcut)
    X[~bp] = 0

    for f in notch:
        mask = np.abs(freqs - f) <= notch_w
        X[mask] = 0

    return np.fft.irfft(X, n=n)

def simulator_reader_thread():
    print(f"[CLEANER] Connecting to simulator {HOST}:{PORT_SIM}")
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((HOST, PORT_SIM))
            print("[CLEANER] Connected.")
            break
        except:
            time.sleep(1)

    buf = b""
    try:
        while True:
            data = s.recv(4096)
            if not data:
                break
            buf += data

            new_eeg, new_emg, new_mode = [], [], []

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    v = line.decode().split(",")
                    eeg, emg, mode = int(v[0]), int(v[1]), v[2]
                    new_eeg.append(eeg)
                    new_emg.append(emg)
                    new_mode.append(mode)
                except:
                    pass

            if new_eeg:
                with buf_lock:
                    proc_eeg_window.extend(new_eeg)
                    proc_emg_window.extend(new_emg)
                    proc_mode_window.extend(new_mode)
                    raw_eeg_buf.extend(new_eeg)
                    raw_emg_buf.extend(new_emg)

    except Exception as e:
        print("[CLEANER] Read error:", e)
    finally:
        s.close()

def cleaned_stream_server(clean_q):
    print(f"[CLEANER] Clean server at {HOST}:{PORT_CLEAN}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT_CLEAN))
        server.listen(1)

        while True:
            conn, addr = server.accept()
            print(f"[CLEANER] Controller connected {addr}")

            with conn:
                while True:
                    if clean_q:
                        eeg, emg, mode = clean_q.pop(0)
                        conn.sendall(f"{int(eeg)},{int(emg)},{mode}\n".encode())
                    else:
                        time.sleep(0.001)
def processing_loop():
    clean_q = []
    threading.Thread(target=cleaned_stream_server,
                     args=(clean_q,), daemon=True).start()

    while True:
        with buf_lock:
            if len(proc_eeg_window) < WINDOW_SIZE:
                time.sleep(0.01)
                continue

            eeg_win = np.array(proc_eeg_window)
            emg_win = np.array(proc_emg_window)
            mode_win = list(proc_mode_window)


        eeg_c = eeg_win - np.mean(eeg_win)
        emg_c = emg_win - np.mean(emg_win)

        eeg_f = fft_mask_clean(eeg_c, 0.5, 35) + 2048
        emg_f = fft_mask_clean(emg_c, 20, 200) + 2048

        eeg_clean = eeg_f[-STRIDE:]
        emg_clean = emg_f[-STRIDE:]

        with buf_lock:
            clean_eeg_buf.extend(eeg_clean)
            clean_emg_buf.extend(emg_clean)

            for i in range(STRIDE):
                idx = WINDOW_SIZE - STRIDE + i
                clean_q.append((eeg_f[idx], emg_f[idx], mode_win[idx]))

            for _ in range(STRIDE):
                proc_eeg_window.popleft()
                proc_emg_window.popleft()
                proc_mode_window.popleft()

        time.sleep(0.01)

def plotting_loop():
    plt.ion()
    fig, axs = plt.subplots(4,1, figsize=(10,8), sharex=True)
    titles = ["Raw EEG", "Clean EEG", "Raw EMG", "Clean EMG"]
    bufs = [raw_eeg_buf, clean_eeg_buf, raw_emg_buf, clean_emg_buf]
    lines = []

    for i, ax in enumerate(axs):
        line, = ax.plot([], [])
        ax.set_title(titles[i])
        ax.set_ylim(0,4096)
        ax.set_xlim(0,WINDOW_PLOT)
        lines.append(line)

    while True:
        with buf_lock:
            d = [list(b) for b in bufs]

        x = range(len(d[0]))
        for i in range(4):
            lines[i].set_data(x, d[i])

        plt.pause(0.1)


if __name__ == "__main__":
    threading.Thread(target=simulator_reader_thread, daemon=True).start()
    threading.Thread(target=processing_loop, daemon=True).start()
    plotting_loop()
