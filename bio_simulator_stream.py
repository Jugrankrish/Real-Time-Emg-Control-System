# bio_simulator_stream.py
import socket
import time
import numpy as np
import threading
import math
import random

HOST = '127.0.0.1'
PORT_SIM = 65432
SAMPLE_RATE = 500
DT = 1.0 / SAMPLE_RATE

MODES = ["REST", "EMG_FLEX", "EMG_CLENCH", "EEG_ACTIVE"]

class BioSim:
    def __init__(self, fs):
        self.fs = fs
        self.t = 0.0
        self.mode = "REST"
        self.mode_change_time = time.time()
        self.mode_duration = 5
        self.rng = np.random.default_rng()

    def maybe_switch_mode(self):
        if time.time() - self.mode_change_time > self.mode_duration:
            self.mode = random.choice(MODES)
            self.mode_change_time = time.time()
            self.mode_duration = 4 + random.uniform(-1, 2)

    def eeg_sample(self, t):

        alpha = 25 * math.sin(2*np.pi*(8 + 4*self.rng.random()) * t)

        beta = 15 * math.sin(2*np.pi*(15 + 10*self.rng.random()) * t)

        burst = 0.0
        if self.mode == "EEG_ACTIVE" and self.rng.random() < 0.01:
            freq = 15 + 15*self.rng.random()
            amp  = 40 + 60*self.rng.random()
            burst = amp * math.sin(2*np.pi*freq*t)

        drift = 6 * math.sin(2*np.pi*0.3*t) 
        noise = 25 * self.rng.normal() 
        mains = 70 * math.sin(2*np.pi*50*t)      

        eeg = alpha + beta + burst + drift + noise + mains
        return int(np.clip(2048 + eeg, 0, 4095))
    def emg_sample(self, t):

        base_freq = 8 + 4*self.rng.random()
        baseline = 15 * math.sin(2*np.pi*base_freq*t) + 10*self.rng.normal()

        burst_prob = {
            "REST": 0.002,
            "EMG_FLEX": 0.01,
            "EMG_CLENCH": 0.03,
            "EEG_ACTIVE": 0.002
        }[self.mode]

        burst = 0.0
        if self.rng.random() < burst_prob:
            freq = 60 + 100*self.rng.random()
            amp  = 80 + 120*self.rng.random()
            burst = amp * math.sin(2*np.pi*freq*t)

        raw = baseline + burst
        return int(np.clip(2048 + raw*3, 0, 4095))

    def step(self):
        self.t += DT
        self.maybe_switch_mode()

        eeg = self.eeg_sample(self.t)
        emg = self.emg_sample(self.t)

        return eeg, emg, self.mode


def client_handler(conn, addr, sim):
    print(f"[SIM] Connected client: {addr}")
    with conn:
        while True:
            eeg, emg, mode = sim.step()
            conn.sendall(f"{eeg},{emg},{mode}\n".encode())
            time.sleep(DT)


def run_server():
    print(f"[SIM] Waiting for connection on {HOST}:{PORT_SIM}")
    sim = BioSim(SAMPLE_RATE)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT_SIM))
        s.listen(1)
        print("[SIM] Simulator running.")

        while True:
            conn, addr = s.accept()
            threading.Thread(target=client_handler,
                             args=(conn, addr, sim),
                             daemon=True).start()


if __name__ == "__main__":
    run_server()
