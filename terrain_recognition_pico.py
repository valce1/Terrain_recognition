"""
terrain_recognition_pico.py
============================
Terrain Recognition for Raspberry Pi Pico (MicroPython)

Hardware assumption
-------------------
* Triaxial accelerometer + triaxial gyroscope connected via I2C.
  Default driver targets MPU-6050 (I2C address 0x68).
  The driver class can be swapped for any other 6-DOF IMU.

Sensor placement
----------------
Designed for a sensor mounted on the handlebar OR behind the saddle.
If the sensor is mounted with a different orientation, the raw axes will
differ from the training orientation, but the features (RMS, std, p2p of
each axis, total-acc magnitude) are largely orientation-invariant for
vibration-based terrain classification.  Retrain with data collected in
the new position for best accuracy.

Classifier
----------
Nearest-Centroid (Euclidean distance after StandardScaler normalisation).
Parameters (SCALER_MEAN, SCALER_STD, CENTROIDS) are loaded from
model_params.py, which is generated on a PC by train_and_validate.py.

Usage
-----
1. Run train_and_validate.py on a PC to generate model_params.py.
2. Copy model_params.py and terrain_recognition_pico.py to the Pico.
3. Connect the IMU and set the I2C pins below.
4. Run main() or import this module.
"""

import math
import time
import struct

from machine import I2C, Pin

# ── load model parameters ──────────────────────────────────────────────────
try:
    from model_params import (
        CLASSES, WINDOW_SIZE, N_FEATURES,
        SCALER_MEAN, SCALER_STD, CENTROIDS
    )
except ImportError:
    raise ImportError(
        "model_params.py not found. "
        "Run train_and_validate.py on a PC first, then copy model_params.py here."
    )


# ===========================================================================
# Hardware configuration  ← edit these to match your wiring
# ===========================================================================
I2C_BUS  = 0          # I2C bus index (0 or 1)
SDA_PIN  = 4          # GP4
SCL_PIN  = 5          # GP5
I2C_FREQ = 400_000    # 400 kHz fast-mode

# MPU-6050 sample rate (approx. samples/s):
#   sample_rate = 1000 / (1 + SMPLRT_DIV)
#   SMPLRT_DIV = 9  → ~100 Hz
SMPLRT_DIV = 9

# Gyroscope full-scale:  0=±250°/s  1=±500  2=±1000  3=±2000
GYRO_FS = 0

# Accelerometer full-scale: 0=±2g  1=±4g  2=±8g  3=±16g
ACCEL_FS = 0

# How often to print the terrain label (in windows)
PRINT_EVERY = 1


# ===========================================================================
# MPU-6050 minimal driver
# ===========================================================================
class MPU6050:
    """Minimal MicroPython driver for MPU-6050 over I2C."""

    _ADDR         = 0x68
    _PWR_MGMT_1   = 0x6B
    _SMPLRT_DIV   = 0x19
    _CONFIG       = 0x1A
    _GYRO_CONFIG  = 0x1B
    _ACCEL_CONFIG = 0x1C
    _ACCEL_XOUT_H = 0x3B
    _TEMP_OUT_H   = 0x41
    _GYRO_XOUT_H  = 0x43

    # Sensitivity divisors  (raw integer → physical unit)
    _ACCEL_SENS = (16384.0, 8192.0, 4096.0, 2048.0)   # LSB/g → mg if *1000
    _GYRO_SENS  = (131.0, 65.5, 32.8, 16.4)            # LSB/(°/s)

    def __init__(self, i2c: I2C, addr: int = _ADDR,
                 accel_fs: int = 0, gyro_fs: int = 0,
                 smplrt_div: int = 9):
        self._i2c  = i2c
        self._addr = addr
        self._accel_div = self._ACCEL_SENS[accel_fs]
        self._gyro_div  = self._GYRO_SENS[gyro_fs]
        self._init(accel_fs, gyro_fs, smplrt_div)

    def _write(self, reg: int, value: int) -> None:
        self._i2c.writeto_mem(self._addr, reg, bytes([value]))

    def _read(self, reg: int, n: int) -> bytes:
        return self._i2c.readfrom_mem(self._addr, reg, n)

    def _init(self, accel_fs: int, gyro_fs: int, smplrt_div: int) -> None:
        self._write(self._PWR_MGMT_1,   0x00)          # wake up, use internal 8 MHz
        self._write(self._SMPLRT_DIV,   smplrt_div)
        self._write(self._CONFIG,        0x01)          # DLPF 188 Hz
        self._write(self._GYRO_CONFIG,  gyro_fs  << 3)
        self._write(self._ACCEL_CONFIG, accel_fs << 3)

    def read_raw(self) -> tuple[float, float, float, float, float, float]:
        """Return (ax, ay, az, gx, gy, gz) in mg and °/s."""
        buf = self._read(self._ACCEL_XOUT_H, 14)
        ax, ay, az, _, gx, gy, gz = struct.unpack(">7h", buf)
        ad = self._accel_div / 1000.0    # convert g → mg factor
        gd = self._gyro_div
        return (
            ax / ad, ay / ad, az / ad,
            gx / gd, gy / gd, gz / gd,
        )


# ===========================================================================
# Feature extraction  (pure Python, no numpy)
# ===========================================================================
def _stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, rms, peak-to-peak) for a 1-D list."""
    n    = len(values)
    mean = sum(values) / n
    var  = sum((v - mean) ** 2 for v in values) / n
    rms  = math.sqrt(sum(v * v for v in values) / n)
    p2p  = max(values) - min(values)
    return mean, math.sqrt(var), rms, p2p


def extract_features(window: list[list[float]]) -> list[float]:
    """
    window : list of WINDOW_SIZE samples, each sample = [ax, ay, az, gx, gy, gz]
    Returns a feature vector of length N_FEATURES (27).
    """
    # Transpose to per-axis lists
    axes = [[window[t][ch] for t in range(len(window))] for ch in range(6)]

    feats: list[float] = []
    for ax_data in axes:
        feats.extend(_stats(ax_data))   # mean, std, rms, p2p   → 4 per axis

    # Total acceleration magnitude
    mag = [math.sqrt(window[t][0] ** 2 + window[t][1] ** 2 + window[t][2] ** 2)
           for t in range(len(window))]
    _, mag_std, _, mag_p2p = _stats(mag)
    mag_mean = sum(mag) / len(mag)
    feats += [mag_mean, mag_std, mag_p2p]   # 3 extra features

    return feats                            # total: 6*4 + 3 = 27


# ===========================================================================
# Nearest-Centroid classifier  (pure Python)
# ===========================================================================
def _normalize(feats: list[float],
               mean: list[float],
               std:  list[float]) -> list[float]:
    return [(feats[i] - mean[i]) / (std[i] if std[i] != 0.0 else 1.0)
            for i in range(len(feats))]


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def predict(feats: list[float]) -> tuple[str, float]:
    """
    Returns (class_name, distance_to_nearest_centroid).
    Lower distance means higher confidence.
    """
    z = _normalize(feats, SCALER_MEAN, SCALER_STD)
    best_label = None
    best_dist  = float("inf")
    for idx, centroid in enumerate(CENTROIDS):
        d = _euclidean(z, centroid)
        if d < best_dist:
            best_dist  = d
            best_label = CLASSES[idx]
    return best_label, best_dist


# ===========================================================================
# Main loop
# ===========================================================================
def main() -> None:
    # ── initialise I2C and IMU ─────────────────────────────────────────────
    i2c = I2C(I2C_BUS, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=I2C_FREQ)
    imu = MPU6050(i2c,
                  accel_fs=ACCEL_FS,
                  gyro_fs=GYRO_FS,
                  smplrt_div=SMPLRT_DIV)

    print("Terrain Recognition – Raspberry Pi Pico")
    print(f"Window size : {WINDOW_SIZE} samples")
    print(f"Classes     : {CLASSES}")
    print("Collecting data…\n")

    # ── target sample interval ────────────────────────────────────────────
    sample_rate_hz = 1000 / (1 + SMPLRT_DIV)   # e.g. 100 Hz
    interval_us    = int(1_000_000 / sample_rate_hz)

    window   : list[list[float]] = []
    win_count: int = 0

    while True:
        t0 = time.ticks_us()

        # ── collect one sample ────────────────────────────────────────────
        try:
            sample = list(imu.read_raw())   # [ax, ay, az, gx, gy, gz]
        except OSError as exc:
            print(f"[WARN] IMU read error: {exc}")
            time.sleep_ms(10)
            continue

        window.append(sample)

        # ── classify when the window is full ──────────────────────────────
        if len(window) >= WINDOW_SIZE:
            feats          = extract_features(window)
            terrain, dist  = predict(feats)
            win_count     += 1

            if win_count % PRINT_EVERY == 0:
                print(f"[{win_count:>6d}]  Terrain: {terrain:<12s}  dist={dist:.3f}")

            # Slide: remove oldest half (50 % overlap, mirrors training)
            window = window[WINDOW_SIZE // 2 :]

        # ── busy-wait for next sample ─────────────────────────────────────
        elapsed = time.ticks_diff(time.ticks_us(), t0)
        wait    = interval_us - elapsed
        if wait > 0:
            time.sleep_us(wait)


# Run immediately when executed as main script
if __name__ == "__main__":
    main()
