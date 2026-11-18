"""bench_sensor_check.py -- repeatedly query the BNO055 at a single, fixed
pose (nothing moves in between) to check whether the garbled/pole-stuck
readings seen during the last axis-sweep run (bench_sweep_test4.csv, poses
14/18/20/21) are a sensor/I2C reliability problem -- independent of any
question about the mechanism or the model, since the pose never changes
here.

Reuses the existing G-command protocol (no firmware change needed): sending
the same G,<Iq1>,<Iq2> repeatedly is a no-op move each time (already at that
position), so it just re-settles and re-reads the BNO055 over and over.

    python bench_sensor_check.py --port /dev/tty.usbserial-10
    python bench_sensor_check.py --port /dev/tty.usbserial-10 --n 30 --pose -16,0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import serial


def wait_for(ser, token, seconds=8):
    deadline = time.time() + seconds
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print("arduino:", line)
        if line == token:
            return True
    print(f"waarschuwing: geen '{token}' gezien binnen {seconds}s, ga toch door")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="bv. /dev/tty.usbserial-10")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=20,
                    help="aantal herhaalde metingen op dezelfde pose")
    ap.add_argument("--pose", default="0,0",
                    help="Iq1,Iq2 in deg om op te houden terwijl er herhaald "
                         "gemeten wordt, bv. '-16,0'")
    args = ap.parse_args()

    iq1_deg, iq2_deg = (float(v) for v in args.pose.split(","))

    ser = serial.Serial(args.port, args.baud, timeout=10)
    wait_for(ser, "READY")
    ser.reset_input_buffer()

    input(f"\nZet het mechanisme fysiek op Iq1={iq1_deg:+.1f} Iq2={iq2_deg:+.1f} "
          "en druk dan Enter -- daarna NIET meer aanraken terwijl de test "
          "loopt (er wordt hier niet bewogen, alleen herhaald gelezen)...")

    quats = []
    for i in range(args.n):
        ser.write(f"G,{iq1_deg:.4f},{iq2_deg:.4f}\n".encode())
        ser.flush()
        resp = ser.readline().decode(errors="replace").strip()
        try:
            q = np.array([float(v) for v in resp.split(",")])
            if q.shape != (4,):
                raise ValueError
        except ValueError:
            print(f"  {i:>3}: onverwacht antwoord {resp!r}")
            quats.append(None)
            continue
        mag = np.linalg.norm(q)
        flag = "" if abs(mag - 1.0) < 0.02 else "  <-- GEEN EENHEIDSQUATERNION"
        print(f"  {i:>3}: w={q[0]:+.4f} x={q[1]:+.4f} y={q[2]:+.4f} "
              f"z={q[3]:+.4f}  |q|={mag:.4f}{flag}")
        quats.append(q)
    ser.close()

    good = [q for q in quats if q is not None and abs(np.linalg.norm(q) - 1.0) < 0.02]
    bad_n = len(quats) - len(good)
    print(f"\n{len(quats)} pogingen, {bad_n} duidelijk ongeldig "
          "(geen eenheidslengte of kapotte regel)")

    if len(good) >= 2:
        ref = good[0] / np.linalg.norm(good[0])
        angs = []
        for q in good[1:]:
            qn = q / np.linalg.norm(q)
            dot = np.clip(abs(np.dot(ref, qn)), -1.0, 1.0)  # |.|: q and -q are the same rotation
            angs.append(np.degrees(2 * np.arccos(dot)))
        angs = np.array(angs)
        print(f"spreiding t.o.v. de 1e geldige meting: gem {angs.mean():.2f} deg   "
              f"max {angs.max():.2f} deg")
        med = np.median(angs)
        outliers = int(np.sum(angs > 5 * med + 1.0))
        if outliers:
            print(f"  waarvan {outliers} duidelijke uitschieter(s) "
                  "(>5x de mediaan-afwijking) -- waarschijnlijk dezelfde "
                  "soort glitch als in de laatste sweep-run")
        else:
            print("  geen uitschieters -- de sensorverbinding is hier stabiel")


if __name__ == "__main__":
    main()
