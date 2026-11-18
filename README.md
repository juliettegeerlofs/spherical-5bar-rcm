# Workspace-Driven Design and Experimental Validation of a Spherical Parallel 5-Bar RCM Mechanism

Code and raw measurement data accompanying the paper *"Workspace-Driven Design
and Experimental Validation of a Spherical Parallel 5-Bar RCM Mechanism"* (Juliette
Geerlofs, TU Delft). This repository contains only the code and data that
were actually used to produce the final design and the results reported in
the paper; earlier exploratory approaches and superseded test sessions have
been left out for clarity.

## Code

| File | Role | Paper section |
|---|---|---|
| `rcm5bar.py` | Closed-form kinematic model (`Linkage`, `solve()`, `required_q5()`) | II-A1 |
| `rcm_workspace.py` | Workspace evaluation: forward kinematics over a grid, Jacobian condition number, solid-angle/coverage/home-margin computation | II-A2 |
| `rcm_collision.py` | Functional requirements: access-cone check, self-collision check, home axis | II-A3 |
| `hole_search.py` | Shared design-space bounds and fixed evaluation settings used by the search | II-A4 |
| `hole_home_check.py` | Home margin (`home_cone_deg`) and related home-anchored metrics | II-A2, II-A4 |
| `search_v2.py` | The design-space search: phase A (random draws) + phases B1-B4 (Pareto perturbation) that produced the final design | II-A4 |
| `rcm_view.py` | Selects the final design from the search log (`best_designs()`, sorted on home margin) | II-A4 |
| `rcm_symbolic.py` | Independent symbolic verification of the closed-form solver | II-B |
| `bench_run.py` | All four experimental validation tests: noise floor, backlash/drift, single-axis sweep, two-dimensional grid test | II-C3, II-C4 |
| `bench_sensor_check.py` | Diagnostic script behind the quaternion-validation thresholds (2% unit-length tolerance, 3° consistency tolerance) used in `bench_run.py` | II-C4c |
| `rcm_control.py` | Interactive viewer/control tool used during design exploration and bench operation | II-A4, II-C |
| `rcm_to_step.py`, `rcm_to_dxf.py` | Export the selected design's geometry to STEP/DXF for physical realization | II-C1 |
| `bench_arduino/bench_arduino.ino` | Arduino firmware driving the two NEMA-11 steppers and reading the BNO055 sensor | II-C2 |

## Data

| File | Corresponds to |
|---|---|
| `results_v2/search_v2.csv` | Full design-space search log (14,800 evaluated candidates); the final design is the highest-`home_cone_deg` row | Table I, Table II |
| `bench_noisefloor.csv` | Sensor noise floor test (n=30) | Results III-C, std 0.0034°/max 0.0101° |
| `bench_roundtrip_newmotor.csv` | Backlash/drift round-trip test, **Run 3** | Table III/IV (Run 3 row) |
| `bench_axissweep_eindtest_runC.csv` + `_calib.csv` | Single-axis sweep test (50 points) + its mounting-orientation calibration set | Figure 6, Results III-C |
| `bench_grid_eindtest_run1.csv` + `_calib.csv` | Two-dimensional grid test (101 points) + its calibration set | Figure 7, Figure 8, Results III-C |

**Note on missing data**: the raw round-trip logs for backlash/drift **Run 1**
and **Run 2** were lost after the measurements were taken and could not be
recovered. Only Run 3's raw log is included here; the summary values for all
three runs (as measured at the time) are reported in Table III and Table IV
of the paper.

## Reproducing key numbers

With the dependencies installed (numpy, matplotlib; `bench_run.py` additionally
needs `pyserial` for live hardware use, not needed to re-analyse existing CSVs):

```bash
# Final design parameters and home margin (Table I/II)
python3 -c "
import csv
with open('results_v2/search_v2.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['feasible']=='True']
best = max(rows, key=lambda r: float(r['home_cone_deg']))
print(best)
"

# Noise floor / backlash-drift / single-axis sweep / grid test analysis
python3 bench_run.py --csv bench_noisefloor.csv          # noise floor
python3 bench_run.py --csv bench_roundtrip_newmotor.csv  # backlash/drift (Run 3)
python3 bench_run.py --csv bench_axissweep_eindtest_runC.csv  # single-axis sweep
python3 bench_run.py --csv bench_grid_eindtest_run1.csv       # grid test
```

(Exact CLI flags may need `--help` on `bench_run.py`, since several of its
analysis modes share the same entry point and auto-detect from the CSV's
`kind` column.)
