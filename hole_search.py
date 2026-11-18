"""hole_search.py -- Step 3: derivative-free search for 5-bar RCM designs with
a small workspace hole AND a large usable region around home.

Two objectives are kept separate on purpose (see the trap in the task brief:
minimising the hole alone drives the workspace to a thin band). Nothing is
collapsed to a weighted sum here; the CSV keeps every metric and the Pareto
front is extracted afterwards by hole_report.py.

Method: random search followed by rounds of Pareto-parent perturbation.
Both are derivative free, which is what a piecewise-constant objective needs.

Settings held fixed for the entire search:
    n_input=301, bin_deg=2.0, kappa_max=8.0, branch=+1,
    home_input=(0,0), full motor range, h=1.0 (metric is scale invariant)

Every result is appended to the CSV as it arrives, so a crash loses nothing.

    python hole_search.py [--phase-a N] [--rounds R] [--per-round N]
                          [--workers W] [--csv PATH] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from rcm5bar import Linkage
from rcm_workspace import workspace, hole_solid_angle

# ---------------------------------------------------------------- fixed
EVAL_KW = dict(n_input=301, bin_deg=2.0, kappa_max=8.0, branch=+1,
               home_input=(0.0, 0.0),
               Iq1_range=(-np.pi, np.pi), Iq2_range=(-np.pi, np.pi))

NAMES = ("l1", "l2", "l3", "l4", "l5", "l6", "q1", "q4")
LO = np.array([0.15] * 6 + [0.10] * 2)
HI = np.array([1.20] * 6 + [1.40] * 2)

FIELDS = ["phase", "idx", *NAMES, "feasible", "omega_dex", "hole_sr",
          "largest_hole_sr", "n_holes", "home_in_hole", "cone_deg",
          "rect_h_deg", "rect_v_deg", "rect_area_deg2", "equiv_cone_deg",
          "omega_raw", "gci", "kappa_median", "samples_per_bin",
          "frac_solved", "pinhole_fraction", "warning", "eval_s", "error"]

WORST = dict(omega_dex=0.0, hole_sr=4 * np.pi, largest_hole_sr=4 * np.pi,
             n_holes=-1, home_in_hole=True, cone_deg=0.0, rect_h_deg=0.0,
             rect_v_deg=0.0, rect_area_deg2=0.0, equiv_cone_deg=0.0,
             omega_raw=0.0, gci=0.0, kappa_median=np.inf,
             samples_per_bin=0.0, frac_solved=0.0, pinhole_fraction=0.0,
             warning="")


def evaluate(job):
    """(phase, idx, params) -> flat record dict. Never raises.

    Infeasible or non-closing designs are scored worst-case (zero workspace,
    whole-sphere hole) so they are dominated by every real design.
    """
    phase, idx, p = job
    rec = dict(phase=phase, idx=idx, **dict(zip(NAMES, [float(v) for v in p])))
    t0 = time.perf_counter()
    try:
        lk = Linkage(h=1.0, **dict(zip(NAMES, [float(v) for v in p])))
        r = workspace(lk, return_maps=True, **EVAL_KW)
        h = hole_solid_angle(r)
        rec.update(
            feasible=1, error="",
            omega_dex=r["omega_dex"], omega_raw=r["omega_raw"],
            hole_sr=h["hole_sr"], largest_hole_sr=h["largest_hole_sr"],
            n_holes=h["n_holes"], home_in_hole=int(h["home_in_hole"]),
            cone_deg=r["cone_deg"], equiv_cone_deg=r["equiv_cone_deg"],
            rect_h_deg=r["rect_deg"][0], rect_v_deg=r["rect_deg"][1],
            rect_area_deg2=4.0 * r["rect_deg"][0] * r["rect_deg"][1],
            gci=r["gci"], kappa_median=r["kappa_median"],
            samples_per_bin=r["samples_per_bin"],
            frac_solved=r["frac_solved"],
            pinhole_fraction=r["pinhole_fraction"],
            warning=r.get("WARNING", ""))
    except Exception as exc:                       # infeasible design or no home
        rec.update(feasible=0, error=f"{type(exc).__name__}: {exc}"[:160],
                   **WORST)
    rec["eval_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------- Pareto
def pareto_mask(obj):
    """Non-dominated mask for an (n, k) array of objectives, all MINIMISED."""
    obj = np.asarray(obj, float)
    n = len(obj)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def objectives(rows):
    """(hole small, cone large, rect area large) as three MINIMISED columns."""
    return np.array([[r["hole_sr"], -r["cone_deg"], -r["rect_area_deg2"]]
                     for r in rows], float)


def front_of(rows):
    live = [r for r in rows if r["feasible"] and r["omega_dex"] > 0.0]
    if not live:
        return []
    m = pareto_mask(objectives(live))
    return [r for r, k in zip(live, m) if k]


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", type=int, default=2400,
                    help="uniform random points (about half are feasible)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=500)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    ap.add_argument("--csv", default="hole_search.csv")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sigma", type=float, default=0.08,
                    help="perturbation width as a fraction of each range")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    span = HI - LO

    fresh = not os.path.exists(args.csv)
    fh = open(args.csv, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        wr.writeheader()
        fh.flush()

    rows = []
    t_start = time.perf_counter()

    def run(jobs, label):
        done = 0
        with Pool(args.workers) as pool:
            for rec in pool.imap_unordered(evaluate, jobs, chunksize=2):
                wr.writerow(rec)
                rows.append(rec)
                done += 1
                if done % 100 == 0:
                    fh.flush()
                    feas = sum(r["feasible"] for r in rows)
                    el = time.perf_counter() - t_start
                    print(f"  [{label}] {done}/{len(jobs)}  "
                          f"feasible so far {feas}/{len(rows)}  "
                          f"elapsed {el/60:.1f} min", flush=True)
        fh.flush()

    # --- phase A: uniform random over the box ---------------------------
    print(f"phase A: {args.phase_a} uniform random points on {args.workers} "
          f"workers", flush=True)
    jobs = [("A", i, rng.uniform(LO, HI)) for i in range(args.phase_a)]
    run(jobs, "A")

    fr = front_of(rows)
    print(f"phase A done: {sum(r['feasible'] for r in rows)} feasible, "
          f"Pareto front {len(fr)} designs", flush=True)

    # --- phase B: perturb Pareto parents --------------------------------
    for rnd in range(args.rounds):
        fr = front_of(rows)
        if not fr:
            print("no feasible designs to refine; stopping", flush=True)
            break
        parents = [np.array([r[n] for n in NAMES], float) for r in fr]
        jobs = []
        for i in range(args.per_round):
            p = parents[rng.integers(len(parents))]
            q = p + rng.normal(0.0, args.sigma, 8) * span
            jobs.append((f"B{rnd+1}", i, np.clip(q, LO, HI)))
        print(f"phase B{rnd+1}: {args.per_round} perturbations of "
              f"{len(parents)} Pareto parents (sigma={args.sigma})", flush=True)
        run(jobs, f"B{rnd+1}")

    fh.close()

    feas = sum(r["feasible"] for r in rows)
    el = time.perf_counter() - t_start
    print(f"\n{len(rows)} points attempted, {feas} feasible evaluations, "
          f"{el/60:.1f} min wall", flush=True)
    fr = front_of(rows)
    print(f"Pareto front: {len(fr)} designs -> {args.csv}", flush=True)
    fr.sort(key=lambda r: r["hole_sr"])
    print(f"{'hole':>7} {'cone':>6} {'rect h x v':>12} {'omega':>7}")
    for r in fr[:20]:
        print(f"{r['hole_sr']:>7.3f} {r['cone_deg']:>6.2f} "
              f"{r['rect_h_deg']:>5.1f} x{r['rect_v_deg']:>5.1f} "
              f"{r['omega_dex']:>7.3f}")


if __name__ == "__main__":
    main()
