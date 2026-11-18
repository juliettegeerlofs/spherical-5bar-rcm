"""search_v2.py -- the search redone with the eye gripped from one side.

What is different from the first run (results_v1/)
--------------------------------------------------
1. The eye is GRIPPED, not entered. A rigid attachment arm runs from the plate
   down the gaze axis to the globe, and it is a load-bearing member: it is
   checked for collisions like any other bar.
2. The accessible side is fixed to the head, and the direction the mechanism
   holds the eye in at home IS straight ahead. So the access axis is each
   design's own home gaze direction, not -N.z, which is what the first
   interference check wrongly assumed.
3. Interference masks poses BEFORE the connected component is taken, so a
   forbidden band splits the workspace instead of being counted in it.
4. The objective is home-anchored from the start: (hole small, cone around
   home large, box around home large). The first run learned the hard way
   that cone_deg alone is gamed by workspaces that wrap the sphere.

Torsion stays out of scope, as agreed.

Cheap rejects first: most parameter sets die at the home pose, so the home
pose is checked with solve() (microseconds) before any grid is touched.

    python search_v2.py [--phase-a 12000] [--rounds 3] [--per-round 600]

Writes results_v2/search_v2.csv as it goes.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from rcm5bar import Linkage, solve
from rcm_workspace import workspace, hole_solid_angle
from rcm_collision import check_pose, home_axis_of, pose_mask, grid_masks
from hole_home_check import home_metrics_from_result
from hole_search import NAMES, LO, HI, EVAL_KW, pareto_mask

# fixed for the whole run, as ratios of h so scale invariance survives
CONSTRAINTS = dict(access_deg=90.0, link_diameter=0.03, eye_radius=0.08,
                   with_arm=True)

FIELDS = ["phase", "idx", *NAMES, "feasible", "reject",
          "omega_dex", "omega_unmasked", "retained_frac", "hole_sr", "n_holes",
          "home_in_hole", "cone_deg", "home_cone_deg", "cone_off_home_deg",
          "rect_h_deg", "rect_v_deg", "rect_area_deg2", "equiv_cone_deg",
          "omega_within_30deg", "gci", "kappa_median", "samples_per_bin",
          "frac_solved", "planar_ok", "max_theta_v_deg", "span_h_bins_deg",
          "home_axis_x", "home_axis_y", "home_axis_z", "worst_access_deg",
          "min_self_gap", "Iq1_lo", "Iq1_hi", "Iq2_lo", "Iq2_hi",
          "access_deg", "link_diameter", "eye_radius", "eval_s", "error"]

WORST = {k: 0.0 for k in FIELDS if k not in
         ("phase", "idx", *NAMES, "feasible", "reject", "error", "eval_s",
          "access_deg", "link_diameter", "eye_radius")}
WORST.update(hole_sr=4 * np.pi, kappa_median=np.inf, n_holes=-1,
             home_in_hole=1, planar_ok=0, worst_access_deg=180.0)


def evaluate(job):
    """One design under the new constraints. Never raises."""
    phase, idx, p = job
    rec = dict(phase=phase, idx=idx,
               **dict(zip(NAMES, [float(v) for v in p])),
               **CONSTRAINTS, **WORST)
    rec.pop("with_arm", None)
    rec.update(feasible=0, reject="", error="")
    t0 = time.perf_counter()
    try:
        lk = Linkage(h=1.0, **dict(zip(NAMES, [float(v) for v in p])))
    except Exception as exc:
        rec.update(reject="geometry", error=f"{type(exc).__name__}"[:60])
        rec["eval_s"] = time.perf_counter() - t0
        return rec

    # --- cheap rejects, before any grid ------------------------------
    try:
        axis = home_axis_of(lk, EVAL_KW["branch"], EVAL_KW["home_input"])
        s = solve(lk, *EVAL_KW["home_input"], branch=EVAL_KW["branch"])
    except Exception as exc:
        rec.update(reject="home does not close",
                   error=f"{type(exc).__name__}"[:60])
        rec["eval_s"] = time.perf_counter() - t0
        return rec

    hit = check_pose(s, access_axis=axis, scale=lk.h, **CONSTRAINTS)
    if not hit["ok"]:
        rec.update(reject="home interferes",
                   worst_access_deg=hit["access_worst_deg"],
                   min_self_gap=hit["self_gap"],
                   home_axis_x=axis[0], home_axis_y=axis[1],
                   home_axis_z=axis[2])
        rec["eval_s"] = time.perf_counter() - t0
        return rec

    # --- the expensive part ------------------------------------------
    try:
        mask = pose_mask(lk, EVAL_KW["branch"], access_axis=axis,
                         **CONSTRAINTS)
        r = workspace(lk, return_maps=True, pose_mask=mask, **EVAL_KW)
        r0 = workspace(lk, **EVAL_KW)          # same design, constraint off
        if r["omega_dex"] <= 0.0 or not r["maps"]["comp_dex"].any():
            # home survives on its own but its whole neighbourhood does not
            rec.update(reject="workspace empty once masked",
                       omega_unmasked=r0["omega_dex"], retained_frac=0.0,
                       home_axis_x=axis[0], home_axis_y=axis[1],
                       home_axis_z=axis[2])
            rec["eval_s"] = time.perf_counter() - t0
            return rec
        h = hole_solid_angle(r)
        d = home_metrics_from_result(r)
        _, _, _, worst, gap = grid_masks(lk, r["maps"]["Iq1"], r["maps"]["Iq2"],
                                         EVAL_KW["branch"], axis,
                                         **CONSTRAINTS)
        comp = r["maps"]["comp_dex"]
        rec.update(
            feasible=1, reject="",
            omega_dex=r["omega_dex"], omega_unmasked=r0["omega_dex"],
            retained_frac=(r["omega_dex"] / r0["omega_dex"]
                           if r0["omega_dex"] > 0 else 0.0),
            hole_sr=h["hole_sr"], n_holes=h["n_holes"],
            home_in_hole=int(h["home_in_hole"]),
            planar_ok=int(h["planar_ok"]),
            max_theta_v_deg=h["max_theta_v_deg"],
            span_h_bins_deg=h["span_h_bins_deg"],
            cone_deg=r["cone_deg"], equiv_cone_deg=r["equiv_cone_deg"],
            home_cone_deg=d["home_cone_deg"],
            cone_off_home_deg=d["cone_off_home_deg"],
            omega_within_30deg=d["omega_within_30deg"],
            rect_h_deg=r["rect_deg"][0], rect_v_deg=r["rect_deg"][1],
            rect_area_deg2=4.0 * r["rect_deg"][0] * r["rect_deg"][1],
            gci=r["gci"], kappa_median=r["kappa_median"],
            samples_per_bin=r["samples_per_bin"],
            frac_solved=r["frac_solved"],
            home_axis_x=axis[0], home_axis_y=axis[1], home_axis_z=axis[2],
            worst_access_deg=float(worst[comp].max()) if comp.any() else 180.0,
            min_self_gap=float(gap[comp].min()) if comp.any() else 0.0,
            Iq1_lo=d["Iq1_span"][0], Iq1_hi=d["Iq1_span"][1],
            Iq2_lo=d["Iq2_span"][0], Iq2_hi=d["Iq2_span"][1])
        if r["omega_dex"] <= 0.0:
            rec.update(feasible=0, reject="workspace empty once masked")
    except Exception as exc:
        rec.update(reject="evaluation failed",
                   error=f"{type(exc).__name__}: {exc}"[:120])
    rec["eval_s"] = time.perf_counter() - t0
    return rec


def front_of(rows):
    """Pareto front over (hole small, home cone large, box large)."""
    live = [r for r in rows if r["feasible"] and r["omega_dex"] > 0
            and r["planar_ok"]]
    if not live:
        return []
    obj = np.array([[r["hole_sr"], -r["home_cone_deg"], -r["rect_area_deg2"]]
                    for r in live], float)
    return [r for r, k in zip(live, pareto_mask(obj)) if k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", type=int, default=12000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=600)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results_v2")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "search_v2.csv")
    rng = np.random.default_rng(args.seed)
    span = HI - LO

    fresh = not os.path.exists(path)
    fh = open(path, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if fresh:
        wr.writeheader()

    rows, t0 = [], time.perf_counter()

    def run(jobs, label):
        done = 0
        with Pool(args.workers) as pool:
            for rec in pool.imap_unordered(evaluate, jobs, chunksize=8):
                wr.writerow(rec)
                rows.append(rec)
                done += 1
                if done % 500 == 0:
                    fh.flush()
                    good = sum(r["feasible"] for r in rows)
                    print(f"  [{label}] {done}/{len(jobs)}  usable {good}  "
                          f"elapsed {(time.perf_counter()-t0)/60:.1f} min",
                          flush=True)
        fh.flush()

    print(f"phase A: {args.phase_a} random draws on {args.workers} workers",
          flush=True)
    run([("A", i, rng.uniform(LO, HI)) for i in range(args.phase_a)], "A")
    fr = front_of(rows)
    print(f"phase A: {sum(r['feasible'] for r in rows)} usable designs, "
          f"front {len(fr)}", flush=True)

    for rnd in range(args.rounds):
        fr = front_of(rows)
        if not fr:
            print("nothing feasible to refine", flush=True)
            break
        good = [r for r in rows if r["feasible"] and r["planar_ok"]]
        extra = sorted(good, key=lambda r: -r["home_cone_deg"])[:15]
        parents = fr + [r for r in extra if r not in fr]
        pv = [np.array([r[n] for n in NAMES], float) for r in parents]
        jobs = []
        for i in range(args.per_round):
            q = pv[rng.integers(len(pv))] + rng.normal(0, args.sigma, 8) * span
            jobs.append((f"B{rnd+1}", i, np.clip(q, LO, HI)))
        print(f"phase B{rnd+1}: {args.per_round} perturbations of "
              f"{len(pv)} parents", flush=True)
        run(jobs, f"B{rnd+1}")
        fr = front_of(rows)
        best = max((r["home_cone_deg"] for r in fr), default=0.0)
        print(f"  front {len(fr)}, best home cone {best:.1f} deg", flush=True)

    fh.close()
    n_ok = sum(r["feasible"] for r in rows)
    print(f"\n{len(rows)} draws, {n_ok} usable, "
          f"{(time.perf_counter()-t0)/60:.1f} min -> {path}")
    why = {}
    for r in rows:
        why[r["reject"] or "usable"] = why.get(r["reject"] or "usable", 0) + 1
    for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<28} {v:6d}  ({100*v/len(rows):.1f}%)")

    fr = front_of(rows)
    fr.sort(key=lambda r: -r["home_cone_deg"])
    print(f"\nPareto front ({len(fr)}):")
    print(f"{'hole':>7} {'homecone':>9} {'cone':>7} {'box hxv':>12} "
          f"{'omega':>7} {'kept':>6}")
    for r in fr[:15]:
        print(f"{r['hole_sr']:>7.3f} {r['home_cone_deg']:>9.1f} "
              f"{r['cone_deg']:>7.1f} {r['rect_h_deg']:>5.0f} x"
              f"{r['rect_v_deg']:>5.0f} {r['omega_dex']:>7.3f} "
              f"{100*r['retained_frac']:>5.0f}%")


if __name__ == "__main__":
    main()
