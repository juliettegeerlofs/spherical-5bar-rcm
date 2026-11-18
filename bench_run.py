"""bench_run.py -- drive the physical RCM mechanism (steppers via
bench_arduino.ino, for Iq1/Iq2) through a set of test poses, read the
GY-521's measured orientation for each, and compare directly against what
the model (rcm5bar.solve(), the same Fick/home-anchored angles
rcm_control.py shows) predicts for the same input.

Three safety layers on top of the plain "sample the usable region" plan:

1. Iq1 mechanical limits. The base clamps Iq1 to a physical range (Iq2 is
   free to fully rotate). --calibrate-iq1 steps Iq1 a little at a time,
   asking you to confirm after every step that it is still moving freely,
   and feeds the discovered range straight into workspace()'s own
   Iq1_range parameter -- so no sampled pose ever asks for more than the
   base mechanically allows.

2. Iq2-vs-Iq1 interference. Link 2 (P2-P3) can hit the stepper motor
   mount when Iq2 sits too far negative relative to wherever Iq1 is --
   not a constraint the kinematics model knows about at all, since it's
   a collision with external hardware, not between two of the mechanism's
   own links. --calibrate-iq2-floor probes that boundary at a few Iq1
   values across the (already-known) Iq1 range, the same step-and-confirm
   way as --calibrate-iq1, and the discovered per-Iq1 floor is masked
   into the same usable-region grid as everything else.

3. Path checking between poses. The usable region (comp_dex, now also
   carrying the Iq2-floor cut) is not convex -- two individually-valid
   poses can have a straight-line motor path between them that cuts
   through the unusable region. Every move is checked, at a fine
   resolution, against the same usable-region grid before it is sent,
   not just its two endpoints.

    pip install pyserial
    python bench_run.py --port /dev/tty.usbmodemXXXX --calibrate-iq1 \
                        --calibrate-iq2-floor
    python bench_run.py --port /dev/tty.usbmodemXXXX \
                        --iq1-min -42 --iq1-max 38 \
                        --iq2-floor "-42.0,-30.0;0.0,-45.0;38.0,-20.0"
    python bench_run.py --check-only --csv results_v2/bench_run.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serial

from rcm5bar import Linkage, solve
from rcm_view import (best_designs, params_of, home_frame, resolve_scale,
                      SOURCES_V1, SOURCES_V2, SOURCES_EYE)
from hole_search import NAMES, EVAL_KW
from rcm_collision import pose_mask, home_axis_of
from rcm_workspace import workspace

FIELDS = ["idx", "kind", "Iq1_deg", "Iq2_deg", "pred_theta_h_deg",
          "pred_theta_v_deg", "measured_theta_h_deg", "measured_theta_v_deg",
          "corrected_theta_h_deg", "corrected_theta_v_deg", "note"]


def quat_to_R(q):
    """Rotation matrix (body -> world) for a unit quaternion [w, x, y, z],
    same convention as the BNO055 library's Quaternion::toMatrix() -- so
    this matches, cell for cell, what the sensor itself considers its
    orientation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def theta_hv_from_quats(q_home, q_cur):
    """Same theta_h/theta_v convention as predict_angles(): rotate the
    tool-axis (fixed in the sensor body at (0,0,1)) through the FULL
    relative rotation from home to current, then project -- not a
    per-channel Euler-angle subtraction, which is only valid for a pure
    single-axis tip and is what originally made the bench look ~90 deg off
    from the model even on grid poses where the mechanism moves both
    joints together.

    t[0]/t[1] (the sensor's own body x/y) are swapped before projecting --
    confirmed empirically (bench_sweep_test5.csv): fitting the best possible
    ORTHOGONAL transform between predicted and measured directions wanted a
    reflection (det=-1), not a rotation, and the specific reflection that
    reduces the leftover compensating rotation to a few degrees (ordinary
    mounting slop) rather than ~90 deg is an x/y swap -- i.e. the BNO055 is
    mounted rotated+mirrored relative to what this formula originally
    assumed, not just rotated. Without this, no single mounting-rotation
    correction could ever fit both axes of a combined-motion grid at once,
    which was the real explanation behind the long-standing Iq1-vs-Iq2
    asymmetry this project spent a lot of time chasing as a mechanical
    defect.

    t[1] is ALSO negated -- this is a second, separate mounting reflection,
    confirmed the same way (bench_sweep_nieuw.csv) on the rebuilt, sturdier
    mechanism: raw error ~51 deg; the best rotation-only fit still needed
    ~130 deg and left ~20 deg residual, while allowing a reflection again
    dropped it to ~2.8 deg, and negating t[1] alone already accounts for
    almost all of that (~4.7 deg). The physical mount changed between the
    two builds -- the BNO055 sits mirrored differently on the rebuilt part
    than it did on the original one -- so this correction is specific to
    the current, rebuilt mechanism, not a universal property of the sensor
    itself."""
    R_home = quat_to_R(q_home)
    R_cur = quat_to_R(q_cur)
    t = (R_home.T @ R_cur)[:, 2]
    th_h = np.degrees(np.arctan2(-t[1], t[2]))
    th_v = np.degrees(np.arcsin(np.clip(t[0], -1.0, 1.0)))
    return th_h, th_v


# ----------------------------------------------------------------------
# per-session mounting correction (see --self-calibrate)
# ----------------------------------------------------------------------
# bench_sweep_nieuw.csv / _nieuw2 / _nieuw3 (three independent re-zeroings of
# the rebuilt mechanism) showed that the best-fit correction on top of the
# structural axis negation above is NOT stable session to session: run 1 and
# run 2 happened to agree within 1.2 deg, but run 3 needed a correction
# 13.6-13.7 deg away from both -- and cross-validation confirmed it, a run's
# OWN fit stays around 2.5-4.5 deg while another run's fit applied to it
# jumps to ~9 deg. So the small residual beyond the axis negation is tied to
# how precisely Iq1=Iq2=0 was eyeballed at zero time THIS session, not a
# fixed mounting property -- it cannot be hardcoded once, only fitted fresh
# from that same session's own sweep data and applied within that session.

def _pred_vec(th_h_deg, th_v_deg):
    h, v = np.radians(th_h_deg), np.radians(th_v_deg)
    return np.array([np.cos(v) * np.sin(h), np.sin(v), np.cos(v) * np.cos(h)])


def _meas_vec(th_h_deg, th_v_deg):
    """Inverse of theta_hv_from_quats's (th_h, th_v) <- t mapping, i.e. the
    vector that theta_hv_from_quats would have been given to produce this
    (th_h, th_v) pair. Mirrors _pred_vec's component order but with t[1]
    negated, matching the atan2(-t[1], t[2]) above."""
    h, v = np.radians(th_h_deg), np.radians(th_v_deg)
    return np.array([np.sin(v), -np.cos(v) * np.sin(h), np.cos(v) * np.cos(h)])


def fit_mounting_correction(rows):
    """Best orthogonal correction R (R @ measured_vec ~ predicted_vec) from a
    set of rows that already carry pred_theta_h/v_deg and
    measured_theta_h/v_deg -- typically this session's own axis-sweep.
    Kabsch/Procrustes fit, reflection allowed on purpose: the structural
    axis negation in theta_hv_from_quats already folds a reflection in, so
    a further reflection should not normally be needed here, but allowing
    it costs nothing and would be an immediate red flag (det=-1 again) if
    something upstream changed.

    Returns (R, mean_deg, max_deg) -- the fitted matrix and the residual
    angular error it leaves on this same data, so the caller can print how
    trustworthy this session's calibration actually is before relying on it.
    """
    rows = [r for r in rows if r["measured_theta_h_deg"] not in ("", None)]
    if len(rows) < 3:
        raise ValueError("te weinig sweep-punten om een correctie op te "
                         "fitten (minstens 3 nodig)")
    P = np.array([_pred_vec(float(r["pred_theta_h_deg"]),
                            float(r["pred_theta_v_deg"])) for r in rows])
    M = np.array([_meas_vec(float(r["measured_theta_h_deg"]),
                            float(r["measured_theta_v_deg"])) for r in rows])
    H = M.T @ P
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    corrected = (R @ M.T).T
    err = np.degrees(np.arccos(np.clip(np.sum(corrected * P, axis=1), -1.0, 1.0)))
    return R, float(err.mean()), float(err.max())


def apply_mounting_correction(R, th_h_deg, th_v_deg):
    """Apply a fitted session correction to one raw (theta_h, theta_v).

    R maps a _meas_vec into _pred_vec's basis (that is what fit_mounting_
    correction solved for), so the result must be decoded with _pred_vec's
    component order (index 0 = h-component, index 1 = v-component) -- NOT
    _meas_vec's (that would silently swap h/v again)."""
    t = _meas_vec(th_h_deg, th_v_deg)
    tc = R @ t
    th_h_c = np.degrees(np.arctan2(tc[0], tc[2]))
    th_v_c = np.degrees(np.arcsin(np.clip(tc[1], -1.0, 1.0)))
    return th_h_c, th_v_c


def load_design(args):
    row = None
    if args.params:
        p = dict(zip(NAMES, [float(v) for v in args.params.split(",")]))
    else:
        src = {"v1": SOURCES_V1, "v2": SOURCES_V2,
              "eye": SOURCES_EYE}[args.results]
        row = best_designs(args.sort, args.design + 1, paths=src)[args.design]
        p = params_of(row)
    args.with_arm = args.results != "eye"
    resolve_scale(args, args.results, row, default_h=0.150)
    scaled = {n: (v * args.h if n.startswith("l") else v) for n, v in p.items()}
    return Linkage(h=args.h, **scaled)


def predict_angles(lk, args, Iq1, Iq2, M):
    """The same theta_h/theta_v rcm_control.py's status line shows: Fick
    angles from home, home-anchored (not solve()'s default reference)."""
    s = solve(lk, Iq1, Iq2, branch=args.branch)
    t = M.T @ s["tool_axis"]
    th_h = np.degrees(np.arctan2(t[0], t[2]))
    th_v = np.degrees(np.arcsin(np.clip(t[1], -1.0, 1.0)))
    return th_h, th_v


# ----------------------------------------------------------------------
# usable-region grid: shared between pose generation and path safety checks
# ----------------------------------------------------------------------
def iq2_floor_mask(m, iq2_floor_pts):
    """pts: [(Iq1_deg, Iq2_min_deg), ...] from calibrate_iq2_floor() or
    --iq2-floor. Piecewise-linear floor on Iq2 as a function of Iq1 --
    deliberately not assumed to be a fixed offset/slope, since the real
    interference geometry (link 2 vs the motor mount) has no reason to be
    exactly linear; probing a few points and interpolating finds whatever
    shape it actually has. np.interp clamps (flat-extrapolates) outside the
    calibrated Iq1 range, which is the safe direction here since poses are
    only ever sampled within that same Iq1 range anyway."""
    pts = sorted(iq2_floor_pts)
    xs = np.radians([p[0] for p in pts])
    ys = np.radians([p[1] for p in pts])
    floor = np.interp(m["Iq1"], xs, ys)
    return m["Iq2"] >= floor


def build_grid(lk, args, iq1_range_deg=None, iq2_floor_pts=None, iq2_max_deg=None):
    """--iq1-margin/--iq2-margin shrink the pose-SELECTION region a few
    degrees inside the calibrated limits -- found necessary because
    build_axis_sweep_poses() always starts its Iq2 sweep at the most
    negative Iq2 the comp mask allows for Iq1 near 0, which landed only
    0.24 deg above the calibrated Iq2 floor: the very first move of every
    --self-calibrate session was going almost exactly to the motor-mount
    collision boundary, from a cold start. A margin here does not touch
    the calibration itself (--iq2-floor / --iq1-min / --iq1-max stay
    exactly as calibrated) -- it only keeps chosen test poses a bit clear
    of it."""
    axis = home_axis_of(lk, args.branch, EVAL_KW["home_input"])
    mask = pose_mask(lk, args.branch, access_axis=axis,
                     access_deg=args.access_deg,
                     link_diameter=args.link_diameter,
                     eye_radius=args.eye_radius, with_arm=args.with_arm)
    kw = dict(EVAL_KW, n_input=args.n_input)
    if iq1_range_deg is not None:
        lo, hi = iq1_range_deg[0] + args.iq1_margin, iq1_range_deg[1] - args.iq1_margin
        if lo >= hi:
            raise SystemExit(f"--iq1-margin {args.iq1_margin:.1f} deg is te groot "
                             f"voor het Iq1-bereik [{iq1_range_deg[0]:.1f}, "
                             f"{iq1_range_deg[1]:.1f}]")
        kw["Iq1_range"] = (np.radians(lo), np.radians(hi))
        print(f"(workspace beperkt tot Iq1 in [{iq1_range_deg[0]:+.1f}, "
              f"{iq1_range_deg[1]:+.1f}] deg, met {args.iq1_margin:.1f} deg "
              f"marge dus standen in [{lo:+.1f}, {hi:+.1f}] deg)")
    r = workspace(lk, return_maps=True, pose_mask=mask, **kw)
    m = r["maps"]
    M = home_frame(r["home_axis"])
    comp = m["comp_dex"]
    if iq2_floor_pts is not None:
        margined = [(a, b + args.iq2_margin) for a, b in iq2_floor_pts]
        comp = comp & iq2_floor_mask(m, margined)
        print("(workspace extra beperkt tot Iq2 boven de gekalibreerde "
              f"grens t.o.v. Iq1 -- motorstandaard-botsing -- plus "
              f"{args.iq2_margin:.1f} deg marge)")
    if iq2_max_deg is not None:
        hi = iq2_max_deg - args.iq2_margin
        comp = comp & (m["Iq2"] <= np.radians(hi))
        print(f"(workspace extra beperkt tot Iq2 <= {iq2_max_deg:+.1f} deg "
              f"min {args.iq2_margin:.1f} deg marge, dus Iq2 <= {hi:+.1f} deg)")
    if not comp.any():
        raise SystemExit("geen bruikbare workspace onder deze constraints -- "
                         "check --access-deg / --link-diameter / "
                         "--eye-radius / --iq1-min / --iq1-max / "
                         "--iq2-floor / --iq1-margin / --iq2-margin tegen "
                         "je build")
    return m, comp, M


def grid_ok(m, comp, iq1, iq2):
    """Nearest-cell lookup on the usable-region grid for an arbitrary
    (Iq1, Iq2) [rad], not just the sampled grid points themselves."""
    a1 = m["Iq1"][:, 0]
    a2 = m["Iq2"][0, :]
    i = int(np.clip(np.searchsorted(a1, iq1), 0, len(a1) - 1))
    if i > 0 and abs(a1[i - 1] - iq1) < abs(a1[i] - iq1):
        i -= 1
    j = int(np.clip(np.searchsorted(a2, iq2), 0, len(a2) - 1))
    if j > 0 and abs(a2[j - 1] - iq2) < abs(a2[j] - iq2):
        j -= 1
    return bool(comp[i, j])


def path_ok(m, comp, iq1_a, iq2_a, iq1_b, iq2_b, n=25):
    """The usable region is not convex -- check n points along the
    straight-line motor move, not just its two endpoints, so a move
    between two individually-valid poses can't cut through the unusable
    region in between."""
    for t in np.linspace(0.0, 1.0, n):
        iq1 = iq1_a + t * (iq1_b - iq1_a)
        iq2 = iq2_a + t * (iq2_b - iq2_a)
        if not grid_ok(m, comp, iq1, iq2):
            return False
    return True


def build_poses(args, m, comp, M, lk):
    # evenly spaced by azimuth of the PREDICTED GAZE DIRECTION, not by angle
    # in Iq1/Iq2 joint space and not by raw position in the nonzero() list.
    # The first fix here (evenly-spaced by joint-space angle instead of by
    # raw index) was still wrong in a subtler way: the map from (Iq1, Iq2)
    # to direction is strongly nonlinear, so a joint-space angle that is
    # evenly spaced is NOT evenly spaced in the direction it produces --
    # some regions compress tens of degrees of joint angle into a few
    # degrees of direction-azimuth (producing a tight cluster on the final
    # theta_h/theta_v coverage plot), while others expand a few degrees of
    # joint angle into a wide swath of direction-azimuth (producing an
    # empty-looking gap there), exactly what the plot showed. Sampling
    # evenly in the direction-azimuth itself (via the same tool-axis
    # projection predict_angles() uses) fixes this at the source.
    ii, jj = np.nonzero(comp)
    a1_all, a2_all = m["Iq1"][ii, jj], m["Iq2"][ii, jj]
    t_all = m["axes"][ii, jj] @ M
    t_all = t_all / np.linalg.norm(t_all, axis=-1, keepdims=True)
    ang = np.arctan2(t_all[..., 1], t_all[..., 0])
    order = np.argsort(ang)
    ii, jj, ang = ii[order], jj[order], ang[order]
    targets = np.linspace(ang[0], ang[-1], args.n_pose)
    nearest = np.clip(np.searchsorted(ang, targets), 0, len(ang) - 1)
    nearest_lo = np.clip(nearest - 1, 0, len(ang) - 1)
    use_lo = np.abs(ang[nearest_lo] - targets) < np.abs(ang[nearest] - targets)
    keep = np.unique(np.where(use_lo, nearest_lo, nearest))

    # The direction-space selection above picks WHICH points to include
    # (for even coverage on the final theta_h/theta_v plot), but visiting
    # them in that same direction-sorted order can zigzag wildly in
    # Iq1/Iq2 -- confirmed directly: a stretch of this grid repeatedly
    # jumped 30-38 deg back and forth between two distant Iq1 values
    # several times in a row, and the readings from that point on carried
    # a lasting ~20-25 deg theta_h offset (same lost-sync signature as the
    # calibration-sweep jumps earlier -- rapid direction reversals are a
    # separately-documented failure mode for this driver, see the removed
    # DITHER_* code in bench_arduino.ino; subdivide() in drive_and_measure
    # bounds single-jump SIZE but does nothing about reversal FREQUENCY).
    # Same fix as build_calibration_poses(): keep the selection, re-order
    # the VISIT sequence by greedy nearest-neighbour chaining in
    # (Iq1,Iq2) space starting from home, so consecutive moves stay small
    # and don't zigzag.
    cand = [(float(m["Iq1"][ii[k], jj[k]]), float(m["Iq2"][ii[k], jj[k]])) for k in keep]
    cur_a, cur_b = 0.0, 0.0
    remaining = cand[:]
    ordered = []
    while remaining:
        dists = [(a - cur_a) ** 2 + (b - cur_b) ** 2 for a, b in remaining]
        j = int(np.argmin(dists))
        cur_a, cur_b = remaining.pop(j)
        ordered.append((cur_a, cur_b))
    poses = [(0.0, 0.0)] + ordered

    rows, idx = [], 0
    for a, b in poses:
        th_h, th_v = predict_angles(lk, args, a, b, M)
        rows.append(dict(idx=idx, kind="grid", Iq1_deg=np.degrees(a),
                         Iq2_deg=np.degrees(b), pred_theta_h_deg=th_h,
                         pred_theta_v_deg=th_v, measured_theta_h_deg="",
                         measured_theta_v_deg="", note=""))
        idx += 1
    # 3 repeat points, same pattern as before -- but visiting them as a
    # straight A->B->C->A->B->C->... cycle, n_repeat times, repeats a
    # 17-39 deg back-and-forth MANY times in a row: structurally the same
    # kind of repeated direction-reversal pattern that corrupted data
    # around pose 69-73 in the main grid (confirmed there: a lasting
    # ~20-25 deg theta_h offset starting right at a cluster of 30+ deg
    # reversals). Alternating direction each round (A->B->C, then
    # C->B->A, then A->B->C, ...) instead makes consecutive rounds share
    # their endpoint -- that boundary becomes a free, zero-distance
    # revisit instead of the single biggest jump (C back to A) repeating
    # every cycle, while still keeping the "approach this point afresh
    # each time" intent the repeat points are for.
    repeat_idx = (0, len(poses) // 3, 2 * len(poses) // 3)
    for rep in range(args.n_repeat):
        order = repeat_idx if rep % 2 == 0 else tuple(reversed(repeat_idx))
        for k in order:
            a, b = poses[k]
            th_h, th_v = predict_angles(lk, args, a, b, M)
            rows.append(dict(idx=idx, kind=f"repeat{rep+1}",
                             Iq1_deg=np.degrees(a), Iq2_deg=np.degrees(b),
                             pred_theta_h_deg=th_h, pred_theta_v_deg=th_v,
                             measured_theta_h_deg="", measured_theta_v_deg="",
                             note="approach from + then -"))
            idx += 1
    return rows


def build_axis_sweep_poses(args, m, comp, M, lk):
    """Vary ONE joint at a time (the other pinned at 0) instead of the 2D
    grid -- isolates a per-axis scale/calibration mismatch (which would
    show up as a straight-line slope != 1 on that axis alone) from a
    coupling/geometry error that only appears once both joints move
    together, and from backlash (which shows up as scatter even along a
    single straight sweep)."""
    a1 = m["Iq1"][:, 0]
    a2 = m["Iq2"][0, :]
    j0 = int(np.argmin(np.abs(a2)))
    i0 = int(np.argmin(np.abs(a1)))

    rows, idx = [], 0

    def add(a, b, kind):
        nonlocal idx
        th_h, th_v = predict_angles(lk, args, a, b, M)
        rows.append(dict(idx=idx, kind=kind, Iq1_deg=np.degrees(a),
                         Iq2_deg=np.degrees(b), pred_theta_h_deg=th_h,
                         pred_theta_v_deg=th_v, measured_theta_h_deg="",
                         measured_theta_v_deg="", note=""))
        idx += 1

    # sweep_iq2 first, right after the fresh Z zero -- so Iq1's "0" during
    # that sweep is as clean as possible, not possibly nudged by backlash
    # left over from already having swung Iq1 across its full range
    valid_j = np.nonzero(comp[i0, :])[0]
    for k in np.linspace(0, len(valid_j) - 1, args.n_sweep).astype(int):
        j = valid_j[k]
        add(a1[i0], a2[j], "sweep_iq2")

    valid_i = np.nonzero(comp[:, j0])[0]
    for k in np.linspace(0, len(valid_i) - 1, args.n_sweep).astype(int):
        i = valid_i[k]
        add(a1[i], a2[j0], "sweep_iq1")

    return rows


def build_round_trip_poses(args, m, comp, M, lk):
    """Same per-axis point set as build_axis_sweep_poses, but driven as a
    there-and-back traversal: most positive down to most negative, then
    immediately the exact same points again from most negative back up to
    most positive -- instead of one single pass. Matching each point's
    forward-visit measurement against its backward-visit measurement
    isolates backlash/hysteresis (a direction-dependent offset that should
    show up as a gap between the two passes at every point) from drift
    (the very first point measured is also the very last point measured,
    since the backward pass's final stop is the forward pass's starting
    point -- a free start-vs-end check spanning the whole round trip)."""
    a1 = m["Iq1"][:, 0]
    a2 = m["Iq2"][0, :]
    j0 = int(np.argmin(np.abs(a2)))
    i0 = int(np.argmin(np.abs(a1)))

    rows, idx = [], 0

    def add(a, b, kind):
        nonlocal idx
        th_h, th_v = predict_angles(lk, args, a, b, M)
        rows.append(dict(idx=idx, kind=kind, Iq1_deg=np.degrees(a),
                         Iq2_deg=np.degrees(b), pred_theta_h_deg=th_h,
                         pred_theta_v_deg=th_v, measured_theta_h_deg="",
                         measured_theta_v_deg="", note=""))
        idx += 1

    def round_trip(values_asc, fixed, axis, kind_prefix):
        for a in values_asc[::-1]:   # most positive first, down to most negative
            iq1, iq2 = (a, fixed) if axis == "iq1" else (fixed, a)
            add(iq1, iq2, kind_prefix + "_fwd")
        for a in values_asc:         # most negative back up to most positive
            iq1, iq2 = (a, fixed) if axis == "iq1" else (fixed, a)
            add(iq1, iq2, kind_prefix + "_bwd")
        # third leg: keep going past the round trip's own endpoint (most
        # positive), back down to the zero stand on this axis, re-visiting
        # the same sampled points between 0 and most positive (descending)
        # a second time. Distinguishes two explanations for a growing
        # fwd/bwd gap that peaks at the most-positive point: (1) drift that
        # accumulates with motion/time regardless of position, which would
        # keep growing or at least not unwind here, vs (2) something
        # localised at/near that specific positive extreme (a soft
        # mechanical limit, cable strain, etc.), which should relax back
        # out once the axis leaves that region, so the zero-stand reading
        # at the end of this leg should come back close to the original
        # q_home reading (theta_h/theta_v ~ 0) if that is the explanation.
        for a in values_asc[::-1]:
            if a < 0:
                continue
            iq1, iq2 = (a, fixed) if axis == "iq1" else (fixed, a)
            add(iq1, iq2, kind_prefix + "_tozero")

    valid_j = np.nonzero(comp[i0, :])[0]
    keep_j = np.linspace(0, len(valid_j) - 1, args.n_sweep).astype(int)
    round_trip(a2[valid_j[keep_j]], a1[i0], "iq2", "roundtrip_iq2")

    valid_i = np.nonzero(comp[:, j0])[0]
    keep_i = np.linspace(0, len(valid_i) - 1, args.n_sweep).astype(int)
    round_trip(a1[valid_i[keep_i]], a2[j0], "iq1", "roundtrip_iq1")

    return rows


def build_noise_floor_poses(args, m, comp, M, lk):
    """Same (Iq1, Iq2) = (0, 0) pose repeated args.n_noise times, with
    nothing else varying -- isolates pure sensor noise from every other
    error source this file measures. The mechanism never moves between
    readings (each G command is a no-op resend of the same target, which
    doGoto() still re-settles and re-reads, same trick read_valid_quat()
    already relies on elsewhere), so any spread across these readings
    cannot be backlash, drift, or model error -- only the BNO055 itself
    and read-to-read electrical noise."""
    th_h, th_v = predict_angles(lk, args, 0.0, 0.0, M)
    rows = []
    for idx in range(args.n_noise):
        rows.append(dict(idx=idx, kind="noise_floor", Iq1_deg=0.0, Iq2_deg=0.0,
                         pred_theta_h_deg=th_h, pred_theta_v_deg=th_v,
                         measured_theta_h_deg="", measured_theta_v_deg="", note=""))
    return rows


def build_calibration_poses(args, m, comp, M, lk):
    """--self-calibrate's pose set: the single-axis sweeps above, PLUS a
    handful of genuinely combined poses (both joints well away from 0).

    bench_grid_nieuw.csv showed why the sweeps alone are not enough: a
    correction fit purely from the two sweep lines reached ~3-4 deg on
    those lines themselves, but left ~10 deg, roughly flat with distance
    from home and uncorrelated with kappa (r=0.13), once applied to a full
    2D grid of genuinely combined poses. Two lines through the origin fit a
    3-parameter orthogonal correction, but apparently do not PIN it down
    well enough for directions neither line ever visits -- the combined
    points here are exactly those missing directions."""
    rows = build_axis_sweep_poses(args, m, comp, M, lk)
    idx = len(rows)

    a1 = m["Iq1"][:, 0]
    a2 = m["Iq2"][0, :]

    # per-row/per-column valid-Iq2/Iq1 extent, so a candidate right at the
    # true edge of comp (e.g. Iq2's max reachable value at some Iq1) can be
    # excluded below -- bench_grid_nieuw4/5/6/8 all independently picked a
    # combined point at Iq2=+48.0 deg, which turned out to be EXACTLY the
    # maximum valid Iq2 at that Iq1 (not "extreme", literally the boundary)
    # and was the single most unstable point across every session,
    # regardless of whether the move there was diagonal or axis-sequential
    # -- a boundary pose stresses the real mechanism (and amplifies any
    # model/zero-offset tolerance) in a way an interior point does not.
    # a2_valid[i, j] = a2[j] where comp allows it, else NaN -- reducing
    # each ROW (axis=1, over j) gives, per Iq1, the valid Iq2 extent
    a2_valid = np.where(comp, a2[None, :], np.nan)
    row_iq2_min, row_iq2_max = np.nanmin(a2_valid, axis=1), np.nanmax(a2_valid, axis=1)
    # a1_valid[i, j] = a1[i] where comp allows it, else NaN -- reducing
    # each COLUMN (axis=0, over i) gives, per Iq2, the valid Iq1 extent
    a1_valid = np.where(comp, a1[:, None], np.nan)
    col_iq1_min, col_iq1_max = np.nanmin(a1_valid, axis=0), np.nanmax(a1_valid, axis=0)

    ii, jj = np.nonzero(comp)
    iq1v, iq2v = a1[ii], a2[jj]
    span1 = max(float(np.abs(iq1v).max()), 1e-9)
    span2 = max(float(np.abs(iq2v).max()), 1e-9)
    # both joints at least a quarter of their own span from 0, so this set
    # cannot quietly collapse back onto (near enough to) the sweep lines it
    # is meant to complement
    combined = (np.abs(iq1v) > 0.25 * span1) & (np.abs(iq2v) > 0.25 * span2)
    iq1_margin_rad = np.radians(args.iq1_margin)   # a1/a2 are in radians,
    iq2_margin_rad = np.radians(args.iq2_margin)   # args.*_margin is in deg
    margin_ok = ((iq1v >= col_iq1_min[jj] + iq1_margin_rad) &
                (iq1v <= col_iq1_max[jj] - iq1_margin_rad) &
                (iq2v >= row_iq2_min[ii] + iq2_margin_rad) &
                (iq2v <= row_iq2_max[ii] - iq2_margin_rad))
    combined = combined & margin_ok
    ci, cj = ii[combined], jj[combined]
    n_extra = min(args.n_calib_combined, len(ci))
    if n_extra == 0:
        print("waarschuwing: geen gecombineerde standen binnen het bruikbare "
              "gebied gevonden voor --self-calibrate; alleen de as-sweeps "
              "worden gebruikt")
        return rows

    # sample evenly by azimuth of the PREDICTED GAZE DIRECTION, not by raw
    # position in ci/cj, not by rank in a joint-space-angle-sorted list, and
    # not even by the joint-space angle VALUE itself (the fix this comment
    # used to describe). That joint-space-angle fix still clumped picks in
    # direction space, because (Iq1, Iq2) -> direction is strongly
    # nonlinear: an evenly-spaced joint-space angle can map to a tightly
    # clustered direction-azimuth in one region and a wide-open gap in
    # another (same effect seen and fixed in build_poses() below). For a
    # calibration set, direction-space spread is exactly what matters
    # anyway -- the fit is between PREDICTED and MEASURED directions, so
    # good coverage of distinct directions is what actually pins it down.
    t_c = m["axes"][ci, cj] @ M
    t_c = t_c / np.linalg.norm(t_c, axis=-1, keepdims=True)
    ang = np.arctan2(t_c[..., 1], t_c[..., 0])
    order = np.argsort(ang)
    ci, cj, ang = ci[order], cj[order], ang[order]
    targets = np.linspace(ang[0], ang[-1], n_extra)
    nearest = np.clip(np.searchsorted(ang, targets), 0, len(ang) - 1)
    nearest_lo = np.clip(nearest - 1, 0, len(ang) - 1)
    use_lo = np.abs(ang[nearest_lo] - targets) < np.abs(ang[nearest] - targets)
    keep = np.unique(np.where(use_lo, nearest_lo, nearest))

    # The direction-space selection above picks WHICH points to include (for
    # fit coverage), but visiting them in that same direction-sorted order
    # can still jump wildly in Iq1/Iq2 -- the (Iq1,Iq2)->direction map is
    # strongly nonlinear, so "close in direction" does not mean "close in
    # joint space" (same lesson as build_poses()'s own azimuth fix, just
    # biting on the travel order here instead of the selection). A large,
    # fast combined-axis jump can stall the motor (no acceleration ramp --
    # see bench_arduino.ino's STEP_PULSE_US comment), leaving a frozen,
    # stale sensor reading that silently corrupts the fit -- seen directly
    # in bench_grid_eindtest_calib.csv: several consecutive "different"
    # commanded poses came back with IDENTICAL measured angles, and the fit
    # residual blew up to 23/86 deg. So: selection stays direction-based
    # (unchanged, still good for coverage), but the VISIT ORDER is re-sorted
    # separately, by greedy nearest-neighbour chaining in (Iq1,Iq2) space
    # starting from wherever the sweep left off, so every move here stays
    # small regardless of the selection order above.
    cand = [(float(m["Iq1"][ci[k], cj[k]]), float(m["Iq2"][ci[k], cj[k]])) for k in keep]
    if rows:
        cur_a, cur_b = np.radians(rows[-1]["Iq1_deg"]), np.radians(rows[-1]["Iq2_deg"])
    else:
        cur_a, cur_b = 0.0, 0.0
    remaining = cand[:]
    ordered = []
    while remaining:
        dists = [(a - cur_a) ** 2 + (b - cur_b) ** 2 for a, b in remaining]
        j = int(np.argmin(dists))
        cur_a, cur_b = remaining.pop(j)
        ordered.append((cur_a, cur_b))

    for a, b in ordered:
        th_h, th_v = predict_angles(lk, args, a, b, M)
        rows.append(dict(idx=idx, kind="sweep_combined", Iq1_deg=np.degrees(a),
                         Iq2_deg=np.degrees(b), pred_theta_h_deg=th_h,
                         pred_theta_v_deg=th_v, measured_theta_h_deg="",
                         measured_theta_v_deg="", note=""))
        idx += 1
    return rows


def send_until(ser, cmd, token, tries=3, seconds=8):
    """Write `cmd`, flush, and wait for `token`; if it doesn't show up in
    time, resend and try again (up to `tries` times) before giving up.
    Manual Serial Monitor tests of the exact same commands have been
    reliable, so a single attempt from Python occasionally not landing
    looks like a transient hiccup rather than a real hang -- a retry is a
    pragmatic way to not let that stall the whole run."""
    for attempt in range(1, tries + 1):
        ser.write(cmd)
        ser.flush()
        if wait_for(ser, token, seconds):
            return True
        print(f"  (poging {attempt}/{tries} gaf geen '{token}', "
              "opnieuw versturen...)")
    return False


def wait_for(ser, token, seconds=8):
    """Read lines until `token` shows up or the deadline passes -- more
    robust than a fixed sleep + single readline, since the auto-reset
    triggered by opening the port can take a variable amount of time."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print("arduino:", line)
        if line == token:
            return True
    print(f"waarschuwing: geen '{token}' gezien binnen {seconds}s, "
          "ga toch door")
    return False


QUAT_MAG_TOL = 0.02   # bench_sensor_check.py showed occasional I2C/serial
                      # byte corruption even with the mechanism dead still
                      # (~1 in 5-10 reads) -- a corrupted component usually
                      # breaks unit length outright (a wildly wrong number,
                      # or a NaN-ish parse), which this catches; a pure sign
                      # flip on one component does NOT change the length and
                      # slips through, but that's rarer and not worth chasing
                      # here -- catching the common case automatically beats
                      # silently keeping it or making every run babysat.


def read_home_quat(ser, seconds=5, tries=3):
    """The line right after a matched 'ZEROED' is always that same
    doZero() call's 'ZQ,<w>,<x>,<y>,<z>' -- the Arduino prints them back to
    back and nothing else writes to the port while Python is waiting on
    this exchange, so reading forward until the ZQ, prefix shows up always
    picks up the quaternion captured by the ZEROED we just matched (even if
    an earlier, garbled attempt's ZEROED/ZQ pair is still being drained).

    Validated and retried just like read_valid_quat() below -- a corrupted
    HOME reading would silently bias every single theta_h/theta_v for the
    entire run, so it deserves at least the same protection as one pose."""
    for attempt in range(1, tries + 1):
        deadline = time.time() + seconds
        q = None
        while time.time() < deadline:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                print("arduino:", line)
            if line.startswith("ZQ,"):
                try:
                    q = np.array([float(v) for v in line.split(",")[1:]])
                except ValueError:
                    q = None
                break
        if (q is not None and q.shape == (4,) and
                abs(np.linalg.norm(q) - 1.0) <= QUAT_MAG_TOL):
            return q
        if attempt < tries:
            print(f"  (poging {attempt}/{tries}: ongeldige home-quaternion, "
                  "Z opnieuw versturen...)")
            ser.write(b"Z\n")
            ser.flush()
    raise SystemExit("geen geldige home-quaternion (ZQ,...) ontvangen na Z, "
                     "ook niet na opnieuw proberen")


def read_valid_quat(ser, cmd, tries=3):
    """Write `cmd` (a G,<Iq1>,<Iq2> command) and read back a quaternion,
    checking it actually has close to unit length before trusting it.
    Resending the same `cmd` is a safe no-op move (already at that target)
    -- it just re-triggers another settle+delay+read on the Arduino, not a
    second physical move. Returns (quat_or_None, last_raw_response)."""
    resp = ""
    for attempt in range(1, tries + 1):
        ser.write(cmd)
        ser.flush()
        resp = ser.readline().decode(errors="replace").strip()
        try:
            q = np.array([float(v) for v in resp.split(",")])
            if q.shape != (4,) or abs(np.linalg.norm(q) - 1.0) > QUAT_MAG_TOL:
                raise ValueError
        except ValueError:
            if attempt < tries:
                print(f"  (poging {attempt}/{tries}: ongeldige/corrupte "
                      f"lezing {resp!r}, opnieuw...)")
            continue
        return q, resp
    return None, resp


def quat_angle(a, b):
    """Angle in degrees between two unit quaternions as rotations -- abs()
    handles q and -q representing the same rotation."""
    return np.degrees(2 * np.arccos(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))


def read_consensus_quat(ser, cmd, n=3, agree_tol_deg=3.0, tries=3):
    """Take independent reads of the same pose (each already unit-length
    validated by read_valid_quat) until two of them agree, up to `n` tries.
    This catches the kind of 'stealth' corruption bench_sensor_check.py
    found even on a stationary sensor -- one component shifted but the
    quaternion still technically unit-length, so read_valid_quat's own
    check doesn't catch it, and it was visible in bench_sweep_test5 as the
    isolated points that broke an otherwise clean trend. Two genuinely
    independent I2C reads are very unlikely to reproduce the exact same
    corruption, so disagreement is the signal a single length check can't
    give. n=3 means: try up to 3 reads, but stop as soon as any two agree
    -- normally 2 reads (1 extra), only ever 3 when the first two clash.
    n=1 reproduces the old single-read behaviour. Returns
    (quat_or_None, last_raw_response)."""
    reads = []
    resp = ""
    for _ in range(max(n, 1)):
        q, resp = read_valid_quat(ser, cmd, tries=tries)
        if q is None:
            return None, resp
        reads.append(q)
        if len(reads) >= 2 and quat_angle(reads[-2], reads[-1]) <= agree_tol_deg:
            return reads[-1], resp
    return reads[-1], resp   # nothing agreed within n tries -- best effort


# ----------------------------------------------------------------------
# Iq1 mechanical-limit calibration -- small steps, human confirms each one
# ----------------------------------------------------------------------
def probe_iq1_limit(ser, direction, step_deg=3.0):
    label = "positieve" if direction > 0 else "negatieve"
    print(f"\n{label} richting -- stapjes van {step_deg:.0f} deg, kijk "
          "goed mee.")
    angle = 0.0
    last_safe = 0.0
    while True:
        angle += step_deg * direction
        ser.write(f"G,{angle:.2f},0\n".encode())
        resp = ser.readline().decode(errors="replace").strip()
        print(f"  Iq1 = {angle:+6.1f} deg  ->  {resp}")
        ans = input("  nog vrij? (Enter = ja, n = hier stoppen) "
                   ).strip().lower()
        if ans == "n":
            break
        last_safe = angle
    ser.write(f"G,{last_safe:.2f},0\n".encode())
    ser.readline()
    print(f"  limiet {label}: Iq1 = {last_safe:+.1f} deg")
    return last_safe


def calibrate_iq1(ser):
    input("\n=== Iq1-limiet kalibreren ===\nZet het mechanisme op "
          "Iq1 = 0, Iq2 = 0 en druk Enter. Na elk stapje vraag ik of "
          "het nog vrij loopt -- antwoord met 'n' zodra je weerstand "
          "voelt/ziet aankomen, niet pas als hij al vastloopt...")
    pos = probe_iq1_limit(ser, +1)
    neg = probe_iq1_limit(ser, -1)
    ser.write(b"G,0,0\n")
    ser.readline()
    print(f"\nIq1-bereik vastgelegd: {neg:+.1f} .. {pos:+.1f} deg")
    print(f"(bewaar voor volgende keer: --iq1-min {neg:.1f} "
          f"--iq1-max {pos:.1f})\n")
    return neg, pos


def probe_iq2_floor(ser, iq1_fixed, step_deg=3.0):
    """At fixed Iq1, step Iq2 negative until link 2 is about to hit the
    motor mount. One-directional (unlike probe_iq1_limit) because the
    interference observed so far was always 'Iq2 too far below Iq1', never
    too far above."""
    print(f"\nIq1 vast op {iq1_fixed:+.1f} deg -- Iq2 stapsgewijs negatief, "
          f"stapjes van {step_deg:.0f} deg, kijk goed mee.")
    angle = 0.0
    last_safe = 0.0
    while True:
        angle -= step_deg
        ser.write(f"G,{iq1_fixed:.2f},{angle:.2f}\n".encode())
        resp = ser.readline().decode(errors="replace").strip()
        print(f"  Iq2 = {angle:+6.1f} deg  ->  {resp}")
        ans = input("  nog vrij? (Enter = ja, n = hier stoppen) "
                   ).strip().lower()
        if ans == "n":
            break
        last_safe = angle
    ser.write(f"G,{iq1_fixed:.2f},{last_safe:.2f}\n".encode())
    ser.readline()
    print(f"  grens bij Iq1={iq1_fixed:+.1f}: Iq2 >= {last_safe:+.1f} deg")
    return last_safe


def calibrate_iq2_floor(ser, iq1_min, iq1_max, n_points=4):
    """Probe how far Iq2 can go negative, relative to Iq1, before link 2
    hits the motor mount -- at a handful of Iq1 values spanning the
    already-calibrated Iq1 range, since the real interference depends on
    where Iq1 is too, not just the Iq1/Iq2 difference (see iq2_floor_mask)."""
    input("\n=== Iq2-ondergrens kalibreren (botsing met motorstandaard) ===\n"
          "Zet het mechanisme op Iq1 = 0, Iq2 = 0 en druk Enter. Op een "
          f"paar Iq1-standen tussen {iq1_min:+.1f} en {iq1_max:+.1f} deg "
          "stap ik Iq2 negatief; antwoord met 'n' zodra link 2 de "
          "motorstandaard nadert, niet pas als hij er al tegenaan zit...")
    iq1_points = np.linspace(iq1_min, iq1_max, n_points)
    floors = []
    for iq1 in iq1_points:
        # always return to Iq2=0 (known safe, already-validated Iq1 range)
        # before moving to the next Iq1 -- a direct diagonal jump from one
        # probe's Iq2 floor straight to the next probe's start could cut
        # through the very collision zone this calibration is trying to map
        ser.write(f"G,{float(iq1):.2f},0\n".encode())
        ser.readline()
        floors.append(probe_iq2_floor(ser, float(iq1)))
    ser.write(b"G,0,0\n")
    ser.readline()
    pts = list(zip([float(x) for x in iq1_points], floors))
    pts_str = ";".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    print(f"\nIq2-ondergrens vastgelegd op {n_points} punten.")
    print(f"(bewaar voor volgende keer: --iq2-floor \"{pts_str}\")\n")
    return pts


def parse_iq2_floor(s):
    """'-42.0,-30.0;0.0,-45.0;38.0,-20.0' -> [(-42.0,-30.0), (0.0,-45.0),
    (38.0,-20.0)] -- the --iq2-floor reuse format, printed verbatim by
    calibrate_iq2_floor()."""
    pts = []
    for part in s.split(";"):
        a, b = part.split(",")
        pts.append((float(a), float(b)))
    return pts


def drive_and_measure(ser, rows, m, comp, q_home, consensus_reads=3,
                      combined_settle_extra=1.0, start=(0.0, 0.0)):
    # a large Iq1/Iq2 swing can take several seconds of blocking stepping
    # on the Arduino before it replies -- the short timeout used for
    # boot/zero detection would give up mid-move and desync the sequence
    # (Python moving on to the next command while the Arduino is still
    # working through the previous one), which is almost certainly what
    # caused the runaway Iq2 rotation.
    ser.timeout = 25
    # `start` [rad] defaults to true home (right after a Z-zero), but MUST
    # be passed explicitly as the previous call's returned end position
    # when this is a second drive_and_measure() in the same zero-session
    # (e.g. the grid after --self-calibrate's own sweep) -- otherwise this
    # function thinks it's starting from (0,0) when the mechanism is
    # actually still wherever the previous call left it, so the very first
    # move's real distance is invisible to choose_waypoints()/subdivide()
    # (both see start==target and skip straight to a single raw jump) and
    # to path_ok() (checked against the wrong starting cell). Confirmed
    # directly: a session whose calibration sweep ended far from home
    # (Iq1=-64, Iq2=-42) showed a consistent ~16-17 deg theta_h offset on
    # EVERY grid pose from the very first one (home itself, post-correction,
    # reading nonzero) -- i.e. that one un-subdivided, unchecked return-to-
    # home jump lost sync and never recovered for the rest of the session.
    cur = start

    # largest single commanded jump along one leg before it gets broken into
    # smaller intermediate stops -- without this, a long leg (e.g. between
    # two widely separated --self-calibrate combined points) can stall the
    # motor (no acceleration ramp, see bench_arduino.ino's STEP_PULSE_US
    # comment): seen directly in bench_grid_eindtest_calib.csv, where a
    # 55-82 deg jump left several consecutive "different" commanded poses
    # reading back an IDENTICAL, stale orientation and wrecked the fit
    # (23/86 deg residual). choose_waypoints() below already avoids
    # diagonal-move instability by going one axis at a time; this adds the
    # orthogonal fix of also keeping each axis's own jump size bounded.
    MAX_STEP_RAD = np.radians(8.0)   # was 15.0 -- the firmware's new
    # acceleration ramp (see bench_arduino.ino's RAMP_* constants) cut the
    # stall rate at the single largest leg in the combined-calibration
    # sweep (63 deg) from "every time" to "sometimes" (+5.2 deg offset
    # with no ramp, +0.1 deg in one ramped run, +3.4 deg in another --
    # still a probabilistic failure, not eliminated). Halving the max leg
    # size doubles how much of each substep is spent in the ramp's
    # acceleration/deceleration zone (a fixed ~1.7 deg at each end),
    # trading a handful of extra settle cycles for a bigger safety margin
    # on exactly the move that has already caused two lost-sync events.

    def subdivide(start, end, max_step):
        dist = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        n = max(1, int(np.ceil(dist / max_step)))
        return [(start[0] + (end[0] - start[0]) * t,
                 start[1] + (end[1] - start[1]) * t)
                for t in np.linspace(0.0, 1.0, n + 1)[1:]]

    def goto(a, b):
        """Move to (a, b) [rad], checked against comp, WITHOUT touching
        cur -- the caller updates cur once the physical move has actually
        happened, same reasoning as the measured-pose case below."""
        a_deg, b_deg = np.degrees(a), np.degrees(b)
        if not path_ok(m, comp, cur[0], cur[1], a, b):
            return False, None, None
        cmd = f"G,{a_deg:.4f},{b_deg:.4f}\n".encode()
        q, resp = read_consensus_quat(ser, cmd, n=consensus_reads)
        return True, q, resp

    def choose_waypoints(start, target):
        """For a genuinely combined move (both joints differ from `start`),
        move one axis at a time instead of diagonally -- simultaneous
        motion visibly destabilises the mechanism at some combined poses
        (P3 has to be pulled past a point while both motors pull at once,
        sometimes leaving a lasting offset, not just a transient wobble
        that settling time would fix). But comp is not a rectangle: the
        corner an axis-first path creates (e.g. the old Iq1 held at its
        new extreme while Iq2 is still at its OLD extreme) can fall
        outside the valid region even when the direct line and both true
        endpoints are fine. Try both orders and take whichever is
        entirely safe; if neither is, fall back to the direct diagonal
        move rather than losing the pose."""
        a, b = target
        if (np.isclose(a, start[0], atol=1e-6) or
                np.isclose(b, start[1], atol=1e-6)):
            return [target]
        for corner in ((a, start[1]), (start[0], b)):
            if (path_ok(m, comp, start[0], start[1], *corner) and
                    path_ok(m, comp, corner[0], corner[1], a, b)):
                return [corner, target]
        return [target]

    for r in rows:
        a_deg, b_deg = float(r["Iq1_deg"]), float(r["Iq2_deg"])
        a, b = np.radians(a_deg), np.radians(b_deg)

        is_combined = (not np.isclose(a, cur[0], atol=1e-6) and
                      not np.isclose(b, cur[1], atol=1e-6))
        coarse_waypoints = choose_waypoints(cur, (a, b))
        waypoints, leg_start = [], cur
        for wp in coarse_waypoints:
            waypoints.extend(subdivide(leg_start, wp, MAX_STEP_RAD))
            leg_start = wp

        ok = True
        q_cur = resp = None
        for wp in waypoints:
            ok, q_cur, resp = goto(*wp)
            if not ok:
                print(f"pose {r['idx']:>3}: pad naar Iq1={np.degrees(wp[0]):+.1f} "
                      f"Iq2={np.degrees(wp[1]):+.1f} loopt buiten het "
                      "bruikbare gebied -- overgeslagen")
                r["note"] = (r["note"] + " pad onveilig, overgeslagen").strip()
                break
            # the move itself already happened (Arduino steps before it ever
            # replies), regardless of whether the quaternion read back was
            # valid -- cur has to track physical reality, not read success,
            # or the NEXT leg's/pose's path_ok() safety check runs against a
            # stale, wrong starting point
            cur = wp

        if not ok:
            continue
        if is_combined and combined_settle_extra > 0 and q_cur is not None:
            # just finished the second (Iq2) leg of a split combined move --
            # still give it the same extra settle+reread as before, now
            # applied once at the true end of the (now L-shaped) transit
            time.sleep(combined_settle_extra)
            _, q_settled, resp_settled = goto(a, b)
            if q_settled is not None:
                q_cur, resp = q_settled, resp_settled
        if q_cur is None:
            print(f"pose {r['idx']:>3}: onverwacht antwoord {resp!r}, "
                  "overgeslagen")
            r["note"] = (r["note"] + " arduino-fout").strip()
            continue
        th_h, th_v = theta_hv_from_quats(q_home, q_cur)
        r["measured_theta_h_deg"] = f"{th_h:.3f}"
        r["measured_theta_v_deg"] = f"{th_v:.3f}"
        print(f"pose {r['idx']:>3}  Iq1={a_deg:+7.2f} Iq2={b_deg:+7.2f}  "
              f"->  theta_h={th_h:+7.2f} theta_v={th_v:+7.2f}")
    return cur


def analyse_axis_sweep(rows, args):
    """For --axis-sweep data: one joint moved at a time, the other pinned
    at 0. Plotted as pred vs measured against the swept joint angle (not
    as a 2D theta_h/theta_v scatter) so a per-axis scale/calibration error
    shows up as a slope != 1 line, a zeroing/backlash offset as a
    vertical shift, and backlash specifically as scatter around an
    otherwise-good line -- each a different, distinguishable shape,
    which a 2D grid run can't separate.

    Uses corrected_theta_h/v_deg when present (--self-calibrate combined
    with --axis-sweep) instead of the raw measured_theta_h/v_deg -- the
    raw comparison is otherwise dominated by the already-understood
    mounting-rotation offset, which would swamp the per-axis effects this
    plot exists to isolate."""
    rows = [r for r in rows if r["measured_theta_h_deg"] not in ("", None)]
    if not rows:
        raise SystemExit("nog geen metingen om te analyseren")
    use_corrected = rows[0].get("corrected_theta_h_deg") not in ("", None)
    h_field = "corrected_theta_h_deg" if use_corrected else "measured_theta_h_deg"
    v_field = "corrected_theta_v_deg" if use_corrected else "measured_theta_v_deg"
    label = "gecorrigeerd gemeten" if use_corrected else "gemeten"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, (kind, joint_field, joint_label) in enumerate(
            [("sweep_iq1", "Iq1_deg", "Iq1"), ("sweep_iq2", "Iq2_deg", "Iq2")]):
        sel = [r for r in rows if r["kind"] == kind]
        if not sel:
            continue
        x = np.array([float(r[joint_field]) for r in sel])
        order = np.argsort(x)
        x = x[order]
        ph = np.array([float(sel[i]["pred_theta_h_deg"]) for i in order])
        pv = np.array([float(sel[i]["pred_theta_v_deg"]) for i in order])
        mh = np.array([float(sel[i][h_field]) for i in order])
        mv = np.array([float(sel[i][v_field]) for i in order])

        print(f"\n{kind} ({len(sel)} poses, {joint_label} varieert, de "
              f"andere as vast op 0, {label}):")
        for name, p, mm in (("theta_h", ph, mh), ("theta_v", pv, mv)):
            if np.ptp(p) < 1e-6:
                print(f"  {name}: voorspelling varieert nauwelijks over "
                      "deze sweep, sla fit over")
                continue
            slope, intercept = np.polyfit(p, mm, 1)
            resid = mm - (slope * p + intercept)
            print(f"  {name}: {label} = {slope:.3f} * voorspeld + "
                  f"{intercept:+.2f} deg   (rest-spreiding "
                  f"{resid.std():.2f} deg)")
            print("    slope 1.0 & intercept 0 = model klopt op deze as; "
                  "slope != 1 wijst op schaal/overbrenging; intercept != 0 "
                  "op een offset; spreiding rond de fit op backlash")

        for row_, (name, p, mm) in enumerate((("theta_h", ph, mh),
                                              ("theta_v", pv, mv))):
            ax = axes[row_, col]
            ax.plot(x, p, "-o", mfc="none", label="voorspeld", ms=5)
            ax.plot(x, mm, "-x", label=label, ms=6)
            ax.set_xlabel(f"{joint_label} [deg]")
            ax.set_ylabel(f"{name} [deg]")
            ax.grid(alpha=.3)
            ax.legend()
            ax.set_title(f"{kind}: {name} vs {joint_label}")
    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + "_check.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\ngeschreven: {out}")


def analyse_round_trip(rows, args):
    """For --round-trip data: forward (positive->negative) and backward
    (negative->positive) passes over the identical points. Plots both as
    lines against the swept joint angle so a hysteresis loop (heen- vs.
    terugpad apart) is directly visible, reports the average/max gap
    between the two passes at matching points (backlash), and reports
    drift separately via the one point visited at both the very start and
    very end of the whole round trip."""
    rows = [r for r in rows if r["measured_theta_h_deg"] not in ("", None)]
    if not rows:
        raise SystemExit("nog geen metingen om te analyseren")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, (axis, joint_field, joint_label) in enumerate(
            [("iq1", "Iq1_deg", "Iq1"), ("iq2", "Iq2_deg", "Iq2")]):
        fwd = [r for r in rows if r["kind"] == f"roundtrip_{axis}_fwd"]
        bwd = [r for r in rows if r["kind"] == f"roundtrip_{axis}_bwd"]
        if not fwd or not bwd:
            continue

        xf = np.array([float(r[joint_field]) for r in fwd])
        xb = np.array([float(r[joint_field]) for r in bwd])
        mhf = np.array([float(r["measured_theta_h_deg"]) for r in fwd])
        mvf = np.array([float(r["measured_theta_v_deg"]) for r in fwd])
        mhb = np.array([float(r["measured_theta_h_deg"]) for r in bwd])
        mvb = np.array([float(r["measured_theta_v_deg"]) for r in bwd])

        tz = [r for r in rows if r["kind"] == f"roundtrip_{axis}_tozero"]
        xz = np.array([float(r[joint_field]) for r in tz]) if tz else None
        mhz = np.array([float(r["measured_theta_h_deg"]) for r in tz]) if tz else None
        mvz = np.array([float(r["measured_theta_v_deg"]) for r in tz]) if tz else None

        for row_i, (mf, mb, mz, name) in enumerate(
                [(mhf, mhb, mhz, "theta_h"), (mvf, mvb, mvz, "theta_v")]):
            ax = axes[row_i][col]
            ax.plot(xf, mf, "o-", label="heen (+ -> -)", ms=4)
            ax.plot(xb, mb, "s-", label="terug (- -> +)", ms=4)
            if tz:
                ax.plot(xz, mz, "^--", label="extra: terug naar 0", ms=4,
                        color="#888888")
            ax.set_xlabel(f"{joint_label} [deg]")
            ax.set_ylabel(f"{name} [deg]")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            ax.set_title(f"{axis}-rondgang: {name} vs {joint_label}")

        # both passes visit the identical angle list -- sort each by angle
        # so matching points line up for a direct heen/terug vergelijking
        order_f = np.argsort(xf)
        order_b = np.argsort(xb)
        gap_h = mhb[order_b] - mhf[order_f]
        gap_v = mvb[order_b] - mvf[order_f]
        gap = np.hypot(gap_h, gap_v)
        print(f"\n{joint_label}-rondgang: {len(xf)} gedeelde standen")
        print(f"  hysteresekloof (terug - heen): gemiddeld {gap.mean():.2f} "
              f"deg, max {gap.max():.2f} deg")

        # drift: the fwd pass's FIRST point (most positive) is re-measured
        # as the bwd pass's LAST point -- the only point visited at both
        # the very start and very end of the whole round trip
        i_first_fwd = int(np.argmax(xf))
        i_last_bwd = int(np.argmax(xb))
        dh = mhb[i_last_bwd] - mhf[i_first_fwd]
        dv = mvb[i_last_bwd] - mvf[i_first_fwd]
        print(f"  drift (zelfde stand, begin vs. eind van de rondgang): "
              f"dtheta_h={dh:+.2f} deg, dtheta_v={dv:+.2f} deg")

        # extra "tozero" leg (if present): third visit to the same
        # 0..most-positive points, re-using the fwd pass (before any
        # turnaround) as the baseline -- shows whether the gap keeps
        # growing with further motion, or relaxes back out once the axis
        # returns to 0, which would point at something localised near the
        # positive extreme rather than motion-accumulated drift
        if tz:
            # match each tozero point back to the fwd pass's ORIGINAL visit
            # to that same angle (the earliest, pre-turnaround measurement)
            # via a plain dict keyed on the angle, rounded to dodge float
            # repr noise -- simpler and more robust than index arithmetic
            fwd_by_angle = {round(x, 3): (h, v) for x, h, v in zip(xf, mhf, mvf)}
            gaps_h, gaps_v = [], []
            for x, h, v in zip(xz, mhz, mvz):
                key = round(x, 3)
                if key in fwd_by_angle:
                    h0, v0 = fwd_by_angle[key]
                    gaps_h.append(h - h0)
                    gaps_v.append(v - v0)
            gap_z = np.hypot(np.array(gaps_h), np.array(gaps_v))
            print(f"  extra been terug naar nulstand: gat t.o.v. de EERSTE "
                  f"(heen-)meting op diezelfde standen: gemiddeld "
                  f"{gap_z.mean():.2f} deg, max {gap_z.max():.2f} deg "
                  f"(vs. {gap.mean():.2f}/{gap.max():.2f} deg voor de "
                  "gewone heen/terug-kloof hierboven)")
            i_near_zero = int(np.argmin(np.abs(xz)))
            ph = float(tz[i_near_zero]["pred_theta_h_deg"])
            pv = float(tz[i_near_zero]["pred_theta_v_deg"])
            print(f"  meting bij {joint_label}={xz[i_near_zero]:+.2f} deg "
                  f"(dichtst bij nulstand), na de hele rondgang: "
                  f"theta_h={mhz[i_near_zero]:+.2f} deg (voorspeld "
                  f"{ph:+.2f}), theta_v={mvz[i_near_zero]:+.2f} deg "
                  f"(voorspeld {pv:+.2f})")

    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + "_roundtrip.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\ngeschreven: {out}")


def analyse_noise_floor(rows, args):
    """For --noise-floor data: the mechanism never moved, so any spread in
    the readings is purely the BNO055 (plus read-to-read electrical
    noise) -- the number to put next to the model-vs-bank error and the
    sensor's own datasheet accuracy spec, to show which one is actually
    the limiting factor."""
    rows = [r for r in rows if r["measured_theta_h_deg"] not in ("", None)]
    if len(rows) < 2:
        raise SystemExit("te weinig metingen om sensorruis te berekenen")

    mh = np.array([float(r["measured_theta_h_deg"]) for r in rows])
    mv = np.array([float(r["measured_theta_v_deg"]) for r in rows])

    print(f"\n{len(rows)} metingen op dezelfde stilstaande pose "
          f"(Iq1=0, Iq2=0) -- puur sensorruis, geen mechanische beweging:")
    print(f"  theta_h: std {mh.std():.3f} deg, spreiding (max-min) "
          f"{mh.max() - mh.min():.3f} deg")
    print(f"  theta_v: std {mv.std():.3f} deg, spreiding (max-min) "
          f"{mv.max() - mv.min():.3f} deg")
    combined = np.hypot(mh.std(), mv.std())
    print(f"  gecombineerd (hypot van beide std's): {combined:.3f} deg")
    print("  vergelijk dit met de BNO055-datasheet-nauwkeurigheid en met "
          "de modelfout na correctie (~3-5 deg) -- ligt dit dicht bij "
          "elkaar, dan is de sensor zelf de beperkende factor, niet het "
          "model of de bank.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (name, vals) in zip(axes, (("theta_h", mh), ("theta_v", mv))):
        ax.plot(vals, "o-", ms=4)
        ax.axhline(vals.mean(), color="gray", ls="--", lw=1)
        ax.set_xlabel("meting #")
        ax.set_ylabel(f"{name} [deg]")
        ax.set_title(f"{name}: std {vals.std():.3f} deg")
        ax.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + "_noisefloor.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\ngeschreven: {out}")


def analyse(rows, args):
    rows = [r for r in rows if r["measured_theta_h_deg"] not in ("", None)]
    if not rows:
        raise SystemExit("nog geen metingen om te analyseren")

    ph = np.array([float(r["pred_theta_h_deg"]) for r in rows])
    pv = np.array([float(r["pred_theta_v_deg"]) for r in rows])
    mh = np.array([float(r["measured_theta_h_deg"]) for r in rows])
    mv = np.array([float(r["measured_theta_v_deg"]) for r in rows])
    kind = np.array([r["kind"] for r in rows])

    def vec(h, v):
        h, v = np.radians(h), np.radians(v)
        return np.stack([np.cos(v) * np.sin(h), np.sin(v),
                         np.cos(v) * np.cos(h)], axis=-1)
    ang = np.degrees(np.arccos(np.clip(np.sum(vec(mh, mv) * vec(ph, pv), -1),
                                       -1, 1)))

    grid = kind == "grid"
    rep = ~grid
    print(f"\n{len(rows)} gemeten poses ({grid.sum()} grid, {rep.sum()} "
          f"repeat)")
    print(f"\nhoekfout, model vs bench:")
    print(f"  gemiddeld {ang[grid].mean():6.2f} deg   rms "
          f"{np.sqrt((ang[grid]**2).mean()):6.2f} deg    max "
          f"{ang[grid].max():6.2f} deg")
    print(f"  bias      theta_h {np.mean(mh[grid]-ph[grid]):+6.2f} deg   "
          f"theta_v {np.mean(mv[grid]-pv[grid]):+6.2f} deg")
    print(f"  spreiding theta_h {np.std(mh[grid]-ph[grid]):6.2f} deg   "
          f"theta_v {np.std(mv[grid]-pv[grid]):6.2f} deg")

    if rows and all(r.get("corrected_theta_h_deg") not in ("", None)
                   for r in rows):
        ch = np.array([float(r["corrected_theta_h_deg"]) for r in rows])
        cv = np.array([float(r["corrected_theta_v_deg"]) for r in rows])
        ang_c = np.degrees(np.arccos(np.clip(
            np.sum(vec(ch, cv) * vec(ph, pv), -1), -1, 1)))
        print(f"\nhoekfout, model vs bench, NA sessie-zelfkalibratie:")
        print(f"  gemiddeld {ang_c[grid].mean():6.2f} deg   rms "
              f"{np.sqrt((ang_c[grid]**2).mean()):6.2f} deg    max "
              f"{ang_c[grid].max():6.2f} deg")

    if rep.any():
        spreads = []
        for k in set(zip(np.round([float(r["Iq1_deg"]) for r in rows], 3),
                         np.round([float(r["Iq2_deg"]) for r in rows], 3))):
            sel = np.array([(round(float(r["Iq1_deg"]), 3),
                             round(float(r["Iq2_deg"]), 3)) == k
                            for r in rows]) & rep
            if sel.sum() > 1:
                spreads.append(np.hypot(mh[sel].std(), mv[sel].std()))
        if spreads:
            print(f"\nherhaalbaarheid (spreiding bij terugkeer naar dezelfde "
                  f"pose): {np.mean(spreads):.2f} deg")
            print("  modelfout is alleen wat hierboven uitkomt -- als beide "
                  "even groot zijn,\n  is de bench de beperkende factor, "
                  "niet het model.")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax[0].plot(ph, pv, "o", mfc="none", label="voorspeld", ms=7)
    ax[0].plot(mh, mv, "x", label="gemeten", ms=7)
    for a_, b_, c_, d_ in zip(ph, pv, mh, mv):
        ax[0].plot([a_, c_], [b_, d_], "-", color="0.7", lw=.8, zorder=0)
    ax[0].set_xlabel(r"$\theta_h$ [deg]"); ax[0].set_ylabel(r"$\theta_v$ [deg]")
    ax[0].set_aspect("equal"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_title("model vs bench")

    sc = ax[1].scatter(ph, pv, c=ang, cmap="magma_r", s=60)
    fig.colorbar(sc, ax=ax[1], label="hoekfout [deg]")
    ax[1].set_xlabel(r"voorspelde $\theta_h$ [deg]")
    ax[1].set_ylabel(r"voorspelde $\theta_v$ [deg]")
    ax[1].set_aspect("equal"); ax[1].grid(alpha=.3)
    ax[1].set_title("waar het model afwijkt")
    fig.tight_layout()
    out = os.path.splitext(args.csv)[0] + "_check.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\ngeschreven: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="bv. /dev/tty.usbmodem1101 (mac) of "
                                   "COM5 (windows)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", default="results_v2/bench_run.csv")
    ap.add_argument("--check-only", action="store_true",
                    help="alleen een bestaande CSV analyseren, niet "
                         "opnieuw aansturen")
    ap.add_argument("--calibrate-iq1", action="store_true",
                    help="Iq1-mechanische limieten interactief meten "
                         "vóór het testen")
    ap.add_argument("--iq1-min", type=float, default=None,
                    help="bekende Iq1-ondergrens in graden (voorkomt dat "
                         "--calibrate-iq1 opnieuw moet)")
    ap.add_argument("--iq1-max", type=float, default=None)
    ap.add_argument("--iq1-margin", type=float, default=3.0,
                    help="veiligheidsmarge [deg] die testposes binnen "
                         "--iq1-min/--iq1-max houdt, zonder die kalibratie "
                         "zelf aan te passen (default 3.0)")
    ap.add_argument("--iq2-margin", type=float, default=3.0,
                    help="veiligheidsmarge [deg] die testposes boven de "
                         "gekalibreerde --iq2-floor houdt, zonder die "
                         "kalibratie zelf aan te passen -- bedoeld tegen "
                         "poses die (zoals het eerste sweep-punt tot nu "
                         "toe) vlak op de motorstandaard-grens landen "
                         "(default 3.0)")
    ap.add_argument("--calibrate-iq2-floor", action="store_true",
                    help="Iq2-ondergrens t.o.v. Iq1 interactief meten "
                         "vóór het testen (link 2 vs de motorstandaard) "
                         "-- vereist een bekend Iq1-bereik (--iq1-min/"
                         "--iq1-max of --calibrate-iq1)")
    ap.add_argument("--iq2-floor", type=str, default=None,
                    help="bekende Iq2-ondergrens-punten, bv. "
                         "'-42.0,-30.0;0.0,-45.0;38.0,-20.0' (voorkomt dat "
                         "--calibrate-iq2-floor opnieuw moet)")
    ap.add_argument("--n-iq2-floor-points", type=int, default=4)
    ap.add_argument("--iq2-max", type=float, default=None,
                    help="absolute bovengrens voor Iq2 [deg] -- los van de "
                         "Iq2-ondergrens-tabel (die alleen een onderkant "
                         "kent); gebruik dit als Iq2 ook niet te ver "
                         "positief mag komen")
    ap.add_argument("--consensus-reads", type=int, default=3,
                    help="tot zoveel keer dezelfde pose onafhankelijk "
                         "lezen tot 2 lezingen het eens zijn (vangt een "
                         "losse 'stille' sensorglitch die nog wel een "
                         "geldige eenheidslengte heeft); 1 = oud gedrag, "
                         "altijd maar 1 lezing")
    ap.add_argument("--combined-settle-extra", type=float, default=1.0,
                    help="extra wachttijd [s] vóór de meting wanneer Iq1 "
                         "EN Iq2 allebei bewegen t.o.v. de vorige pose "
                         "(de Arduino antwoordt al zodra de stappen klaar "
                         "zijn, maar het mechanisme kan dan nog mechanisch "
                         "natrillen -- vooral merkbaar bij gecombineerde "
                         "bewegingen). 0 = uit")
    ap.add_argument("--design", type=int, default=0)
    ap.add_argument("--sort", default="home_cone",
                    choices=["home_cone", "round", "box", "omega"])
    ap.add_argument("--results", default="v2", choices=["v1", "v2", "eye"])
    ap.add_argument("--params")
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--branch", type=int, default=EVAL_KW["branch"])
    ap.add_argument("--n-pose", type=int, default=24)
    ap.add_argument("--n-repeat", type=int, default=3)
    ap.add_argument("--axis-sweep", action="store_true",
                    help="1 as tegelijk doorlopen (de andere vast op 0) "
                         "i.p.v. het 2D-grid -- isoleert een schaal-/"
                         "overbrengingsfout per as van een koppel-/"
                         "geometriefout die alleen optreedt als beide "
                         "assen samen bewegen")
    ap.add_argument("--n-sweep", type=int, default=13)
    ap.add_argument("--round-trip", action="store_true",
                    help="1 as tegelijk doorlopen van het meest positieve "
                         "punt naar het meest negatieve, en dan dezelfde "
                         "punten weer terug -- heen/terug-verschil per "
                         "punt is backlash, het verschil tussen het "
                         "eerste en laatste punt (hetzelfde punt, aan "
                         "weerszijden van de hele rondgang) is drift. "
                         "Niet te combineren met --axis-sweep of "
                         "--self-calibrate.")
    ap.add_argument("--noise-floor", action="store_true",
                    help="dezelfde stilstaande pose (Iq1=0, Iq2=0) "
                         "--n-noise keer achter elkaar lezen zonder te "
                         "bewegen -- isoleert pure sensorruis van elke "
                         "andere foutbron in dit bestand. Niet te "
                         "combineren met --axis-sweep, --round-trip of "
                         "--self-calibrate.")
    ap.add_argument("--n-noise", type=int, default=30,
                    help="aantal herhaalde lezingen voor --noise-floor")
    ap.add_argument("--self-calibrate", action="store_true",
                    help="vóór de grid-test, binnen dezelfde zero-sessie, "
                         "eerst een as-sweep rijden en daaruit de "
                         "montagecorrectie fitten (asnegatie plus het "
                         "restje dat per keer zoeken net iets anders "
                         "uitvalt -- zie theta_hv_from_quats). Die "
                         "correctie wordt gebruikt om corrected_theta_h/"
                         "v_deg te vullen in de grid-CSV van diezelfde "
                         "sessie; measured_theta_h/v_deg blijft de ruwe "
                         "waarde. Samen met --axis-sweep: geen aparte "
                         "grid-test, de gerapporteerde sweep IS de "
                         "kalibratie-sweep (zelfde punten, niet dubbel "
                         "gereden) -- handig voor een losse, dichtere "
                         "gekalibreerde sweep-test zonder de grid-test "
                         "langer te maken (zet --n-sweep dan hoger).")
    ap.add_argument("--n-calib-combined", type=int, default=10,
                    help="aantal echt-gecombineerde standen (beide "
                         "gewrichten weg van 0) die --self-calibrate naast "
                         "de as-sweeps meeneemt in de kalibratieset -- de "
                         "sweeps alleen leggen de correctie niet genoeg "
                         "vast voor gecombineerde beweging (zie "
                         "build_calibration_poses)")
    ap.add_argument("--n-input", type=int, default=301)
    ap.add_argument("--access-deg", type=float, default=90.0)
    ap.add_argument("--link-diameter", type=float, default=0.03)
    ap.add_argument("--eye-radius", type=float, default=None)
    args = ap.parse_args()

    combo_count = sum([args.self_calibrate, args.axis_sweep, args.round_trip,
                      args.noise_floor])
    # --self-calibrate + --axis-sweep together is the one allowed pair: a
    # standalone, self-calibrated single-axis sweep test (its own --n-sweep
    # density, independent of and not adding to a separate --n-pose grid
    # run) -- see the drive step below, which reuses the calibration
    # sweep's own measurement instead of driving it a second time. Every
    # other combination still conflicts (each of the four decides its own
    # pose set from scratch).
    if combo_count > 1 and not (combo_count == 2 and args.self_calibrate
                                and args.axis_sweep):
        raise SystemExit("--self-calibrate, --axis-sweep, --round-trip en "
                         "--noise-floor combineren niet onderling (behalve "
                         "--self-calibrate + --axis-sweep samen) -- elk "
                         "bepaalt zelf welke poses er gereden worden")

    if args.check_only:
        with open(args.csv, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if any(r["kind"] == "noise_floor" for r in rows):
            analyse_noise_floor(rows, args)
        elif any(r["kind"].startswith("roundtrip_") for r in rows):
            analyse_round_trip(rows, args)
        elif any(r["kind"].startswith("sweep_") for r in rows):
            analyse_axis_sweep(rows, args)
        else:
            analyse(rows, args)
        return

    if not args.port:
        raise SystemExit("--port is verplicht, tenzij --check-only")

    lk = load_design(args)

    iq1_range_deg = None
    if args.iq1_min is not None and args.iq1_max is not None:
        iq1_range_deg = (args.iq1_min, args.iq1_max)

    iq2_floor_pts = parse_iq2_floor(args.iq2_floor) if args.iq2_floor else None

    need_calibration = ((iq1_range_deg is None and args.calibrate_iq1) or
                        (iq2_floor_pts is None and args.calibrate_iq2_floor))

    if not need_calibration:
        # nothing live to probe -- do the (possibly slow, CPU-only)
        # workspace computation BEFORE opening the serial connection, so
        # there is no long idle gap on the link between connecting and
        # actually sending Z. A long silent gap on a USB-serial link is a
        # known source of flakiness (e.g. macOS power-managing the port)
        # that a quick manual Serial Monitor test would never hit.
        m, comp, M = build_grid(lk, args, iq1_range_deg, iq2_floor_pts, args.iq2_max)
        rows = (build_noise_floor_poses(args, m, comp, M, lk) if args.noise_floor
                else build_round_trip_poses(args, m, comp, M, lk) if args.round_trip
                else build_axis_sweep_poses(args, m, comp, M, lk) if args.axis_sweep
                else build_poses(args, m, comp, M, lk))
        sweep_rows = (build_calibration_poses(args, m, comp, M, lk)
                     if args.self_calibrate else None)
        ser = serial.Serial(args.port, args.baud, timeout=2)
        wait_for(ser, "READY")
        ser.reset_input_buffer()
    else:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        wait_for(ser, "READY")
        ser.reset_input_buffer()
        if iq1_range_deg is None and args.calibrate_iq1:
            iq1_range_deg = calibrate_iq1(ser)
        if iq2_floor_pts is None and args.calibrate_iq2_floor:
            if iq1_range_deg is None:
                raise SystemExit("--calibrate-iq2-floor heeft een bekend "
                                 "Iq1-bereik nodig: geef --iq1-min/--iq1-max "
                                 "of --calibrate-iq1 mee")
            iq2_floor_pts = calibrate_iq2_floor(ser, *iq1_range_deg,
                                                n_points=args.n_iq2_floor_points)
        m, comp, M = build_grid(lk, args, iq1_range_deg, iq2_floor_pts, args.iq2_max)
        rows = (build_noise_floor_poses(args, m, comp, M, lk) if args.noise_floor
                else build_round_trip_poses(args, m, comp, M, lk) if args.round_trip
                else build_axis_sweep_poses(args, m, comp, M, lk) if args.axis_sweep
                else build_poses(args, m, comp, M, lk))
        sweep_rows = (build_calibration_poses(args, m, comp, M, lk)
                     if args.self_calibrate else None)

    # explicit every time, not just relied on from setup() -- opening the
    # serial port doesn't reliably reset the board on every run, so without
    # this the drivers can still be holding position from a PREVIOUS
    # session's last move, fighting the hand trying to re-zero here
    send_until(ser, b"F\n", "FREE")
    input("\nZet het mechanisme fysiek op Iq1 = 0, Iq2 = 0 (home) en "
          "druk dan Enter om de testreeks te starten...")
    ser.reset_input_buffer()   # discard any stale bytes left over from
                              # calibration/the wait above
    send_until(ser, b"Z\n", "ZEROED")
    q_home = read_home_quat(ser)

    R_session = None
    cur = (0.0, 0.0)  # rad -- tracks real physical position across the two
                      # drive_and_measure() calls below, so the second one
                      # (the grid) knows where the first one (the
                      # calibration sweep) actually left the mechanism,
                      # instead of wrongly assuming it's still at home
    if args.self_calibrate:
        print(f"\n=== zelfkalibratie: {len(sweep_rows)} sweep-punten, "
              "binnen deze zero-sessie ===")
        cur = drive_and_measure(ser, sweep_rows, m, comp, q_home,
                                consensus_reads=args.consensus_reads,
                                combined_settle_extra=args.combined_settle_extra,
                                start=cur)
        calib_path = os.path.splitext(args.csv)[0] + "_calib.csv"
        os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)
        with open(calib_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(sweep_rows)
        print(f"geschreven: {calib_path}")
        try:
            R_session, mean_e, max_e = fit_mounting_correction(sweep_rows)
            print(f"sessie-correctie gefit: rest-fout gemiddeld {mean_e:.2f} "
                  f"deg, max {max_e:.2f} deg (op de sweep zelf)")
            if mean_e > 8.0:
                print("  WAARSCHUWING: deze fit is zelf al slecht -- de "
                      "correctie hieronder is dan niet te vertrouwen; check "
                      "bedrading/montage voor je de grid-resultaten gebruikt")
        except ValueError as exc:
            print(f"kon geen sessie-correctie fitten ({exc}), grid-test "
                  "gaat door zonder corrected_theta_h/v_deg")

    if args.self_calibrate and args.axis_sweep:
        # standalone calibrated-sweep mode: the reported sweep IS the
        # calibration sweep (same points, same zero-session) -- it was
        # already driven and measured above, so reuse that instead of
        # sending the whole sweep through the motors a second time
        rows = [r for r in sweep_rows if r["kind"] in ("sweep_iq1", "sweep_iq2")]
    else:
        drive_and_measure(ser, rows, m, comp, q_home,
                          consensus_reads=args.consensus_reads,
                          combined_settle_extra=args.combined_settle_extra,
                          start=cur)
    ser.close()

    if R_session is not None:
        for r in rows:
            if r["measured_theta_h_deg"] in ("", None):
                continue
            ch, cv = apply_mounting_correction(
                R_session, float(r["measured_theta_h_deg"]),
                float(r["measured_theta_v_deg"]))
            r["corrected_theta_h_deg"] = f"{ch:.3f}"
            r["corrected_theta_v_deg"] = f"{cv:.3f}"

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\ngeschreven: {args.csv}")

    if args.noise_floor:
        analyse_noise_floor(rows, args)
    elif args.round_trip:
        analyse_round_trip(rows, args)
    elif args.axis_sweep:
        analyse_axis_sweep(rows, args)
    else:
        analyse(rows, args)


if __name__ == "__main__":
    main()
