"""hole_home_check.py -- where does the usable region actually sit?

cone_deg is the largest cone that fits ANYWHERE in the workspace, and
rect_deg is a box around home, so a band-shaped workspace can score a big
cone that is nowhere near the direction the surgeon starts from. This script
reports, for each design in designs/representatives.json:

  * the angle from home to the centre of the largest inscribed cone
  * the largest cone centred ON home (a home-centred version of cone_deg)
  * how much of the workspace lies within 30 deg of home
  * how much motor travel the usable component needs

    python hole_home_check.py [--json designs/representatives.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from rcm5bar import Linkage
from rcm_workspace import (workspace, hole_solid_angle, target_coverage,
                           _dense, _bins_to_vec, nearest_angle)
from hole_search import NAMES, EVAL_KW

# Human eye rotation range around the rest direction (theory, not a
# mechanism property): about +/-50 deg horizontal, +/-45 deg vertical.
TARGET_H_DEG = 50.0
TARGET_V_DEG = 45.0


def home_metrics_from_result(r, target_h_deg=TARGET_H_DEG,
                             target_v_deg=TARGET_V_DEG):
    """Home-centred descriptors from a workspace(..., return_maps=True) result.

    cone_deg and hole can both be gamed by a workspace that wraps the sphere:
    a band has no enclosed hole and a big cone somewhere along it, while the
    region actually reachable around the surgeon's starting direction stays
    small. These descriptors are anchored on home, so they cannot be.

        home_cone_deg       largest cone CENTRED ON home that stays inside
        cone_off_home_deg   how far the largest inscribed cone sits from home
        omega_within_30deg  workspace solid angle within 30 deg of home
                            (a full 30 deg cap is 0.842 sr)
        target_coverage     fraction of the target_h_deg x target_v_deg eye
                            range around home that is actually reached (see
                            rcm_workspace.target_coverage) -- unlike rect_deg
                            this is scored against the eye's own range, not
                            against whatever box happens to fit
        target_scale        that range's guaranteed hole-free fraction
    """
    m = r["maps"]
    key, step = m["bin_key"], m["bin_step"]
    img, h0, v0 = _dense(key, pad=3)

    fi, fj = np.nonzero(img)
    hi, hj = np.nonzero(~img)
    fpts = _bins_to_vec(fi + h0, fj + v0, step)
    hpts = _bins_to_vec(hi + h0, hj + v0, step)
    ang = nearest_angle(fpts, hpts)                 # clearance of each bin

    # home direction in the bin frame is (0, 0, 1): theta_h = theta_v = 0
    home_vec = np.array([0.0, 0.0, 1.0])
    b = int(np.argmax(ang))
    cone_centre = fpts[b]
    off_home = np.degrees(np.arccos(np.clip(cone_centre @ home_vec, -1, 1)))

    # largest cone centred on home = distance from home to the nearest empty bin
    home_clear = float(nearest_angle(home_vec[None, :], hpts)[0])

    # share of the workspace within 30 deg of home
    vc = (fj + v0 + 0.5) * step
    w = np.cos(vc)
    near = np.degrees(np.arccos(np.clip(fpts @ home_vec, -1, 1))) <= 30.0
    om = float(w.sum()) * step * step
    om_near = float(w[near].sum()) * step * step

    # motor travel the usable component needs
    sel = m["comp_dex"]
    i1 = np.degrees(m["Iq1"][sel]); i2 = np.degrees(m["Iq2"][sel])
    h = hole_solid_angle(r)
    t = target_coverage(r, target_h_deg, target_v_deg)
    return dict(n_input=r["n_input"], omega=om, hole=h["hole_sr"],
                cone_deg=r["cone_deg"], cone_off_home_deg=off_home,
                home_cone_deg=max(0.0, np.degrees(home_clear) - np.degrees(step)),
                omega_within_30deg=om_near,
                frac_within_30deg=om_near / om if om else 0.0,
                Iq1_span=(float(i1.min()), float(i1.max())),
                Iq2_span=(float(i2.min()), float(i2.max())),
                rect_deg=r["rect_deg"], planar_ok=h["planar_ok"],
                max_theta_v_deg=h["max_theta_v_deg"],
                span_h_bins_deg=h["span_h_bins_deg"],
                target_h_deg=target_h_deg, target_v_deg=target_v_deg,
                target_coverage=t["coverage"], target_scale=t["scale"],
                target_reach_h_deg=t["scale"] * target_h_deg,
                target_reach_v_deg=t["scale"] * target_v_deg)


def home_metrics(lk, n_input=601, target_h_deg=TARGET_H_DEG,
                 target_v_deg=TARGET_V_DEG):
    return home_metrics_from_result(
        workspace(lk, return_maps=True, **dict(EVAL_KW, n_input=n_input)),
        target_h_deg=target_h_deg, target_v_deg=target_v_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="designs/representatives.json")
    ap.add_argument("--n-input", type=int, default=601)
    ap.add_argument("--target-h", type=float, default=TARGET_H_DEG,
                    help="eye rotation half-range, horizontal [deg]")
    ap.add_argument("--target-v", type=float, default=TARGET_V_DEG,
                    help="eye rotation half-range, vertical [deg]")
    args = ap.parse_args()

    with open(args.json) as fh:
        reps = json.load(fh)

    ph = dict(l1=0.060 / 0.150, l2=0.080 / 0.150, l3=0.070 / 0.150,
              l4=0.050 / 0.150, l5=0.090 / 0.150, l6=0.050 / 0.150,
              q1=0.50, q4=0.60)
    cases = [("PLACEHOLDER (scaled to h=1)", ph)] + \
            [(r["tag"], {n: r["params"][n] for n in NAMES}) for r in reps]

    for tag, p in cases:
        lk = Linkage(h=1.0, **p)
        d = home_metrics(lk, args.n_input, args.target_h, args.target_v)
        print(f"\n{tag}")
        print(f"  omega {d['omega']:.3f} sr, hole {d['hole']:.3f} sr "
              f"(planar test {'ok' if d['planar_ok'] else 'UNSAFE'}, "
              f"max|theta_v| {d['max_theta_v_deg']:.0f} deg, azimuth span "
              f"{d['span_h_bins_deg']:.0f} deg)")
        print(f"  largest cone {d['cone_deg']:.1f} deg, centred "
              f"{d['cone_off_home_deg']:.1f} deg away from home")
        print(f"  largest cone CENTRED ON HOME {d['home_cone_deg']:.1f} deg; "
              f"rect {d['rect_deg'][0]:.0f} x {d['rect_deg'][1]:.0f} deg")
        print(f"  eye-range target +/-{d['target_h_deg']:.0f} x "
              f"+/-{d['target_v_deg']:.0f} deg: "
              f"{100*d['target_coverage']:.0f}% of that area reached, "
              f"{100*d['target_scale']:.0f}% guaranteed hole-free "
              f"({d['target_reach_h_deg']:.1f} x "
              f"{d['target_reach_v_deg']:.1f} deg)")
        print(f"  workspace within 30 deg of home: "
              f"{d['omega_within_30deg']:.3f} sr "
              f"({100*d['frac_within_30deg']:.0f}% of the total; a full "
              f"30 deg cap is 0.842 sr)")
        print(f"  motor travel used: Iq1 {d['Iq1_span'][0]:+.0f} .. "
              f"{d['Iq1_span'][1]:+.0f} deg, Iq2 {d['Iq2_span'][0]:+.0f} .. "
              f"{d['Iq2_span'][1]:+.0f} deg")


if __name__ == "__main__":
    main()
