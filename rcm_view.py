"""rcm_view.py -- shared viewing helpers for the workspace scripts.

Two things live here because both `test.py` (flip through designs) and
`rcm_control.py` (drive one design with sliders) need them:

  * picking designs out of the scored CSVs
  * drawing the gaze sphere as a real projection of a ball, so that a patch
    45 deg wide looks like what it is instead of like a flat map

The projection matters. A pixel at screen radius r is the gaze direction at
angle theta from home where r = sin(theta) (orthographic), NOT r = theta/90.
That is why the parallels crowd towards the rim: on a ball they do.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from hole_search import NAMES

# v1: the first run, eye entered through an incision, no interference model
# v2: the eye gripped from one side, attachment arm and collisions enforced
# eye: direct suction cups on P3/P5, no arm, l3 free above the eye radius
SOURCES_V1 = ["hole_search_aug.csv", "hole_refine_aug.csv"]
SOURCES_V2 = [os.path.join("results_v2", "search_v2.csv")]
SOURCES_EYE = [os.path.join("results_eye", "search_eye_range.csv")]
SOURCES = SOURCES_V1

# span of the search box, used to judge whether two designs are worth
# looking at separately
SPAN = np.array([1.05] * 6 + [1.30] * 2)


# ----------------------------------------------------------------------
# picking designs
# ----------------------------------------------------------------------
def load_compact(paths=SOURCES):
    """Scored designs whose workspace does not wrap the sphere.

    Reads either run: v1 rows carry `aug_ok`, v2 rows carry `feasible`.
    """
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                good = r.get("aug_ok", r.get("feasible", "0")) == "1"
                if good and int(r["planar_ok"]) == 1 \
                        and float(r["omega_dex"]) > 0:
                    rows.append(r)
    return rows


def rank_key(r, how):
    hc = float(r["home_cone_deg"])
    if how == "home_cone":
        return -hc
    if how == "round":
        eq = float(r["equiv_cone_deg"])
        return -(hc / eq if eq > 0 else 0.0)
    if how == "box":
        return -float(r["rect_area_deg2"])
    if how == "omega":
        return -float(r["omega_dex"])
    raise ValueError(how)


def distinct(rows, top, tol=0.04):
    """Greedily keep designs that differ enough to be worth a separate look.

    Perturbation rounds produce near-clones; without this the first twenty
    pictures are the same picture. Distance is in units of the search box, so
    tol=0.04 means "4% of the range apart in some parameter".
    """
    keep, vecs = [], []
    for r in rows:
        v = np.array([float(r[n]) for n in NAMES]) / SPAN
        if all(np.max(np.abs(v - w)) > tol for w in vecs):
            keep.append(r)
            vecs.append(v)
        if len(keep) >= top:
            break
    return keep


def best_designs(sort="home_cone", top=1, paths=SOURCES):
    rows = load_compact(paths)
    if not rows:
        raise SystemExit("no scored designs found; run hole_search.py and "
                         "hole_augment.py first")
    rows.sort(key=lambda r: rank_key(r, sort))
    return distinct(rows, top)


def params_of(row):
    return {n: float(row[n]) for n in NAMES}


# ----------------------------------------------------------------------
# which collision model / physical scale a design was found under
# ----------------------------------------------------------------------
def constraints_of(results, row=None):
    """Collision-check kwargs implied by which run a design came from.

    v1   no interference model was applied when it was found -- callers
         should use mask=None / skip the check, not this function.
    v2   rigid attachment arm, fixed eye radius: search_v2.CONSTRAINTS.
    eye  direct suction cups, no arm, and a per-design eye radius (l3 is
         free above it, so unlike v2 this is NOT one constant -- it is
         carried in the row itself).
    """
    if results == "v2":
        from search_v2 import CONSTRAINTS
        return dict(CONSTRAINTS)
    if results == "eye":
        if row is None:          # e.g. --params with no CSV row to read
            return dict(access_deg=90.0, link_diameter=0.03,
                       eye_radius=0.08, with_arm=False)
        return dict(access_deg=float(row["access_deg"]),
                   link_diameter=float(row["link_diameter"]),
                   eye_radius=float(row["eye_radius"]), with_arm=False)
    raise ValueError(f"no collision model for results={results!r}")


def resolve_scale(args, results, row, default_h=1.0, default_eye_radius=0.08):
    """Fill in --h / --eye-radius from a loaded eye-range design's own
    physical scale when the caller left them unset (None), since l3 there
    was searched against a real eye_radius_mm at a specific h_mm, not as a
    scale-invariant ratio like every other length. Warns instead of silently
    overriding if the caller pinned --h to something else. Mutates `args`.
    """
    if results == "eye" and row is not None:
        if "h_mm" not in row or "eye_radius_mm" not in row:
            raise SystemExit(
                "this results_eye CSV predates h_mm/eye_radius_mm (l3 was "
                "not yet pinned to a real eye radius when it was written) "
                "-- it is not compatible with the current search_eye_range.py "
                "or this viewer. Re-run: python search_eye_range.py --h-mm "
                "<...> to regenerate it.")
        h_native = float(row["h_mm"]) / 1000.0
        if args.h is None:
            args.h = h_native
            print(f"--h defaulted to this design's own scale: "
                  f"{args.h:.4f} m ({row['h_mm']} mm)")
        elif abs(args.h - h_native) > 1e-9:
            print(f"WARNING: --h={args.h:g} m differs from the "
                  f"{h_native:.4f} m this design's l3 was actually searched "
                  "against; the eye-radius margin may no longer hold at "
                  "this scale")
        if args.eye_radius is None:
            args.eye_radius = float(row["eye_radius_mm"]) / (args.h * 1000.0)
    if args.h is None:
        args.h = default_h
    if args.eye_radius is None:
        args.eye_radius = default_eye_radius


# ----------------------------------------------------------------------
# the ball projection
# ----------------------------------------------------------------------
def screen_radius(deg, projection="ortho", dist=6.0):
    """Where an angle from home lands on the picture, as a fraction of the rim.

    ortho        r = sin(theta): the sphere seen from infinitely far. The rim
                 is exactly 90 deg, so the whole visible hemisphere fits.
    perspective  a camera `dist` eye-radii away. Compression is stronger, but
                 the limb moves IN to arccos(1/dist) -- at 6 radii you see out
                 to 80.4 deg only, and the rest is behind the curve.
    """
    th = np.radians(np.asarray(deg, float))
    if projection == "ortho":
        return np.sin(th)
    lim = np.arccos(1.0 / dist)
    u = np.sin(th) / (dist - np.cos(th))
    u_max = np.sin(lim) / (dist - np.cos(lim))
    return np.where(th <= lim, u / u_max, np.nan)


def theta_of_radius(r, projection="ortho", dist=6.0):
    """Inverse of screen_radius: angle from home at a screen radius in [0,1]."""
    if projection == "ortho":
        return np.arcsin(np.clip(r, 0.0, 1.0))
    lim = np.arccos(1.0 / dist)
    u_max = np.sin(lim) / (dist - np.cos(lim))
    u = np.clip(r, 0.0, 1.0) * u_max
    return (np.arcsin(np.clip(u * dist / np.sqrt(1 + u * u), -1, 1))
            - np.arctan(u))


def project_direction(t, projection="ortho", dist=6.0):
    """Unit direction in the home frame -> (x, y, visible) on the picture."""
    t = np.asarray(t, float)
    tz = float(t[2])
    theta = np.arccos(np.clip(tz, -1.0, 1.0))
    psi = np.arctan2(t[1], t[0])
    if projection == "ortho":
        vis = tz >= 0.0
        rad = np.sin(theta)
    else:
        vis = np.degrees(theta) <= np.degrees(np.arccos(1.0 / dist))
        rad = float(screen_radius(np.degrees(theta), projection, dist)) \
            if vis else np.nan
    return (rad * np.cos(psi), rad * np.sin(psi), bool(vis)) if np.isfinite(rad) \
        else (np.nan, np.nan, False)


def ball_image(d, n=560, projection="ortho", dist=6.0,
               sclera=(0.95, 0.94, 0.92), patch=(0.16, 0.50, 0.72)):
    """The workspace painted on the eyeball, seen down the home direction.

    `d` carries the bin image: keys img, h0, v0, step. Shading follows
    cos(theta) so the ball reads as a ball rather than a flat disc.
    """
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g, indexing="xy")
    rad = np.hypot(X, Y)
    on_ball = rad <= 1.0

    theta = theta_of_radius(rad, projection, dist)
    psi = np.arctan2(Y, X)
    st, ct = np.sin(theta), np.cos(theta)
    tx, ty, tz = st * np.cos(psi), st * np.sin(psi), ct

    th_h = np.arctan2(tx, tz)
    th_v = np.arcsin(np.clip(ty, -1.0, 1.0))
    ih = np.floor(th_h / d["step"]).astype(int) - d["h0"]
    iv = np.floor(th_v / d["step"]).astype(int) - d["v0"]
    ok = (ih >= 0) & (ih < d["img"].shape[0]) & \
         (iv >= 0) & (iv < d["img"].shape[1])
    inside = np.zeros_like(on_ball)
    inside[ok] = d["img"][ih[ok], iv[ok]]
    inside &= on_ball

    rgba = np.zeros(X.shape + (4,))
    rgba[on_ball] = list(sclera) + [1.0]
    rgba[inside] = list(patch) + [1.0]
    shade = (0.45 + 0.55 * np.clip(ct, 0.0, 1.0))[..., None]
    rgba[..., :3] *= np.where(on_ball[..., None], shade, 1.0)
    return rgba


def draw_graticule(ax, projection="ortho", dist=6.0, pe=None):
    """Parallels every 15 deg and meridians every 30 deg, on the ball."""
    t = np.linspace(0, 2 * np.pi, 360)
    for deg in (15, 30, 45, 60, 75, 90):
        rad = float(screen_radius(deg, projection, dist))
        if not np.isfinite(rad):
            continue
        rim = deg == 90 or np.isclose(rad, 1.0)
        ax.plot(rad * np.cos(t), rad * np.sin(t), color="k" if rim else "w",
                lw=1.2 if rim else .8, alpha=.9 if rim else .65, zorder=3)
        lab_r, lab_a = (1.07, 35) if rim else (rad, 35 if deg != 75 else 55)
        ax.text(lab_r * np.cos(np.radians(lab_a)),
                lab_r * np.sin(np.radians(lab_a)),
                f"{deg}" + ("$^\\circ$" if rim else ""),
                color="k" if rim else "w", fontsize=7.5, ha="center",
                va="center", alpha=.95, zorder=4,
                path_effects=None if (rim or pe is None) else
                [pe.withStroke(linewidth=1.6, foreground="0.25")])
    r_edge = 1.0
    for a in range(0, 360, 30):
        ax.plot([0, r_edge * np.cos(np.radians(a))],
                [0, r_edge * np.sin(np.radians(a))],
                color="w", lw=.6, alpha=.45, zorder=3)
    ax.set_xlim(-1.16, 1.16)
    ax.set_ylim(-1.16, 1.16)
    ax.set_aspect("equal")
    ax.set_axis_off()


def home_frame(home_axis, right=(1.0, 0.0, 0.0)):
    """Matrix whose columns are the picture axes, same convention as fick()."""
    z_m = np.asarray(home_axis, float)
    z_m = z_m / np.linalg.norm(z_m)
    x_m = np.asarray(right, float)
    x_m = x_m - np.dot(x_m, z_m) * z_m
    if np.linalg.norm(x_m) < 1e-9:
        x_m = np.cross(z_m, [0.0, 1.0, 0.0])
    x_m = x_m / np.linalg.norm(x_m)
    y_m = np.cross(z_m, x_m)
    return np.column_stack((x_m, y_m, z_m))
