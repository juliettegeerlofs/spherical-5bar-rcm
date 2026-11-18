"""rcm_collision.py -- interference checks for the 5-bar spherical RCM.

Two constraints that the kinematics alone does not know about:

ACCESS      The eye is only reachable from one side. Everything the linkage
            does must stay inside a cone of half-angle `access_deg` about an
            access axis through the RCM; anything past that is behind the eye,
            in the orbit and the face. The default axis is -N.z, pointing from
            the RCM back towards the base at P1, which is the side the
            instrument comes in from, and the default half-angle 90 deg makes
            it the plain "stay in front of the eye" half-space.

SELF-HIT    Links that do not share a joint must not touch each other. The
            five bars are P1-P2 (l1), P2-P3 (l2), P1-P4 (l4), P4-P5 (l5) and
            the plate chord P3-P5 (l6); five of the ten pairs share a joint by
            construction and are skipped, the other five are checked as
            segment-to-segment distance against a link diameter.

Both are expressed as fractions of h, so the scale invariance of the workspace
metric survives: a design that clears at h = 1 clears at any h, as long as the
bar thickness and the eye scale with it.

Everything here is pure NumPy and vectorises over a grid of poses, so the same
functions serve the single-pose readout in rcm_control.py and a whole-workspace
mask.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LINKS", "NON_ADJACENT", "links_of", "segment_distance",
           "point_segment_distance", "access_angles", "self_clearance",
           "eye_clearance", "check_pose", "grid_masks"]

# (name, endpoint A, endpoint B) -- the order fixes the indices below
LINKS = [("l1  P1-P2", "P1", "P2"),
         ("l2  P2-P3", "P2", "P3"),
         ("l4  P1-P4", "P1", "P4"),
         ("l5  P4-P5", "P4", "P5"),
         ("l6  P3-P5", "P3", "P5")]

# pairs that do NOT share a joint; the other five touch by construction
NON_ADJACENT = [(0, 3), (0, 4), (1, 2), (1, 3), (2, 4)]

# The eye is gripped, not entered. Two variants:
#   with_arm=True   a rigid arm runs from the plate down the gaze axis to the
#                    globe. It is a load-bearing member and collides like any
#                    other, so it is checked as a sixth bar; it is rigid with
#                    the plate (l6), which is why that one pair is exempt.
#   with_arm=False  no separate arm -- P3 and P5 attach directly to the eye
#                    (e.g. custom suction cups), so l3 == eye_radius by
#                    construction. P3 and P5 are then endpoints of l2, l5 AND
#                    l6, and every one of those three necessarily touches the
#                    eye_radius sphere exactly at that shared endpoint --
#                    that is the attachment, not an interference, so all
#                    three (not just l6) are exempt from the eye test. l1
#                    (P1-P2) and l4 (P1-P4) touch neither P3 nor P5 and stay
#                    checked either way.
ARM = "arm plate-eye"
ARM_EXEMPT = {"l6  P3-P5"}
EYE_EXEMPT_ARM = {ARM}
EYE_EXEMPT_SUCTION = {"l2  P2-P3", "l5  P4-P5", "l6  P3-P5"}


def arm_endpoints(rcm, P3, P5, tool_axis, eye_radius):
    """The attachment arm: from the eye surface out to the plate midpoint.

    Inner end sits on the globe at `eye_radius` along the gaze axis -- the arm
    grips the eye there and must not push past it. Outer end is the midpoint
    of the plate chord, where it bolts on.
    """
    rcm = np.asarray(rcm, float)
    inner = rcm + eye_radius * np.asarray(tool_axis, float)
    outer = 0.5 * (np.asarray(P3, float) + np.asarray(P5, float))
    return inner, outer


def links_of(s, eye_radius=None):
    """[(name, A, B)] for one solve() result, as (3,) arrays.

    With `eye_radius` given, the attachment arm is appended.
    """
    out = [(nm, np.asarray(s[a], float), np.asarray(s[b], float))
           for nm, a, b in LINKS]
    if eye_radius is not None:
        out.append((ARM,) + arm_endpoints(s["RCM"], s["P3"], s["P5"],
                                          s["tool_axis"], eye_radius))
    return out


def _pairs(links):
    """Index pairs to test: everything except joints shared by construction."""
    n_bars = len(LINKS)
    out = list(NON_ADJACENT)
    for i in range(n_bars, len(links)):          # the arm, if present
        for j in range(n_bars):
            if links[j][0] not in ARM_EXEMPT:
                out.append((j, i))
    return out


# ----------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------
def segment_distance(p1, q1, p2, q2, eps=1e-12):
    """Shortest distance between segments p1q1 and p2q2, vectorised.

    Standard clamped-parameter solution: solve the unconstrained least
    squares for the two line parameters, clamp into [0, 1], and re-solve the
    other one against the clamp. Inputs broadcast on all leading axes.
    """
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)

    denom = a * e - b * b
    s = np.where(denom > eps, np.clip((b * f - c * e) /
                                      np.where(denom > eps, denom, 1.0),
                                      0.0, 1.0), 0.0)
    t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1.0), 0.0)
    t_c = np.clip(t, 0.0, 1.0)
    s = np.where(a > eps, np.clip((b * t_c - c) / np.where(a > eps, a, 1.0),
                                  0.0, 1.0), 0.0)
    diff = (p1 + s[..., None] * d1) - (p2 + t_c[..., None] * d2)
    return np.linalg.norm(diff, axis=-1)


def point_segment_distance(x, p, q, eps=1e-12):
    """Distance from point x to segment pq, vectorised."""
    x, p, q = (np.asarray(v, float) for v in (x, p, q))
    d = q - p
    dd = np.sum(d * d, -1)
    t = np.where(dd > eps, np.sum((x - p) * d, -1) / np.where(dd > eps, dd, 1.0),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(x - (p + t[..., None] * d), axis=-1)


# ----------------------------------------------------------------------
# the two constraints
# ----------------------------------------------------------------------
def access_angles(pts_a, pts_b, rcm, axis, n_sample=33):
    """Largest angle from the access axis reached along a segment [deg].

    Sampled rather than solved: the maximum of the angle along a straight
    segment can sit anywhere on it, and 33 samples put the error well under a
    tenth of a degree at these link lengths, with no case analysis.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    ts = np.linspace(0.0, 1.0, n_sample)
    worst = None
    for t in ts:
        p = pts_a + t * (pts_b - pts_a) - rcm
        nrm = np.linalg.norm(p, axis=-1)
        cos = np.where(nrm > 1e-15, np.sum(p * axis, -1) /
                       np.where(nrm > 1e-15, nrm, 1.0), 1.0)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        worst = ang if worst is None else np.maximum(worst, ang)
    return worst


def self_clearance(links):
    """[(name_a, name_b, distance)] for every pair that may not touch."""
    out = []
    for i, j in _pairs(links):
        na, a0, a1 = links[i]
        nb, b0, b1 = links[j]
        out.append((na, nb, float(segment_distance(a0, a1, b0, b1))))
    return out


def eye_clearance(links, rcm, eye_radius):
    """[(name, gap)] where gap < 0 means the bar is inside the eyeball."""
    return [(nm, float(point_segment_distance(rcm, a, b)) - eye_radius)
            for nm, a, b in links]


def check_pose(s, access_axis=None, access_deg=90.0,
               link_diameter=0.03, eye_radius=0.08, scale=1.0,
               with_arm=True):
    """Full interference report for one solve() result.

    link_diameter and eye_radius are fractions of h; `scale` is h itself, so
    the same numbers work at any size. `access_axis` defaults to the gaze
    direction of this very pose, which is only right at home -- pass the home
    gaze axis explicitly for anything else, since the accessible side is
    fixed to the head, not to where the eye currently looks.
    """
    rcm = np.asarray(s["RCM"], float)
    thick = link_diameter * scale
    r_eye = eye_radius * scale
    links = links_of(s, r_eye if with_arm else None)
    if access_axis is None:
        access_axis = s["tool_axis"]

    per_link = []
    for nm, a, b in links:
        ang = float(access_angles(a, b, rcm, access_axis))
        per_link.append((nm, ang, ang > access_deg))
    worst_link, worst_ang, _ = max(per_link, key=lambda r: r[1])

    pairs = self_clearance(links)
    tight = min(pairs, key=lambda r: r[2])
    # whichever part is meant to touch the globe is exempt from the eye test:
    # the arm if there is one, otherwise every link ending at P3 or P5 (they
    # sit on the eye directly, e.g. via suction cups -- l3 == eye_radius by
    # construction, so each necessarily touches a sphere of that radius at
    # its P3/P5 endpoint, which is the attachment, not an interference)
    eye_exempt = EYE_EXEMPT_ARM if with_arm else EYE_EXEMPT_SUCTION
    eyes = eye_clearance([l for l in links if l[0] not in eye_exempt], rcm, r_eye)
    eye_tight = min(eyes, key=lambda r: r[1])

    return {
        "access_ok": worst_ang <= access_deg,
        "access_worst_deg": worst_ang,
        "access_worst_link": worst_link,
        "access_margin_deg": access_deg - worst_ang,
        "per_link": per_link,
        "self_ok": tight[2] >= thick,
        "self_gap": tight[2],
        "self_pair": (tight[0], tight[1]),
        "self_limit": thick,
        "pairs": pairs,
        "eye_ok": eye_tight[1] >= 0.0,
        "eye_gap": eye_tight[1],
        "eye_link": eye_tight[0],
        "ok": (worst_ang <= access_deg and tight[2] >= thick
               and eye_tight[1] >= 0.0),
    }


# ----------------------------------------------------------------------
# whole-grid version, for masking a workspace
# ----------------------------------------------------------------------
def grid_links(lk, Iq1, Iq2, branch=+1, eye_radius=0.08, with_arm=True):
    """[(name, A, B)] over a whole grid of motor pairs, plus the valid mask."""
    from rcm_workspace import tool_axis_grid
    axes, ok, pts = tool_axis_grid(lk, Iq1, Iq2, branch, return_points=True)
    shape = pts["P2"].shape
    P = {"P1": np.zeros(shape), "P4": np.broadcast_to(lk.P4, shape).copy(),
         **pts}
    links = [(nm, P[a], P[b]) for nm, a, b in LINKS]
    if with_arm:
        inner, outer = arm_endpoints(lk.RCM, P["P3"], P["P5"], axes,
                                     eye_radius * lk.h)
        links.append((ARM, inner, outer))
    return links, ok, axes


def grid_masks(lk, Iq1, Iq2, branch=+1, access_axis=None,
               access_deg=90.0, link_diameter=0.03, eye_radius=0.08,
               with_arm=True):
    """Interference masks over a grid of motor pairs.

    Returns (access_ok, self_ok, eye_ok, worst_access_deg, min_self_gap), all
    with the shape of the input grid. `access_axis` defaults to the gaze at
    the home input, which is the direction the eye faces when the mechanism
    is neutral, and therefore the side the head leaves open.
    """
    links, ok, axes = grid_links(lk, Iq1, Iq2, branch, eye_radius, with_arm)
    rcm = lk.RCM
    thick = link_diameter * lk.h
    r_eye = eye_radius * lk.h
    if access_axis is None:
        access_axis = home_axis_of(lk, branch)

    worst = None
    for _, a, b in links:
        ang = access_angles(a, b, rcm, access_axis)
        worst = ang if worst is None else np.maximum(worst, ang)

    gap = None
    for i, j in _pairs(links):
        d = segment_distance(links[i][1], links[i][2],
                             links[j][1], links[j][2])
        gap = d if gap is None else np.minimum(gap, d)

    eye_gap = None
    eye_exempt = EYE_EXEMPT_ARM if with_arm else EYE_EXEMPT_SUCTION
    for nm, a, b in links:
        if nm in eye_exempt:                # meant to sit at the eye surface
            continue
        d = point_segment_distance(rcm, a, b) - r_eye
        eye_gap = d if eye_gap is None else np.minimum(eye_gap, d)

    return (ok & (worst <= access_deg), ok & (gap >= thick),
            ok & (eye_gap >= 0.0), worst, gap)


def home_axis_of(lk, branch=+1, home_input=(0.0, 0.0)):
    """Gaze direction at the home pose -- the accessible side of the head.

    The eye is gripped, so the head cannot be assumed to face along -N.z: the
    direction the mechanism holds the eye in at home IS straight ahead, and
    everything more than access_deg away from it is behind the face.
    """
    from rcm_workspace import tool_axis_grid
    axes, ok = tool_axis_grid(lk, np.array([home_input[0]]),
                              np.array([home_input[1]]), branch)
    if not ok[0]:
        raise ValueError("home pose does not close on this branch")
    return axes[0]


def pose_mask(lk, branch=+1, access_axis=None, access_deg=90.0,
              link_diameter=0.03, eye_radius=0.08, with_arm=True):
    """A callable for workspace(pose_mask=...): poses free of interference.

    Applied BEFORE the connected component is found, which is the point: a
    forbidden band can cut the workspace into islands, and only the island
    holding home can actually be driven to. Filtering afterwards would count
    islands the mechanism can never reach.
    """
    axis = access_axis if access_axis is not None else home_axis_of(lk, branch)

    def mask(lk_, Iq1, Iq2, branch_):
        acc, sel, eye, _, _ = grid_masks(lk_, Iq1, Iq2, branch_, axis,
                                         access_deg, link_diameter,
                                         eye_radius, with_arm)
        return acc & sel & eye

    mask.access_axis = axis
    return mask
