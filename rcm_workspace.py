"""
rcm_workspace.py -- workspace metrics for the 5-bar spherical RCM mechanism.

The output of this mechanism is a *direction* (the instrument axis), so the
workspace is a region on the unit sphere and its natural measure is SOLID
ANGLE, in steradians.  That measure is scale invariant: multiplying h and all
li by the same factor changes nothing, so the workspace depends only on the
six length ratios plus q1 and q4.  Eight effective design parameters.

What counts as reachable
------------------------
Not every input pair that closes is usable:

  * the two assembly branches are separated by lam = 0 singularities, so a
    mechanism cannot travel from one to the other without disassembly;
  * regions with a badly conditioned Jacobian need huge motor speeds and
    transmit force poorly, and are useless in practice;
  * the reachable set may have several disconnected islands, and only the one
    containing the home pose can be driven to.

So the workspace is computed as: the connected component, in the (Iq1, Iq2)
input plane, that contains the home pose, restricted to one branch and to
cells whose condition number is below a threshold.  That component is then
mapped to the sphere and its solid angle measured by equal-area binning in
Fick coordinates, where dOmega = cos(theta_v) dtheta_h dtheta_v exactly.

Main entry point: workspace(lk, ...) -> dict of metrics.
For optimisation use one scalar, e.g. result["omega_dex"] or
result["cone_deg"].
"""

from __future__ import annotations

import numpy as np

from rcm5bar import Linkage, _unit

__all__ = ["tool_axis_grid", "workspace", "workspace_score", "fick",
           "hole_solid_angle", "target_coverage"]


# ----------------------------------------------------------------------
# SciPy stand-ins
# ----------------------------------------------------------------------
# SciPy is unavailable in some environments (JupyterLite / Pyodide), so the
# three routines this module needs are written out here in plain NumPy:
#   label_component     replaces ndimage.label + picking the seed's component
#   binary_closing_3x3  replaces ndimage.binary_closing
#   nearest_angle       replaces scipy.spatial.cKDTree
# All three were checked pixel-identical against SciPy before inlining.
# ----------------------------------------------------------------------

def label_component(mask, i0, j0):
    """Boolean image of the 4-connected region of `mask` containing (i0, j0).

    Scanline flood fill: each step swallows a whole horizontal run at once, so
    the Python-level loop runs once per run rather than once per pixel.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    if not mask[i0, j0]:
        return out

    n_rows, n_cols = mask.shape
    stack = [(i0, j0)]
    while stack:
        i, j = stack.pop()
        if out[i, j] or not mask[i, j]:
            continue

        # widen to the full run on this row
        lo = j
        while lo > 0 and mask[i, lo - 1] and not out[i, lo - 1]:
            lo -= 1
        hi = j
        while hi < n_cols - 1 and mask[i, hi + 1] and not out[i, hi + 1]:
            hi += 1
        out[i, lo:hi + 1] = True

        # seed the rows above and below, one seed per contiguous run
        for ii in (i - 1, i + 1):
            if not 0 <= ii < n_rows:
                continue
            row = mask[ii, lo:hi + 1] & ~out[ii, lo:hi + 1]
            if not row.any():
                continue
            # starts of runs: True where row is True and the previous is not
            starts = np.nonzero(row & ~np.concatenate(([False], row[:-1])))[0]
            for s in starts:
                stack.append((ii, lo + int(s)))
    return out


def _shift(a, di, dj, fill):
    """Copy of `a` translated by (di, dj), with `fill` shifted in at the edge."""
    out = np.full_like(a, fill)
    n, m = a.shape
    src_i = slice(max(0, -di), n - max(0, di))
    dst_i = slice(max(0, di), n - max(0, -di))
    src_j = slice(max(0, -dj), m - max(0, dj))
    dst_j = slice(max(0, dj), m - max(0, -dj))
    out[dst_i, dst_j] = a[src_i, src_j]
    return out


def binary_closing_3x3(img):
    """Morphological closing with a full 3x3 structuring element.

    Closing = dilate then erode. It fills isolated holes without growing the
    outer boundary. Both passes treat everything outside the image as False,
    which is what scipy.ndimage.binary_closing does by default; verified
    pixel-identical in the __main__ block.
    """
    img = np.asarray(img, dtype=bool)
    offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]

    dil = np.zeros_like(img)
    for di, dj in offsets:
        dil |= _shift(img, di, dj, False)

    ero = np.ones_like(dil)
    for di, dj in offsets:
        ero &= _shift(dil, di, dj, False)

    return ero


def nearest_angle(fpts, hpts, chunk=2000):
    """Great-circle angle from each row of `fpts` to the nearest row of `hpts`.

    Both arrays hold unit vectors. Chunked brute force: for a few thousand
    points this is comparable to a KD-tree, and it has no dependencies.
    """
    fpts = np.asarray(fpts, float)
    hpts = np.asarray(hpts, float)
    out = np.empty(len(fpts))
    for s in range(0, len(fpts), chunk):
        blk = fpts[s:s + chunk]
        dots = np.clip(blk @ hpts.T, -1.0, 1.0)
        out[s:s + chunk] = np.arccos(dots.max(axis=1))
    return out



# ----------------------------------------------------------------------
# vectorised forward kinematics -> tool axis only
# ----------------------------------------------------------------------
def tool_axis_grid(lk: Linkage, Iq1, Iq2, branch=+1,
                   tool_dir_plate=(0.0, 0.0, 1.0), sing_tol=1e-6,
                   return_points=False):
    """Tool axis for many actuator pairs at once.

    Iq1, Iq2 : broadcastable arrays of the same shape S
    returns  : (axes, valid) with axes shape S+(3,) and valid shape S.
               Invalid entries (singular or unreachable) hold NaN.
               With return_points=True a third item is returned, a dict with
               P2, P3 and P5 of shape S+(3,) -- the moving joints, which
               interference checks need and the tool axis alone cannot give.
    """
    Iq1 = np.asarray(Iq1, float)
    Iq2 = np.asarray(Iq2, float)
    Iq1, Iq2 = np.broadcast_arrays(Iq1, Iq2)
    shape = Iq1.shape

    RCM = lk.RCM
    c1, s1 = np.cos(Iq1), np.sin(Iq1)

    def stack(x, y, z):
        return np.stack(np.broadcast_arrays(x, y, z), axis=-1)

    # --- leg 1 -----------------------------------------------------------
    A_z = stack(np.sin(lk.q1) * c1, np.sin(lk.q1) * s1, np.cos(lk.q1))
    A_y = stack(-s1, c1, np.zeros_like(s1))
    P2 = lk.l1 * A_z

    # --- leg 2 -----------------------------------------------------------
    c2, s2 = np.cos(Iq2), np.sin(Iq2)
    I4x, I4y, I4z = lk.R_I4[:, 0], lk.R_I4[:, 1], lk.R_I4[:, 2]
    I4s_x = c2[..., None] * I4x + s2[..., None] * I4y
    E_z = np.cos(lk.q5) * I4z + np.sin(lk.q5) * I4s_x
    P5 = lk.P4 + lk.l5 * E_z

    # --- the unknown direction RCM -> P3 ---------------------------------
    a = RCM - P2
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    a = -a                                  # a_hat = unit(P2 - RCM)
    c = P5 - RCM
    c = c / np.linalg.norm(c, axis=-1, keepdims=True)

    g = np.sum(a * c, axis=-1)
    good = np.abs(g) < 1.0 - sing_tol
    denom = np.where(good, 1.0 - g * g, 1.0)

    mu = (lk.cos_a - g * lk.cos_c) / denom
    nu = (lk.cos_c - g * lk.cos_a) / denom
    lam2 = (1.0 - mu * mu - nu * nu - 2.0 * mu * nu * g) / denom
    good &= lam2 >= 0.0
    lam = branch * np.sqrt(np.where(good, lam2, 0.0))

    b = (mu[..., None] * a + nu[..., None] * c
         + lam[..., None] * np.cross(a, c))
    P3 = RCM + lk.l3 * b

    # --- plate frame and tool axis ---------------------------------------
    pz = 0.5 * (P3 + P5) - RCM
    nz = np.linalg.norm(pz, axis=-1, keepdims=True)
    good &= nz[..., 0] > 1e-12
    pz = pz / np.where(nz > 1e-12, nz, 1.0)

    px = P5 - P3
    px = px / np.linalg.norm(px, axis=-1, keepdims=True)
    py = np.cross(pz, px)

    d = np.asarray(tool_dir_plate, float)
    axes = d[0] * px + d[1] * py + d[2] * pz
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)

    axes = np.where(good[..., None], axes, np.nan)
    if return_points:
        pts = {"P2": np.broadcast_to(P2, shape + (3,)).copy(),
               "P3": P3.reshape(shape + (3,)),
               "P5": np.broadcast_to(P5, shape + (3,)).copy()}
        return axes.reshape(shape + (3,)), good.reshape(shape), pts
    return axes.reshape(shape + (3,)), good.reshape(shape)


# ----------------------------------------------------------------------
# Fick decomposition, vectorised
# ----------------------------------------------------------------------
def fick(axes, home=(0.0, 0.0, -1.0), right=(1.0, 0.0, 0.0)):
    """(theta_h, theta_v) in radians for an array of unit axes."""
    z_m = _unit(home)
    x_m = np.asarray(right, float)
    x_m = _unit(x_m - np.dot(x_m, z_m) * z_m)
    y_m = np.cross(z_m, x_m)
    M = np.column_stack((x_m, y_m, z_m))

    t = np.asarray(axes, float) @ M          # = M.T @ t per row
    th_h = np.arctan2(t[..., 0], t[..., 2])
    th_v = np.arcsin(np.clip(t[..., 1], -1.0, 1.0))
    return th_h, th_v


# ----------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------
def workspace(lk: Linkage, branch=+1, home_input=(0.0, 0.0),
              Iq1_range=(-np.pi, np.pi), Iq2_range=(-np.pi, np.pi),
              n_input=301, bin_deg=1.0, kappa_max=8.0,
              tool_dir_plate=(0.0, 0.0, 1.0), right=(1.0, 0.0, 0.0),
              fd_step=1e-4, wrap=None, fill_pinholes=True, return_maps=False,
              pose_mask=None):
    """Measure the usable workspace of one linkage.

    Parameters
    ----------
    lk            Linkage
    branch        assembly mode to analyse (+1 or -1)
    home_input    (Iq1, Iq2) defining the home pose. The reported workspace is
                  the connected region containing it, and the Fick angles are
                  zeroed on its tool axis.
    Iq1_range,
    Iq2_range     motor travel limits. Narrow these to your real actuators.
    n_input       samples per axis over the input ranges
    bin_deg       sphere bin size in degrees
    kappa_max     largest tolerated Jacobian condition number. Set to np.inf
                  to measure the raw reachable set instead of the dexterous one.
    fd_step       finite-difference step for the Jacobian [rad]
    wrap          (bool, bool) whether each input axis wraps for connectivity.
                  Defaults to True on any axis spanning a full turn.
    return_maps   also return the raw grids, for plotting

    Returns
    -------
    dict with, among others:
        omega_dex    solid angle of the usable workspace [sr]   <- optimise this
        omega_raw    solid angle ignoring the conditioning filter [sr]
        cone_deg     half-angle of the largest cone fully inside the
                     workspace [deg]                            <- or this
        rect_deg     (half-width_h, half-width_v) of the largest axis-aligned
                     rectangle in Fick coordinates centred on home [deg]
        gci          global conditioning index, area-weighted mean of 1/kappa
        kappa_median median condition number over the workspace
        span_h_deg,
        span_v_deg   bounding extents [deg]
        home_axis    tool axis at the home pose
        frac_solved  fraction of sampled input cells that closed at all
    """
    a1 = np.linspace(Iq1_range[0], Iq1_range[1], n_input)
    a2 = np.linspace(Iq2_range[0], Iq2_range[1], n_input)
    G1, G2 = np.meshgrid(a1, a2, indexing="ij")

    if wrap is None:
        wrap = (np.isclose(Iq1_range[1] - Iq1_range[0], 2 * np.pi),
                np.isclose(Iq2_range[1] - Iq2_range[0], 2 * np.pi))

    axes, ok = tool_axis_grid(lk, G1, G2, branch, tool_dir_plate)

    # Poses the mechanism can assemble but must not use -- links behind the
    # eye, links hitting each other. Applied HERE, before the connected
    # component is taken, so that a forbidden band splits the workspace
    # instead of being quietly counted in it.
    if pose_mask is not None:
        ok = ok & pose_mask(lk, G1, G2, branch)

    # --- Jacobian by central differences, in the tangent plane at t -------
    kappa = np.full(G1.shape, np.inf)
    with np.errstate(invalid="ignore"):
        ap1, o1 = tool_axis_grid(lk, G1 + fd_step, G2, branch, tool_dir_plate)
        am1, o2 = tool_axis_grid(lk, G1 - fd_step, G2, branch, tool_dir_plate)
        ap2, o3 = tool_axis_grid(lk, G1, G2 + fd_step, branch, tool_dir_plate)
        am2, o4 = tool_axis_grid(lk, G1, G2 - fd_step, branch, tool_dir_plate)
    fd_ok = ok & o1 & o2 & o3 & o4

    d1 = (ap1 - am1) / (2 * fd_step)
    d2 = (ap2 - am2) / (2 * fd_step)
    # project onto the tangent plane (removes the radial component)
    d1 = d1 - np.sum(d1 * axes, axis=-1, keepdims=True) * axes
    d2 = d2 - np.sum(d2 * axes, axis=-1, keepdims=True) * axes

    # singular values of the 2x3 stack via the 2x2 Gram matrix
    a11 = np.sum(d1 * d1, axis=-1)
    a22 = np.sum(d2 * d2, axis=-1)
    a12 = np.sum(d1 * d2, axis=-1)
    tr = a11 + a22
    det = a11 * a22 - a12 * a12
    disc = np.clip(tr * tr / 4 - det, 0.0, None)
    ev_hi = tr / 2 + np.sqrt(disc)
    ev_lo = tr / 2 - np.sqrt(disc)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.sqrt(np.where(ev_lo > 1e-18, ev_hi / ev_lo, np.inf))
    kappa = np.where(fd_ok, k, np.inf)
    sigma_min = np.where(fd_ok, np.sqrt(np.clip(ev_lo, 0, None)), 0.0)

    # --- connected component containing home ------------------------------
    i0 = int(np.argmin(np.abs(a1 - home_input[0])))
    j0 = int(np.argmin(np.abs(a2 - home_input[1])))

    def component(mask):
        m = mask.copy()
        if not m[i0, j0]:
            return np.zeros_like(m)
        # emulate wraparound by tiling the wrapped axes before labelling
        pad_i, pad_j = (1 if wrap[0] else 0), (1 if wrap[1] else 0)
        big = np.tile(m, (1 + 2 * pad_i, 1 + 2 * pad_j))
        ci = pad_i * m.shape[0] + i0
        cj = pad_j * m.shape[1] + j0
        sel = label_component(big, ci, cj)
        # fold the tiles back together
        out = np.zeros_like(m)
        for bi in range(1 + 2 * pad_i):
            for bj in range(1 + 2 * pad_j):
                out |= sel[bi * m.shape[0]:(bi + 1) * m.shape[0],
                           bj * m.shape[1]:(bj + 1) * m.shape[1]]
        return out

    comp_raw = component(ok)
    comp_dex = component(ok & (kappa <= kappa_max))

    home_axis = axes[i0, j0]
    if not np.all(np.isfinite(home_axis)):
        raise ValueError("the home pose itself does not close on this branch")

    th_h, th_v = fick(axes, home=home_axis, right=right)

    # --- equal-area binning in Fick coordinates ---------------------------
    step = np.radians(bin_deg)

    def binned(mask):
        """Bin indices hit by a mask, with pinholes from finite input
        sampling optionally closed."""
        if not mask.any():
            return np.zeros((0, 2), int), np.array([]), np.array([]), 0
        hh = th_h[mask]
        vv = th_v[mask]
        ih = np.floor(hh / step).astype(int)
        iv = np.floor(vv / step).astype(int)
        key = np.unique(np.stack([ih, iv], axis=1), axis=0)
        n_before = len(key)
        if fill_pinholes and n_before:
            img, h0, v0 = _dense(key)
            closed = binary_closing_3x3(img)
            fi, fj = np.nonzero(closed)
            key = np.stack([fi + h0, fj + v0], axis=1)
        return key, hh, vv, n_before

    def omega_of(key):
        if not len(key):
            return 0.0
        vc = (key[:, 1] + 0.5) * step
        return float(np.sum(np.cos(vc)) * step * step)

    key_raw, _, _, _ = binned(comp_raw)
    key, hh, vv, n_before = binned(comp_dex)
    omega_raw = omega_of(key_raw)
    omega_dex = omega_of(key)

    # --- shape descriptors -------------------------------------------------
    if len(key):
        span_h = float(hh.max() - hh.min())
        span_v = float(vv.max() - vv.min())
        img, h0, v0 = _dense(key)
        cone = _inscribed_cone(img, h0, v0, step)
        rect = _centred_rect(img, h0, v0, step)
        spb = float(comp_dex.sum()) / max(n_before, 1)
        holes = 1.0 - n_before / max(len(key), 1)
    else:
        span_h = span_v = cone = 0.0
        rect = (0.0, 0.0)
        spb = 0.0
        holes = 0.0

    kk = kappa[comp_dex]
    gci = float(np.mean(1.0 / kk)) if kk.size else 0.0

    res = {
        "omega_dex": omega_dex,
        "omega_raw": omega_raw,
        "omega_dex_deg2": omega_dex * (180 / np.pi) ** 2,
        "cone_deg": np.degrees(cone),
        "equiv_cone_deg": float(np.degrees(np.arccos(
            np.clip(1 - omega_dex / (2 * np.pi), -1, 1)))),
        "rect_deg": (np.degrees(rect[0]), np.degrees(rect[1])),
        "span_h_deg": np.degrees(span_h),
        "span_v_deg": np.degrees(span_v),
        "gci": gci,
        "kappa_median": float(np.median(kk)) if kk.size else np.inf,
        "sigma_min_min": float(sigma_min[comp_dex].min()) if kk.size else 0.0,
        "frac_solved": float(ok.mean()),
        "frac_dexterous": float(comp_dex.mean()),
        "samples_per_bin": spb,
        "pinhole_fraction": holes,
        "home_axis": home_axis,
        "n_input": n_input, "bin_deg": bin_deg, "kappa_max": kappa_max,
        "branch": branch,
    }
    if spb < 5.0:
        res["WARNING"] = (f"only {spb:.1f} input samples per sphere bin; "
                          "raise n_input or bin_deg before trusting the area")
    if return_maps:
        res["maps"] = {
            "Iq1": G1, "Iq2": G2, "ok": ok, "kappa": kappa,
            "sigma_min": sigma_min, "comp_dex": comp_dex, "comp_raw": comp_raw,
            "theta_h": th_h, "theta_v": th_v, "axes": axes,
            "bin_key": key, "bin_step": step,
        }
    return res


def _dense(key, pad=3):
    """Boolean image of the bin set over its padded bounding box."""
    h0, h1 = key[:, 0].min() - pad, key[:, 0].max() + pad
    v0, v1 = key[:, 1].min() - pad, key[:, 1].max() + pad
    img = np.zeros((h1 - h0 + 1, v1 - v0 + 1), bool)
    img[key[:, 0] - h0, key[:, 1] - v0] = True
    return img, h0, v0


def _bins_to_vec(ih, iv, step):
    hh = (ih + 0.5) * step
    vv = (iv + 0.5) * step
    return np.stack([np.cos(vv) * np.sin(hh), np.sin(vv),
                     np.cos(vv) * np.cos(hh)], axis=-1)


def _inscribed_cone(img, h0, v0, step):
    """Largest half-angle cone whose spherical cap lies inside the bin set.

    Distances are true angles on the sphere (a KD-tree on the 3D bin
    directions), not Euclidean distances in the index grid, so the result is
    correct even where cos(theta_v) departs from 1.
    """
    fi, fj = np.nonzero(img)
    hi, hj = np.nonzero(~img)
    if fi.size == 0:
        return 0.0
    if hi.size == 0:
        return np.pi

    fpts = _bins_to_vec(fi + h0, fj + v0, step)
    hpts = _bins_to_vec(hi + h0, hj + v0, step)

    ang = nearest_angle(fpts, hpts)
    return max(0.0, float(ang.max()) - step)


def _box_sum(ii, shape, a, b, c, d):
    """Sum of a boolean image's integral image `ii` over the inclusive box
    rows [a, b], cols [c, d]; -1 if the box runs outside the image."""
    if a < 0 or c < 0 or b >= shape[0] or d >= shape[1]:
        return -1
    s = ii[b, d]
    if a > 0:
        s -= ii[a - 1, d]
    if c > 0:
        s -= ii[b, c - 1]
    if a > 0 and c > 0:
        s += ii[a - 1, c - 1]
    return int(s)


def _centred_rect(img, h0, v0, step, max_bins=400):
    """Largest Fick rectangle centred on the home pose (bin index 0,0) that is
    fully covered. Returns (half-width_h, half-width_v) in radians."""
    ii = np.cumsum(np.cumsum(img.astype(np.int64), axis=0), axis=1)

    best, best_area = (0.0, 0.0), 0.0
    for n in range(1, max_bins):
        a, b = -n - h0, n - 1 - h0
        if a < 0 or b >= img.shape[0]:
            break
        for m in range(1, max_bins):
            c, d = -m - v0, m - 1 - v0
            if c < 0 or d >= img.shape[1]:
                break
            if _box_sum(ii, img.shape, a, b, c, d) != (2 * n) * (2 * m):
                break
            area = (2 * n * step) * (2 * np.sin(m * step))
            if area > best_area:
                best_area, best = area, (n * step, m * step)
    return best


def hole_solid_angle(result, pad=4):
    """Solid angle of the enclosed holes in the workspace, in steradians.

    `result` must come from workspace(..., return_maps=True); the measurement
    is made on exactly the bin set whose area is reported as omega_dex, so
    hole and workspace are always commensurate.

    Method
    ------
    The bin image is padded by `pad` empty bins on every side, so its border
    is guaranteed empty. A 4-connected flood fill of the empty bins started at
    a corner labels the OUTSIDE. Empty bins the fill never reaches are
    enclosed by workspace on all sides: those are the holes. Each is weighted
    by cos(theta_v) exactly as omega_of does, so a hole solid angle and
    omega_dex add up to the area of the filled outline.

    A hole here is a region the linkage never maps onto -- not a region where
    it fails to close -- so this is a property of the mapping, not of
    feasibility.

    Note: the bins live in Fick coordinates and are treated as a PLANE. The
    solid angles are exact either way, but "enclosed" is a statement about
    connectivity, and the plane is only the sphere's topology while the
    workspace stays away from the Fick poles (theta_v = +/-90 deg) and does
    not wrap around theta_h = +/-180 deg. A workspace that wraps is a band on
    the sphere with no hole in the topological sense, yet its planar image can
    look either holed or not depending on where the cut falls. Both conditions
    are tested and reported in `planar_ok` / `warning`; a hole number carrying
    a warning should not be compared with one that does not.

    Returns
    -------
    dict with keys
        hole_sr        total enclosed-hole solid angle [sr]
        n_holes        number of separate enclosed holes
        home_in_hole   True if the home bin (0, 0) lies in a hole
        largest_hole_sr  solid angle of the biggest single hole [sr]
        hole_bins      number of empty bins enclosed
        hole_fraction  hole_sr / (hole_sr + omega_dex)
        max_theta_v_deg  how close the workspace gets to the Fick pole
        span_h_bins_deg  azimuth extent of the occupied bins
        planar_ok      False if the planar treatment above is unsafe
        warning        why, when planar_ok is False
    """
    if "maps" not in result:
        raise ValueError("call workspace(..., return_maps=True) first")
    key = result["maps"]["bin_key"]
    step = result["maps"]["bin_step"]

    empty = {"hole_sr": 0.0, "n_holes": 0, "home_in_hole": False,
             "largest_hole_sr": 0.0, "hole_bins": 0, "hole_fraction": 0.0,
             "max_theta_v_deg": 0.0, "span_h_bins_deg": 0.0,
             "planar_ok": True, "warning": ""}
    if not len(key):
        return empty

    # --- is the planar treatment of the bin image safe here? --------------
    vv = (key[:, 1] + 0.5) * step
    hh = (key[:, 0] + 0.5) * step
    max_v = float(np.degrees(np.abs(vv).max()))
    span_h = float(np.degrees(hh.max() - hh.min()))
    why = []
    if max_v > 80.0:
        why.append(f"workspace reaches theta_v = {max_v:.0f} deg, where Fick "
                   "bins degenerate at the pole")
    if span_h > 350.0:
        why.append(f"workspace spans {span_h:.0f} deg of azimuth, so it wraps "
                   "and 'enclosed' is not well defined in the plane")
    diag = {"max_theta_v_deg": max_v, "span_h_bins_deg": span_h,
            "planar_ok": not why, "warning": "; ".join(why)}
    empty.update(diag)

    img, h0, v0 = _dense(key, pad=pad)
    free = ~img                                   # empty bins
    outside = label_component(free, 0, 0)         # the padded corner
    holes = free & ~outside
    if not holes.any():
        return empty

    def omega_of_mask(mask):
        _, hj = np.nonzero(mask)
        vc = (hj + v0 + 0.5) * step
        return float(np.sum(np.cos(vc)) * step * step)

    # split into separate holes, largest first
    per_hole = []
    todo = holes.copy()
    while todo.any():
        i0, j0 = (int(x[0]) for x in np.nonzero(todo))
        one = label_component(todo, i0, j0)
        per_hole.append(omega_of_mask(one))
        todo &= ~one
    per_hole.sort(reverse=True)

    hi, hj = -h0, -v0                             # home is bin (0, 0)
    in_range = 0 <= hi < holes.shape[0] and 0 <= hj < holes.shape[1]
    home_in_hole = bool(in_range and holes[hi, hj])

    hole_sr = float(sum(per_hole))
    om = result.get("omega_dex", 0.0)
    return {
        "hole_sr": hole_sr,
        "n_holes": len(per_hole),
        "home_in_hole": home_in_hole,
        "largest_hole_sr": per_hole[0],
        "hole_bins": int(holes.sum()),
        "hole_fraction": hole_sr / (hole_sr + om) if hole_sr + om > 0 else 0.0,
        **diag,
    }


def target_coverage(result, target_h_deg=50.0, target_v_deg=45.0):
    """How much of the physiological eye-rotation range the workspace covers.

    The human eye itself rotates about +/-50 deg horizontally and +/-45 deg
    vertically around its rest direction (theory, not a property of any
    mechanism -- see e.g. standard oculomotor range figures). `rect_deg`
    reports the largest box that HAPPENS to be fully covered, so a design can
    win on area without matching that shape at all: a 70x20 deg box scores
    more area than 50x45 while covering less of what the eye actually needs.
    This instead fixes the box at the physiological target and measures how
    much of THAT the mechanism reaches, so a shape mismatch can no longer
    look like a better design than one actually built for the task.

    `result` must come from workspace(..., return_maps=True).

    Returns
    -------
    dict with
        target_sr    solid angle of the full target box [sr], fixed by the
                     target angles alone
        covered_sr   dexterous solid angle inside that box [sr]
        coverage     covered_sr / target_sr, in [0, 1]           <- optimise this
        scale        largest s in [0, 1] such that the box scaled by s
                     (s*target_h_deg, s*target_v_deg) is entirely free of
                     holes or gaps -- the guaranteed range, where `coverage`
                     gives partial credit for a patchy box
    """
    if "maps" not in result:
        raise ValueError("call workspace(..., return_maps=True) first")
    key, step = result["maps"]["bin_key"], result["maps"]["bin_step"]
    th, tv = np.radians(target_h_deg), np.radians(target_v_deg)
    target_sr = 4.0 * th * np.sin(tv)

    if not len(key):
        return {"target_sr": target_sr, "covered_sr": 0.0, "coverage": 0.0,
                "scale": 0.0}

    img, h0, v0 = _dense(key)

    # --- partial credit: dexterous area actually inside the target box ----
    hh = (np.arange(img.shape[0]) + h0 + 0.5) * step
    vv = (np.arange(img.shape[1]) + v0 + 0.5) * step
    in_target = (np.abs(hh) <= th)[:, None] & (np.abs(vv) <= tv)[None, :]
    covered = img & in_target
    covered_sr = float(np.sum(np.cos(vv)[None, :] * covered) * step * step)

    # --- guaranteed credit: largest hole-free box, scaled towards target --
    n_max = max(1, int(np.ceil(th / step)))
    m_max = max(1, int(np.ceil(tv / step)))
    ii = np.cumsum(np.cumsum(img.astype(np.int64), axis=0), axis=1)
    scale = 0.0
    for n in range(1, n_max + 1):
        a, b = -n - h0, n - 1 - h0
        if a < 0 or b >= img.shape[0]:
            break
        for m in range(1, m_max + 1):
            c, d = -m - v0, m - 1 - v0
            if c < 0 or d >= img.shape[1]:
                break
            if _box_sum(ii, img.shape, a, b, c, d) != (2 * n) * (2 * m):
                break
            scale = max(scale, min(n / n_max, m / m_max))

    coverage = covered_sr / target_sr if target_sr > 0 else 0.0
    return {"target_sr": target_sr, "covered_sr": covered_sr,
            # binning can push a fully-covered box slightly past 1.0 at the
            # boundary bins; clip since "more than the whole target" is not
            # a meaningful reading
            "coverage": min(1.0, coverage), "scale": scale}


def workspace_score(params, base: Linkage = None, metric="omega_dex",
                    target_h_deg=50.0, target_v_deg=45.0, **kw):
    """Thin wrapper for optimisers: params -> scalar to MAXIMISE.

    params = (l1, l2, l3, l4, l5, l6, q1, q4) with h fixed at 1, since the
    metric is scale invariant.
    metric        "omega_dex" (default, raw usable solid angle) or
                  "target_coverage" (fraction of the physiological
                  target_h_deg x target_v_deg range around home that is
                  reached -- optimise this to shape the workspace at the
                  eye's own rotation range instead of maximising raw area)
    Returns 0.0 for any infeasible design instead of raising, so a
    derivative-free optimiser can walk through the infeasible region.
    """
    l1, l2, l3, l4, l5, l6, q1, q4 = params
    try:
        lk = Linkage(h=1.0, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5, l6=l6,
                     q1=q1, q4=q4)
        if metric == "target_coverage":
            r = workspace(lk, return_maps=True, **kw)
            return target_coverage(r, target_h_deg, target_v_deg)["coverage"]
        return workspace(lk, **kw)["omega_dex"]
    except Exception:
        return 0.0
