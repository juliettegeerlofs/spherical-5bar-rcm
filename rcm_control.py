"""rcm_control.py -- drive one design by hand and watch the gaze move.

Takes the best design found by the search (or any design you name), puts the
two motor angles on sliders, and shows live what the mechanism does:

    left    the reachable gaze region on the eyeball, with a red dot at the
            CURRENT gaze direction. Drag the sliders and the dot moves.
    middle  the input plane (Iq1, Iq2), coloured by condition number, with the
            usable region outlined in red and a crosshair at the current pair.
            Click anywhere in this panel to jump the mechanism there.
    right   the linkage itself in 3D at the current pose, with the eye drawn
            at the RCM and the instrument axis as a line through it. Drag to
            rotate the view.

The readout under the sliders gives theta_h, theta_v, roll, the condition
number kappa at that pose, and whether the pose is inside the usable
workspace.

Controls
--------
    sliders                 Iq1, Iq2
    click in input plane    jump to that motor pair
    arrow keys              nudge 1 deg (shift: 5 deg)
    h                       back to the home pose
    b                       switch assembly branch (recomputes the workspace)
    s                       save a PNG
    q                       quit

    python rcm_control.py                        # best design by home cone
    python rcm_control.py --design 3 --sort round
    python rcm_control.py --h 0.150              # your real scale, in metres
    python rcm_control.py --params 1.0758,0.8314,0.7033,0.6617,0.8432,1.0742,0.3132,0.2091

`--h` only scales the drawing and the reported lengths; the angles and the
workspace are scale invariant. The lengths you pass with --params are ratios
at h = 1 and are multiplied by --h internally, which is exactly the scaling
step that is easy to get wrong by hand.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from rcm5bar import Linkage, solve, RCMError, eye_angles
from rcm_workspace import workspace, tool_axis_grid, _dense, target_coverage
from hole_search import NAMES, EVAL_KW
from hole_home_check import TARGET_H_DEG, TARGET_V_DEG
from rcm_view import (best_designs, params_of, ball_image, draw_graticule,
                      project_direction, home_frame, resolve_scale,
                      SOURCES_V1, SOURCES_V2, SOURCES_EYE)
from rcm_collision import (check_pose, grid_masks, home_axis_of,
                           arm_endpoints, LINKS, ARM)


# ----------------------------------------------------------------------
# model side
# ----------------------------------------------------------------------
def measure(lk, n_input, branch, mask=None):
    """Workspace maps for one branch, plus the bin image for the ball view.

    `mask`, when given, is a workspace(pose_mask=...) callable: poses behind
    the eye or self-hitting are then excluded BEFORE the connected component
    is taken, so the reported/plotted region is what the mechanism can
    actually be driven to, not just what closes kinematically. With the mask
    on, that home-connected region can legitimately be empty (the branch is
    fully blocked from home), which is real information, not an error --
    _dense() only handles a non-empty bin set, so that case gets a 1x1
    all-empty image instead of a crash, and the rest of the UI already
    renders an all-False bin image as "nothing reachable".
    """
    r = workspace(lk, return_maps=True, pose_mask=mask,
                  **dict(EVAL_KW, n_input=n_input, branch=branch))
    key = r["maps"]["bin_key"]
    if len(key):
        img, h0, v0 = _dense(key, pad=1)
    else:
        img, h0, v0 = np.zeros((1, 1), bool), 0, 0
    d = dict(img=img, h0=h0, v0=v0, step=r["maps"]["bin_step"])
    return r, d


def kappa_at(lk, Iq1, Iq2, branch, step=1e-4):
    """Condition number of the gaze Jacobian at one pose, by central
    differences -- the same quantity workspace() maps over the whole plane."""
    a = np.array([Iq1 + step, Iq1 - step, Iq1, Iq1])
    b = np.array([Iq2, Iq2, Iq2 + step, Iq2 - step])
    axes, ok = tool_axis_grid(lk, a, b, branch)
    if not ok.all():
        return np.inf
    t, _ = tool_axis_grid(lk, np.array([Iq1]), np.array([Iq2]), branch)
    t = t[0]
    d1 = (axes[0] - axes[1]) / (2 * step)
    d2 = (axes[2] - axes[3]) / (2 * step)
    d1 = d1 - np.dot(d1, t) * t
    d2 = d2 - np.dot(d2, t) * t
    a11, a22, a12 = d1 @ d1, d2 @ d2, d1 @ d2
    tr, det = a11 + a22, a11 * a22 - a12 * a12
    disc = max(tr * tr / 4 - det, 0.0)
    hi, lo = tr / 2 + np.sqrt(disc), tr / 2 - np.sqrt(disc)
    return float(np.sqrt(hi / lo)) if lo > 1e-18 else np.inf


def in_workspace(r, Iq1, Iq2):
    """Is this motor pair inside the usable component containing home?"""
    m = r["maps"]
    a1, a2 = m["Iq1"][:, 0], m["Iq2"][0, :]
    i = int(np.argmin(np.abs(a1 - Iq1)))
    j = int(np.argmin(np.abs(a2 - Iq2)))
    return bool(m["comp_dex"][i, j])


def _wrap180(deg):
    """Wrap an angle in degrees to (-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


def _minimal_rotation(a, b):
    """The rotation matrix R with R @ a = b that adds no twist about a x b --
    i.e. the shortest rotation from unit vector a to unit vector b.

    This is the geometric definition of Listing's law: an eye position is
    Listing-compliant exactly when it is reached from primary position by
    this minimal rotation (rotation axis a x b is then automatically
    perpendicular to a, i.e. lies in "Listing's plane" through the primary
    direction -- there is no other, independent condition to check).
    Rodrigues' rotation formula from the cross product of a and b.
    """
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:                     # a and b already parallel
            return np.eye(3)
        # antipodal: any 180 deg rotation about an axis perp to a takes a to
        # b, so this is genuinely undetermined (never reached in practice --
        # would need the current gaze exactly opposite home). Must still be
        # a proper rotation (det +1), not the det -1 reflection an earlier
        # version of this returned here: 2*outer(k,k) - I for unit k perp to
        # a IS a proper 180 deg rotation about k.
        k = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        k = k / np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


# ----------------------------------------------------------------------
# the UI
# ----------------------------------------------------------------------
class Control:
    def __init__(self, lk, args):
        self.lk, self.args = lk, args
        self.branch = args.branch
        self.Iq1, self.Iq2 = EVAL_KW["home_input"]
        # The eye is gripped, so the head faces where the mechanism holds it
        # at home: that direction, not -N.z, is the accessible side. Resolved
        # BEFORE the first measure() call, since the interference mask needs it.
        if args.access_axis is None:
            args.access_axis = home_axis_of(lk, self.branch,
                                            EVAL_KW["home_input"])
            print("access axis = home gaze direction "
                  f"[{', '.join(f'{v:+.3f}' for v in args.access_axis)}]")
        print(f"measuring the workspace at n_input={args.n_input} ...")
        self.r, self.bins = measure(lk, args.n_input, self.branch,
                                    mask=self.build_mask())
        self.M = home_frame(self.r["home_axis"])
        self.update_target()
        self.update_home_pose()

        self.fig = plt.figure(figsize=(16.5, 8.2))
        # top left at 0.85/0.83 rather than 0.93/0.91: the ball title now
        # carries an extra eye-range-coverage line, and needs the headroom
        # to not run into the figure suptitle above it
        self.ax_ball = self.fig.add_axes([0.02, 0.36, 0.30, 0.49])
        self.ax_in = self.fig.add_axes([0.37, 0.40, 0.24, 0.43])
        self.ax_3d = self.fig.add_axes([0.66, 0.36, 0.32, 0.49],
                                       projection="3d")
        ax_s1 = self.fig.add_axes([0.10, 0.275, 0.50, 0.032])
        ax_s2 = self.fig.add_axes([0.10, 0.225, 0.50, 0.032])
        self.s1 = Slider(ax_s1, "Iq1 [deg]", -180.0, 180.0,
                         valinit=np.degrees(self.Iq1), valstep=0.5)
        self.s2 = Slider(ax_s2, "Iq2 [deg]", -180.0, 180.0,
                         valinit=np.degrees(self.Iq2), valstep=0.5)
        self.s1.on_changed(self.on_slider)
        self.s2.on_changed(self.on_slider)
        self.txt = self.fig.text(0.06, 0.015, "", family="monospace",
                                 fontsize=9.0, va="bottom", linespacing=1.5)

        self.static_ball()
        self.static_input()
        self.init_gaze()
        self.init_3d()
        self.update()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def build_mask(self):
        """The access/self-hit/eye mask for workspace(pose_mask=...), or None
        if interference is switched off entirely (--no-interference-map),
        giving the raw kinematic region.

        static_input() draws the same acc/sel/eye split again as the
        hatch/white line, so the call here caches it on self.clean instead of
        making grid_masks() walk the whole input plane twice per measurement.
        """
        self.clean = None
        if not self.args.interference_map:
            return None

        def mask(lk_, Iq1, Iq2, branch_):
            acc, sel, eye, _, _ = grid_masks(
                lk_, Iq1, Iq2, branch_, self.args.access_axis,
                self.args.access_deg, self.args.link_diameter,
                self.args.eye_radius, self.args.with_arm)
            self.clean = acc & sel & eye
            return self.clean

        return mask

    def update_target(self):
        """How much of the physiological eye-rotation range this design (on
        this branch, under this interference mask) actually reaches. This is
        a property of the whole measured workspace, not of the current
        slider pose, so it is computed once per measure() call (init and
        branch switch) rather than on every frame.
        """
        self.target = target_coverage(self.r, self.args.target_h_deg,
                                      self.args.target_v_deg)
        print(f"eye range +/-{self.args.target_h_deg:.0f} x "
              f"+/-{self.args.target_v_deg:.0f} deg: "
              f"{100*self.target['coverage']:.0f}% reached, "
              f"{100*self.target['scale']:.0f}% guaranteed hole-free")

    def update_home_pose(self):
        """Cache the home pose's plate orientation and roll, anchored to
        THIS mechanism's own home direction rather than solve()'s default
        reference axis (0, 0, -1) -- which is unrelated to it (this design's
        home gaze is typically nowhere near that direction) and, evaluated
        away from home, makes roll drift by a coordinate artifact that grows
        with distance from home: verified against an independent
        from-scratch implementation, a ~32-38 deg gaze eccentricity should
        produce at most a few degrees of Listing's-law torsion, not the
        tens of degrees solve()'s default reference was giving.

        roll_home       roll AT home in this anchored convention -- a fixed
                        offset (e.g. a mount you could physically pre-twist),
                        used as the zero-point for status()'s "roll here,
                        relative to home".
        R_plate_home,
        tool_axis_home  used by status() to predict, at any OTHER pose, what
                        roll Listing's law would produce there, correctly
                        anchored the same way.
        """
        home_s = solve(self.lk, *EVAL_KW["home_input"], branch=self.branch)
        self.tool_axis_home = home_s["tool_axis"]
        self.R_plate_home = home_s["R_plate"]
        *_, self.roll_home = eye_angles(self.tool_axis_home, self.R_plate_home,
                                        home=self.tool_axis_home)

    # -- static layers -------------------------------------------------
    def static_ball(self):
        self.ax_ball.clear()
        self.ax_ball.imshow(ball_image(self.bins, projection=self.args.projection,
                                       dist=self.args.view_dist),
                            extent=(-1, 1, -1, 1), origin="upper",
                            interpolation="nearest", zorder=1)
        draw_graticule(self.ax_ball, self.args.projection, self.args.view_dist,
                       pe=pe)
        self.ax_ball.plot(0, 0, "*", color="orange", ms=14, mec="k", zorder=5)
        sub = ("blue = reachable, access-clear and self-collision-free"
              if self.args.interference_map else
              "blue = reachable on this branch (kinematics only, no "
              "interference check)")
        sub += (f"\neye range +/-{self.args.target_h_deg:.0f}"
               f"x+/-{self.args.target_v_deg:.0f} deg: "
               f"{100*self.target['coverage']:.0f}% reached, "
               f"{100*self.target['scale']:.0f}% guaranteed hole-free")
        self.ax_ball.set_title(
            "gaze on the eye  (star = home, dot = now)\n" + sub,
            fontsize=9, linespacing=1.4)

    def static_input(self):
        m = self.r["maps"]
        self.ax_in.clear()
        k = np.log10(np.clip(m["kappa"], 1, 1e3))
        self.ax_in.pcolormesh(np.degrees(m["Iq1"]), np.degrees(m["Iq2"]), k,
                              shading="auto", cmap="viridis_r", zorder=1)
        self.ax_in.contour(np.degrees(m["Iq1"]), np.degrees(m["Iq2"]),
                           m["comp_dex"].astype(float), levels=[0.5],
                           colors="red", linewidths=1.6, zorder=2)

        # what is left once the eye may only be approached from one side --
        # self.clean was already computed once by build_mask()'s pose_mask
        # callable during measure(), on this exact (Iq1, Iq2) grid
        if self.args.interference_map:
            clean = self.clean
            self.ax_in.contourf(np.degrees(m["Iq1"]), np.degrees(m["Iq2"]),
                                (~clean).astype(float), levels=[0.5, 1.5],
                                colors="none", hatches=["///"], zorder=2)
            self.ax_in.contour(np.degrees(m["Iq1"]), np.degrees(m["Iq2"]),
                               clean.astype(float), levels=[0.5],
                               colors="white", linewidths=1.1, zorder=3)
            comp = m["comp_dex"]
            frac = (clean & comp).sum() / max(comp.sum(), 1)
            self.frac_clean = float(frac)
            print(f"interference-free share of the usable workspace: "
                  f"{100*frac:.1f}%")
        self.ax_in.plot(np.degrees(EVAL_KW["home_input"][0]),
                        np.degrees(EVAL_KW["home_input"][1]), "*",
                        color="orange", ms=13, mec="k", zorder=4)
        self.ax_in.set_xlabel("Iq1 [deg]")
        self.ax_in.set_ylabel("Iq2 [deg]")
        sub = ("red = usable; hatched = behind the eye or self-hitting; "
               "white = what is left" if self.args.interference_map else
               "red outline = usable region; click to jump")
        self.ax_in.set_title("motor plane, colour = $\\log_{10}\\kappa$\n" + sub,
                             fontsize=8.5)
        self.vline = self.ax_in.axvline(0.0, color="w", lw=1.0, zorder=3)
        self.hline = self.ax_in.axhline(0.0, color="w", lw=1.0, zorder=3)
        self.here, = self.ax_in.plot([], [], "o", color="red", ms=7, mec="k",
                                     zorder=5)

    def init_gaze(self):
        self.dot, = self.ax_ball.plot([], [], "o", color="crimson", ms=11,
                                      mec="k", zorder=6)
        self.behind = self.ax_ball.text(0, -1.28, "", ha="center", fontsize=9,
                                        color="crimson")

    def init_3d(self):
        a = self.ax_3d
        self.l_links = [a.plot([], [], [], color=c, lw=lw)[0] for c, lw in
                        [("#777777", 2.4), ("#777777", 2.4), ("#333333", 2.8),
                         ("#777777", 2.4), ("#c05000", 3.0)]]
        self.l_joints, = a.plot([], [], [], "o", color="k", ms=5)
        # the attachment arm: plate -> globe, the part that grips the eye
        self.l_arm, = a.plot([], [], [], color="#1b7837", lw=3.6)
        self.l_tool, = a.plot([], [], [], color="crimson", lw=2.2)
        self.l_home, = a.plot([], [], [], color="orange", lw=1.4, ls="--")
        # the eye itself, a solid ball to scale at the RCM, so that a bar
        # passing through it is visible and not just a line in the report
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        rad = self.args.eye_radius * self.lk.h
        ex = rad * np.outer(np.cos(u), np.sin(v)) + self.lk.RCM[0]
        ey = rad * np.outer(np.sin(u), np.sin(v)) + self.lk.RCM[1]
        ez = rad * np.outer(np.ones_like(u), np.cos(v)) + self.lk.RCM[2]
        a.plot_surface(ex, ey, ez, color="#e8eef2", shade=True, alpha=.95,
                       linewidth=0, antialiased=True, rstride=1, cstride=1)
        a.plot_wireframe(ex, ey, ez, color="0.55", lw=.35, rstride=4,
                         cstride=4, alpha=.5)
        a.scatter(*self.lk.RCM, color="tab:orange", s=26, depthshade=False)
        self.draw_access_boundary(a)

        # frame the linkage itself, not an arbitrary cube: the mechanism sits
        # between P1 at the origin and the RCM at h*z, and swings about that
        lk = self.lk
        half = 0.85 * max(lk.h, lk.l1 + lk.l2, lk.l4 + lk.l5)
        cz = 0.5 * lk.h
        a.set_xlim(-half, half)
        a.set_ylim(-half, half)
        a.set_zlim(cz - half, cz + half)
        a.set_box_aspect((1, 1, 1))
        a.view_init(elev=22, azim=-60)
        a.set_title("the mechanism now (drag to rotate)\n"
                    "orange dashed = home gaze, red = instrument axis",
                    fontsize=10)
        a.set_xlabel("x"); a.set_ylabel("y"); a.set_zlabel("z")

    def draw_access_boundary(self, a):
        """The rim of the allowed cone, drawn at the RCM.

        Everything on the far side of this circle is behind the eye: orbit,
        socket, face. With the default 90 deg it is simply the plane through
        the RCM perpendicular to the access axis.
        """
        ax = np.asarray(self.args.access_axis, float)
        ax = ax / np.linalg.norm(ax)
        e1 = np.cross(ax, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(ax, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        half = np.radians(self.args.access_deg)
        rad = 0.9 * max(self.lk.h, self.lk.l1 + self.lk.l2)
        t = np.linspace(0, 2 * np.pi, 160)
        ring = (self.lk.RCM
                + rad * np.cos(half) * ax
                + rad * np.sin(half) * (np.cos(t)[:, None] * e1
                                        + np.sin(t)[:, None] * e2))
        a.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="tab:red", lw=1.2,
               ls=":", alpha=.8)

    # -- live update ---------------------------------------------------
    def update(self):
        lk = self.lk
        try:
            s = solve(lk, self.Iq1, self.Iq2, branch=self.branch)
            err = None
        except RCMError as exc:
            s, err = None, f"{type(exc).__name__}: {exc}".split(";")[0]

        if s is None:
            self.dot.set_data([], [])
            self.behind.set_text("")
            for ln in self.l_links:
                ln.set_data([], []); ln.set_3d_properties([])
            self.l_joints.set_data([], []); self.l_joints.set_3d_properties([])
            self.l_tool.set_data([], []); self.l_tool.set_3d_properties([])
            self.status(None, err)
        else:
            t = self.M.T @ s["tool_axis"]
            x, y, vis = project_direction(t, self.args.projection,
                                          self.args.view_dist)
            if vis:
                self.dot.set_data([x], [y])
                self.behind.set_text("")
            else:
                self.dot.set_data([], [])
                self.behind.set_text("gaze points behind the visible "
                                     "hemisphere")
            hit = check_pose(s, access_axis=self.args.access_axis,
                             access_deg=self.args.access_deg,
                             link_diameter=self.args.link_diameter,
                             eye_radius=self.args.eye_radius,
                             scale=self.lk.h, with_arm=self.args.with_arm)
            self.draw_mech(s, hit)
            self.status(s, None, hit)

        self.here.set_data([np.degrees(self.Iq1)], [np.degrees(self.Iq2)])
        self.vline.set_xdata([np.degrees(self.Iq1)] * 2)
        self.hline.set_ydata([np.degrees(self.Iq2)] * 2)
        self.fig.canvas.draw_idle()

    def draw_mech(self, s, hit=None):
        P = {k: np.asarray(s[k], float) for k in ("P1", "P2", "P3", "P4", "P5")}
        RCM = np.asarray(s["RCM"], float)
        # LINKS fixes the order, so the colouring lines up with the report
        bad_links = set()
        if hit is not None:
            bad_links |= {nm for nm, _, viol in hit["per_link"] if viol}
            if not hit["self_ok"]:
                bad_links |= set(hit["self_pair"])
        for ln, (nm, a, b) in zip(self.l_links, LINKS):
            ln.set_data([P[a][0], P[b][0]], [P[a][1], P[b][1]])
            ln.set_3d_properties([P[a][2], P[b][2]])
            base = "#333333" if nm.startswith("l4") else "#777777"
            if nm.startswith("l6"):
                base = "#c05000"
            ln.set_color("red" if nm in bad_links else base)
            ln.set_linewidth(3.4 if nm in bad_links else
                             (2.8 if nm.startswith("l4") else 2.4))
        pts = np.array(list(P.values()))
        self.l_joints.set_data(pts[:, 0], pts[:, 1])
        self.l_joints.set_3d_properties(pts[:, 2])

        if self.args.with_arm:
            inner, outer = arm_endpoints(RCM, P["P3"], P["P5"], s["tool_axis"],
                                         self.args.eye_radius * self.lk.h)
            self.l_arm.set_data([inner[0], outer[0]], [inner[1], outer[1]])
            self.l_arm.set_3d_properties([inner[2], outer[2]])
            self.l_arm.set_color("red" if ARM in bad_links else "#1b7837")
        else:
            # direct suction on P3/P5, no separate arm to draw
            self.l_arm.set_data([], [])
            self.l_arm.set_3d_properties([])

        reach = 0.55 * self.lk.h            # long enough to read the direction
        tip = RCM + reach * np.asarray(s["tool_axis"], float)
        self.l_tool.set_data([RCM[0], tip[0]], [RCM[1], tip[1]])
        self.l_tool.set_3d_properties([RCM[2], tip[2]])
        hm = RCM + reach * np.asarray(self.r["home_axis"], float)
        self.l_home.set_data([RCM[0], hm[0]], [RCM[1], hm[1]])
        self.l_home.set_3d_properties([RCM[2], hm[2]])

    def status(self, s, err, hit=None):
        i1, i2 = np.degrees(self.Iq1), np.degrees(self.Iq2)
        head = (f"Iq1 {i1:+7.1f} deg   Iq2 {i2:+7.1f} deg   branch "
                f"{self.branch:+d}")
        if s is None:
            self.txt.set_text(f"{head}\n{err}\n"
                              "this motor pair does not assemble on this "
                              "branch")
            self.txt.set_color("crimson")
            return
        th_h, th_v = np.degrees(s["theta_h"]), np.degrees(s["theta_v"])
        # angles measured from the home gaze, which is what the picture shows
        t = self.M.T @ s["tool_axis"]
        fh = np.degrees(np.arctan2(t[0], t[2]))
        fv = np.degrees(np.arcsin(np.clip(t[1], -1, 1)))
        off = np.degrees(np.arccos(np.clip(t[2], -1, 1)))
        k = kappa_at(self.lk, self.Iq1, self.Iq2, self.branch)
        inside = in_workspace(self.r, self.Iq1, self.Iq2)
        kmax = EVAL_KW["kappa_max"]
        k_flag = "ok" if k <= kmax else f"above kappa_max={kmax:.0f}"
        # Both rolls below are measured with home=self.tool_axis_home (NOT
        # solve()'s default reference (0,0,-1), which is unrelated to this
        # design's actual home direction and, used away from home, drifts by
        # a coordinate artifact worth tens of degrees -- verified against an
        # independent from-scratch reimplementation of Listing's law).
        #
        # roll_listing: what an idealised Listing's-law eye would show at
        # this SAME gaze direction -- built by applying the minimal
        # (no-extra-spin) rotation from home to the home plate orientation.
        # Checked against a synthetic exactly-Listing-compliant pose, where
        # roll_now - roll_listing below reads exactly 0.
        R_listing = _minimal_rotation(self.tool_axis_home,
                                      s["tool_axis"]) @ self.R_plate_home
        *_, roll_actual = eye_angles(s["tool_axis"], s["R_plate"],
                                     home=self.tool_axis_home)
        *_, roll_listing = eye_angles(s["tool_axis"], R_listing,
                                      home=self.tool_axis_home)
        roll_home_deg = np.degrees(self.roll_home)
        roll_now = _wrap180(np.degrees(roll_actual) - roll_home_deg)
        roll_normal = _wrap180(np.degrees(roll_listing) - roll_home_deg)
        roll_deviation = _wrap180(np.degrees(roll_actual)
                                  - np.degrees(roll_listing))
        # --- interference, the part the kinematics cannot see -----------
        acc = (f"access:  worst bar {hit['access_worst_link'].split()[0]} at "
               f"{hit['access_worst_deg']:5.1f} deg from the access axis "
               f"(limit {self.args.access_deg:.0f})   "
               + ("OK, margin %.1f deg" % hit["access_margin_deg"]
                  if hit["access_ok"] else
                  "BEHIND THE EYE by %.1f deg" % -hit["access_margin_deg"]))
        gap_h = hit["self_gap"] / self.lk.h
        lim_h = hit["self_limit"] / self.lk.h
        sel = (f"self-hit: closest free pair {hit['self_pair'][0].split()[0]}"
               f"-{hit['self_pair'][1].split()[0]} at {gap_h:.3f} h "
               f"(bar diameter {lim_h:.3f} h)   "
               + ("clear" if hit["self_ok"] else "CRASH"))
        if not hit["eye_ok"]:
            sel += f"   |   {hit['eye_link'].split()[0]} IS INSIDE THE EYEBALL"

        self.txt.set_text(
            f"{head}\n"
            f"gaze from home:  theta_h {fh:+7.1f}   theta_v {fv:+7.1f}   "
            f"{off:5.1f} deg off home\n"
            f"roll rel. home:  system {roll_now:+7.1f}    normal eye here "
            f"{roll_normal:+7.1f}  (Listing's law)    difference "
            f"{roll_deviation:+7.1f}\n"
            f"kappa {k:8.2f} ({k_flag})"
            f"     inside the usable workspace: {'YES' if inside else 'no'}"
            f"     world Fick theta_h {th_h:+.1f}, theta_v {th_v:+.1f}\n"
            f"{acc}\n{sel}")
        self.txt.set_color("crimson" if not hit["ok"] else
                           ("k" if inside else "darkorange"))

    # -- events --------------------------------------------------------
    def on_slider(self, _):
        self.Iq1 = np.radians(self.s1.val)
        self.Iq2 = np.radians(self.s2.val)
        self.update()

    def on_click(self, ev):
        if ev.inaxes is self.ax_in and ev.xdata is not None:
            self.s1.set_val(round(float(ev.xdata) * 2) / 2)
            self.s2.set_val(round(float(ev.ydata) * 2) / 2)

    def on_key(self, ev):
        step = 5.0 if (ev.key or "").startswith("shift+") else 1.0
        key = (ev.key or "").replace("shift+", "")
        if key == "left":
            self.s1.set_val(max(-180.0, self.s1.val - step))
        elif key == "right":
            self.s1.set_val(min(180.0, self.s1.val + step))
        elif key == "down":
            self.s2.set_val(max(-180.0, self.s2.val - step))
        elif key == "up":
            self.s2.set_val(min(180.0, self.s2.val + step))
        elif key == "h":
            self.s1.set_val(np.degrees(EVAL_KW["home_input"][0]))
            self.s2.set_val(np.degrees(EVAL_KW["home_input"][1]))
        elif key == "b":
            self.branch = -self.branch
            print(f"branch {self.branch:+d}, recomputing the workspace ...")
            self.r, self.bins = measure(self.lk, self.args.n_input,
                                        self.branch, mask=self.build_mask())
            self.M = home_frame(self.r["home_axis"])
            self.update_target()
            self.update_home_pose()
            self.static_ball()
            self.static_input()
            self.init_gaze()
            self.update()
        elif key == "s":
            path = f"designs/control_Iq1{self.s1.val:+.0f}_Iq2{self.s2.val:+.0f}.png"
            self.fig.savefig(path, dpi=130)
            print("saved", path)
        elif key == "q":
            plt.close(self.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=int, default=0,
                    help="index into the ranked list of compact designs")
    ap.add_argument("--sort", default="home_cone",
                    choices=["home_cone", "round", "box", "omega"])
    ap.add_argument("--params", help="l1,l2,l3,l4,l5,l6,q1,q4 as ratios at h=1")
    ap.add_argument("--h", type=float, default=None,
                    help="your real P1-to-RCM distance; lengths are scaled by "
                         "it, angles are unaffected. Defaults to 1.0, except "
                         "for --results eye, where it defaults to the loaded "
                         "design's own h_mm/1000 (l3 there was searched "
                         "against a real eye radius at that specific scale)")
    ap.add_argument("--branch", type=int, default=EVAL_KW["branch"],
                    choices=[-1, 1])
    ap.add_argument("--n-input", type=int, default=401)
    ap.add_argument("--projection", default="ortho",
                    choices=["ortho", "perspective"])
    ap.add_argument("--view-dist", type=float, default=6.0)
    ap.add_argument("--eye-radius", type=float, default=None,
                    help="eye radius as a fraction of h (12 mm eye at "
                         "h = 150 mm is 0.08); also used for the eye check. "
                         "Defaults to 0.08, except for --results eye, where "
                         "it is derived from the loaded design's own "
                         "eye_radius_mm")
    ap.add_argument("--access-axis", default=None,
                    help="direction from the RCM towards the accessible side "
                         "in world coordinates, e.g. 0,0,-1. Default: the "
                         "home gaze direction of this design, which is where "
                         "the head faces when the eye is gripped")
    ap.add_argument("--results", default="v2", choices=["v1", "v2", "eye"],
                    help="which run to take designs from: v1 had no "
                         "interference model, v2 grips the eye with a rigid "
                         "arm, eye grips it directly with suction cups (l3 "
                         "free above the eye radius)")
    ap.add_argument("--access-deg", type=float, default=90.0,
                    help="half-angle of the allowed cone about that axis; "
                         "90 = the whole front half-space")
    ap.add_argument("--link-diameter", type=float, default=0.03,
                    help="bar diameter as a fraction of h, for the "
                         "self-collision test")
    ap.add_argument("--no-interference-map", dest="interference_map",
                    action="store_false",
                    help="skip the hatched overlay on the motor plane")
    ap.add_argument("--target-h", type=float, default=TARGET_H_DEG,
                    dest="target_h_deg",
                    help="eye rotation half-range, horizontal [deg], for "
                         "the coverage readout on the left panel")
    ap.add_argument("--target-v", type=float, default=TARGET_V_DEG,
                    dest="target_v_deg",
                    help="eye rotation half-range, vertical [deg], for "
                         "the coverage readout on the left panel")
    args = ap.parse_args()
    if args.access_axis is not None:
        args.access_axis = np.array([float(v) for v in
                                     args.access_axis.split(",")], float)

    # with_arm is a property of which family a design came from, not an
    # independent choice: v1/v2 designs assume a rigid attachment arm, eye
    # designs assume none (suction cups directly on P3/P5) -- see
    # rcm_view.constraints_of().
    args.with_arm = args.results != "eye"

    row = None
    if args.params:
        vals = [float(v) for v in args.params.split(",")]
        if len(vals) != 8:
            raise SystemExit("--params needs 8 numbers: l1..l6, q1, q4")
        p = dict(zip(NAMES, vals))
    else:
        src = {"v1": SOURCES_V1, "v2": SOURCES_V2,
              "eye": SOURCES_EYE}[args.results]
        rows = best_designs(args.sort, args.design + 1, paths=src)
        if len(rows) <= args.design:
            raise SystemExit(f"only {len(rows)} distinct designs available")
        row = rows[args.design]
        p = params_of(row)

    resolve_scale(args, args.results, row)

    scaled = {n: (v * args.h if n.startswith("l") else v) for n, v in p.items()}
    lk = Linkage(h=args.h, **scaled)
    print(lk)
    print("  ratios at h=1: " + ", ".join(f"{n}={p[n]:.4f}" for n in NAMES))
    print(f"  collision model: {'arm-gripped (v1/v2)' if args.with_arm else 'direct suction, no arm (eye)'}"
          + (f", eye radius {args.eye_radius:.4f} (ratio)" if not args.with_arm else ""))

    ui = Control(lk, args)
    ui.fig.suptitle(
        "5-bar spherical RCM -- manual control    "
        + "  ".join(f"{n}={scaled[n]:.4f}" for n in NAMES)
        + f"   q5={lk.q5:.4f} (derived)   h={lk.h:g}", fontsize=9)
    print("sliders drive the motors; click the motor plane to jump; "
          "arrows nudge; h = home, b = branch, s = save, q = quit")
    plt.show()


if __name__ == "__main__":
    main()
