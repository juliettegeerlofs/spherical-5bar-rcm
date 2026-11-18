"""
rcm5bar.py -- closed-form kinematics of the 5-bar spherical RCM mechanism.

Geometry / conventions
----------------------
Ground point P1 at the origin of the world frame N.
The remote centre of motion RCM sits at h * N.z.

Every joint axis in the chain passes through RCM, so the mechanism is a
SPHERICAL 5R linkage: each point keeps a constant distance to RCM.

    leg 1 (actuated by Iq1):  P1 --l1--> P2 --l2--> P3
    ground link (fixed)    :  P1 --l4--> P4
    leg 2 (actuated by Iq2):  P4 --l5--> P5
    end plate (isoceles)   :  P3 --l6--> P5, both at distance l3 from RCM

Angles
------
    Iq1, Iq2        actuated (motors)
    q1, q4, q5      fixed structural angles, built into the parts
    q2, q3, q6      passive pin rotations, solved from loop closure

DESIGN CONSTRAINT: |P5 - RCM| must equal l3, which makes q5 a *dependent*
quantity, not a free choice.  Use required_q5() to obtain it.

Solution
--------
With Iq1, Iq2 given, P2 and P5 follow by direct forward kinematics.  The only
unknown left is the direction b_hat = unit(P3 - RCM), fixed by two cone
constraints (angle to P2 set by l2, angle to P5 set by l6).  Writing

    b_hat = mu * a_hat + nu * c_hat + lam * (a_hat x c_hat)

turns the problem into a 2x2 linear system plus one square root.  The sign of
lam is the only genuine assembly-mode ambiguity: two branches.

All angles are in radians unless a function name says otherwise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Linkage",
    "required_q5",
    "check_design",
    "solve",
    "eye_angles",
    "SingularConfiguration",
    "UnreachableConfiguration",
    "InconsistentDesign",
]


# ----------------------------------------------------------------------
# exceptions
# ----------------------------------------------------------------------
class RCMError(ValueError):
    pass


class InconsistentDesign(RCMError):
    """Link lengths / fixed angles cannot ever close the loop."""


class UnreachableConfiguration(RCMError):
    """Valid design, but this (Iq1, Iq2) pair lies outside the workspace."""


class SingularConfiguration(RCMError):
    """P2 and P5 are collinear as seen from RCM -> b_hat undetermined."""


# ----------------------------------------------------------------------
# small vector helpers
# ----------------------------------------------------------------------
def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise RCMError("zero-length vector, cannot normalise")
    return np.asarray(v, float) / n


def _frame(y_axis, z_axis):
    """Right-handed orthonormal frame from a y and a z direction (x = y x z).

    Columns of the returned 3x3 are the x, y, z axes expressed in N, i.e. the
    matrix maps frame-local coordinates to world coordinates.
    """
    z = _unit(z_axis)
    y = _unit(y_axis)
    y = _unit(y - np.dot(y, z) * z)          # re-orthogonalise against drift
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def _spin_z(R, angle):
    """Rotate frame R about its own z axis by `angle`."""
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ Rz


def _angle_in_frame(v, R):
    """Azimuth of v about the z axis of frame R, measured from R's x axis."""
    local = R.T @ v
    return float(np.arctan2(local[1], local[0]))


# ----------------------------------------------------------------------
# linkage description
# ----------------------------------------------------------------------
class Linkage:
    """Container for the geometric parameters.

    Parameters
    ----------
    h                  distance P1 -> RCM along N.z
    l1, l2             leg 1 link lengths (P1->P2, P2->P3)
    l3                 plate radius: |P3 - RCM| = |P5 - RCM|
    l4                 ground link P1 -> P4
    l5                 leg 2 link length P4 -> P5
    l6                 plate chord |P3 - P5|
    q1                 fixed angle of l1 away from the P1 rotation axis (N.z)
    q4                 fixed angle of the ground link away from N.z
    q5                 fixed angle of l5 away from the P4->RCM axis.
                       Leave as None to have it computed from the closure
                       condition |P5 - RCM| = l3 (recommended).
    """

    def __init__(self, h, l1, l2, l3, l4, l5, l6, q1, q4, q5=None):
        self.h = float(h)
        self.l1, self.l2, self.l3 = float(l1), float(l2), float(l3)
        self.l4, self.l5, self.l6 = float(l4), float(l5), float(l6)
        self.q1, self.q4 = float(q1), float(q4)

        # distances to RCM that the spherical structure holds constant
        self.d2 = float(np.sqrt(self.l1**2 + self.h**2
                                - 2 * self.l1 * self.h * np.cos(self.q1)))
        self.d4 = float(np.sqrt(self.l4**2 + self.h**2
                                - 2 * self.l4 * self.h * np.cos(self.q4)))

        self.q5 = required_q5(self.l3, self.l5, self.d4) if q5 is None else float(q5)

        check_design(self)

        # constant cone angles used by the solver
        self.cos_a = (self.d2**2 + self.l3**2 - self.l2**2) / (2 * self.d2 * self.l3)
        self.cos_c = 1.0 - self.l6**2 / (2 * self.l3**2)

        # phi2 / delta: reported for completeness, not needed by the solver
        self.phi2 = float(np.arccos(
            (self.l2**2 + self.d2**2 - self.l3**2) / (2 * self.l2 * self.d2)))
        self.delta = float(np.arccos(self.l6 / (2 * self.l3)))

        # --- pieces that do not depend on the actuators -------------------
        # ground link direction and P4
        self.D_z = np.array([np.sin(self.q4), 0.0, np.cos(self.q4)])
        self.P4 = self.l4 * self.D_z
        self.RCM = np.array([0.0, 0.0, self.h])
        # frame at P4: z toward RCM, y inherited from the ground link (= N.y)
        self.R_I4 = _frame(np.array([0.0, 1.0, 0.0]), self.RCM - self.P4)

    def __repr__(self):
        return (f"Linkage(h={self.h:.4g}, l1={self.l1:.4g}, l2={self.l2:.4g}, "
                f"l3={self.l3:.4g}, l4={self.l4:.4g}, l5={self.l5:.4g}, "
                f"l6={self.l6:.4g}, q1={self.q1:.4g}, q4={self.q4:.4g}, "
                f"q5={self.q5:.4g})")


def required_q5(l3, l5, d4):
    """Fixed angle at P4 that puts P5 on the sphere of radius l3 around RCM."""
    if d4 <= 0 or l5 <= 0:
        raise InconsistentDesign("l5 and d4 must be positive")
    c = (l5**2 + d4**2 - l3**2) / (2 * l5 * d4)
    if not -1.0 <= c <= 1.0:
        raise InconsistentDesign(
            f"no q5 exists for l3={l3:.6g}, l5={l5:.6g}, d4={d4:.6g}; "
            "the triangle P4-P5-RCM does not close. Adjust l5, l4, q4 or h.")
    return float(np.arccos(c))


def check_design(lk: "Linkage", tol=1e-9):
    """Raise if the parameter set can never close, whatever the inputs are."""
    r5 = np.sqrt(lk.l5**2 + lk.d4**2 - 2 * lk.l5 * lk.d4 * np.cos(lk.q5))
    if abs(r5 - lk.l3) > tol * max(1.0, lk.l3):
        raise InconsistentDesign(
            f"|P5-RCM| = {r5:.9g} but l3 = {lk.l3:.9g}. q5 is not a free "
            "parameter; use required_q5() or pass q5=None.")
    if lk.l6 > 2 * lk.l3:
        raise InconsistentDesign(f"plate chord l6={lk.l6} exceeds 2*l3={2*lk.l3}")
    if not abs(lk.l2 - lk.l3) <= lk.d2 <= lk.l2 + lk.l3:
        raise InconsistentDesign(
            f"triangle P2-P3-RCM impossible: d2={lk.d2:.6g}, l2={lk.l2:.6g}, "
            f"l3={lk.l3:.6g}")
    return True


# ----------------------------------------------------------------------
# the solver
# ----------------------------------------------------------------------
def solve(lk: "Linkage", Iq1, Iq2, branch=+1, tool_dir_plate=(0.0, 0.0, 1.0),
          home=(0.0, 0.0, -1.0), right=(1.0, 0.0, 0.0), sing_tol=1e-6):
    """Full pose of the mechanism for one pair of actuator angles.

    Parameters
    ----------
    lk              Linkage
    Iq1, Iq2        actuator angles [rad]
    branch          +1 or -1, the assembly mode (sign of lam)
    tool_dir_plate  instrument axis expressed in the plate frame.
                    (0,0,1) = plate symmetry axis, i.e. RCM -> midpoint(P3,P5)
    home, right     define the measurement frame for the eye angles:
                    home is the straight-ahead direction, right the horizontal
                    reference. Both in world coordinates.
    sing_tol        how close |cos(P2,RCM,P5)| may get to 1 before refusing

    Returns
    -------
    dict with keys:
        theta_h, theta_v        horizontal / vertical eye angle [rad] (Fick)
        theta_h_helm, theta_v_helm   same in Helmholtz order [rad]
        tool_axis               unit instrument direction in world coords
        roll                    rotation of the plate about the tool axis [rad]
        P1..P5, RCM             point positions
        q2, q3, q6              passive joint angles [rad]
        R_plate                 plate frame (columns x,y,z) in world coords
        lam, g, branch          solver internals / branch bookkeeping
    """
    if branch not in (+1, -1):
        raise ValueError("branch must be +1 or -1")

    RCM = lk.RCM
    P1 = np.zeros(3)

    # --- leg 1: base spin Iq1 about N.z, then fixed tilt q1 ---------------
    R_I1 = _spin_z(np.eye(3), Iq1)                 # frame at P1
    A_y = R_I1[:, 1]
    A_z = (np.cos(lk.q1) * R_I1[:, 2] + np.sin(lk.q1) * R_I1[:, 0])
    P2 = lk.l1 * A_z
    R_I2 = _frame(A_y, RCM - P2)                   # z toward RCM, y from leg 1

    # --- leg 2: spin Iq2 about the P4->RCM axis, then fixed tilt q5 -------
    R_I4s = _spin_z(lk.R_I4, Iq2)
    E_z = np.cos(lk.q5) * R_I4s[:, 2] + np.sin(lk.q5) * R_I4s[:, 0]
    P5 = lk.P4 + lk.l5 * E_z
    R_I5 = _frame(R_I4s[:, 1], RCM - P5)

    # --- the one real unknown: direction from RCM to P3 -------------------
    a_hat = _unit(P2 - RCM)
    c_hat = _unit(P5 - RCM)
    g = float(np.dot(a_hat, c_hat))
    if abs(g) > 1.0 - sing_tol:
        raise SingularConfiguration(
            f"P2 and P5 nearly collinear from RCM (cos = {g:.9f}); "
            "P3 is undetermined here.")

    denom = 1.0 - g * g
    mu = (lk.cos_a - g * lk.cos_c) / denom
    nu = (lk.cos_c - g * lk.cos_a) / denom
    lam2 = (1.0 - mu * mu - nu * nu - 2.0 * mu * nu * g) / denom
    if lam2 < 0.0:
        if lam2 > -1e-12:
            lam2 = 0.0                                     # on the boundary
        else:
            raise UnreachableConfiguration(
                f"no assembly for Iq1={Iq1:.6g}, Iq2={Iq2:.6g} "
                f"(lambda^2 = {lam2:.3e} < 0)")
    lam = branch * np.sqrt(lam2)

    b_hat = mu * a_hat + nu * c_hat + lam * np.cross(a_hat, c_hat)
    P3 = RCM + lk.l3 * b_hat

    # --- passive joint angles --------------------------------------------
    q2 = _angle_in_frame(P3 - P2, R_I2)
    R_I2s = _spin_z(R_I2, q2)
    R_I3 = _frame(R_I2s[:, 1], RCM - P3)
    q3 = _angle_in_frame(P5 - P3, R_I3)
    q6 = _angle_in_frame(P3 - P5, R_I5)

    # --- end-effector plate frame ----------------------------------------
    mid = 0.5 * (P3 + P5)
    plate_z = _unit(mid - RCM)                 # symmetry axis of the plate
    plate_x = _unit(P5 - P3)                   # chord, perpendicular to z
    R_plate = _frame(np.cross(plate_z, plate_x), plate_z)

    tool_axis = _unit(R_plate @ np.asarray(tool_dir_plate, float))

    # --- eye / instrument orientation ------------------------------------
    th_h, th_v, th_h_h, th_v_h, roll = eye_angles(tool_axis, R_plate, home, right)

    return {
        "theta_h": th_h, "theta_v": th_v,
        "theta_h_helm": th_h_h, "theta_v_helm": th_v_h,
        "theta_h_deg": np.degrees(th_h), "theta_v_deg": np.degrees(th_v),
        "tool_axis": tool_axis, "roll": roll,
        "P1": P1, "P2": P2, "P3": P3, "P4": lk.P4.copy(), "P5": P5,
        "RCM": RCM.copy(),
        "q2": q2, "q3": q3, "q6": q6,
        "R_plate": R_plate, "lam": float(lam), "g": g, "branch": branch,
        "Iq1": float(Iq1), "Iq2": float(Iq2),
    }


def eye_angles(tool_axis, R_plate, home=(0.0, 0.0, -1.0), right=(1.0, 0.0, 0.0)):
    """Decompose a tool direction into horizontal / vertical angles.

    A measurement frame M is built with M.z = home, M.x = right (orthogonalised),
    M.y = M.z x M.x.  With t = tool_axis expressed in M:

        Fick      (horizontal first, then vertical)
            theta_h = atan2(t_x, t_z)      theta_v = asin(t_y)
        Helmholtz (vertical first, then horizontal)
            theta_h = asin(t_x)            theta_v = atan2(t_y, t_z)

    Both are zero when the tool points along `home`.
    Returns (fick_h, fick_v, helm_h, helm_v, roll).
    """
    z_m = _unit(home)
    x_m = np.asarray(right, float)
    x_m = _unit(x_m - np.dot(x_m, z_m) * z_m)
    y_m = np.cross(z_m, x_m)
    M = np.column_stack((x_m, y_m, z_m))

    t = M.T @ np.asarray(tool_axis, float)
    t = t / np.linalg.norm(t)

    fick_h = float(np.arctan2(t[0], t[2]))
    fick_v = float(np.arcsin(np.clip(t[1], -1.0, 1.0)))
    helm_h = float(np.arcsin(np.clip(t[0], -1.0, 1.0)))
    helm_v = float(np.arctan2(t[1], t[2]))

    # roll: rotation of the plate x axis about the tool axis, measured from the
    # projection of M.x
    ref = x_m - np.dot(x_m, tool_axis) * np.asarray(tool_axis, float)
    if np.linalg.norm(ref) < 1e-12:
        roll = float("nan")
    else:
        ref = _unit(ref)
        px = R_plate[:, 0]
        px = px - np.dot(px, tool_axis) * np.asarray(tool_axis, float)
        if np.linalg.norm(px) < 1e-12:
            roll = float("nan")
        else:
            px = _unit(px)
            roll = float(np.arctan2(
                np.dot(np.cross(ref, px), tool_axis), np.dot(ref, px)))

    return fick_h, fick_v, helm_h, helm_v, roll
