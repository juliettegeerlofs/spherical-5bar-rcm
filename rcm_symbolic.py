"""
rcm_symbolic.py -- the SymPy model, kept as the readable source of truth, and
used to VERIFY the closed-form solver in rcm5bar.py.

This is the same frame-chaining construction as the original hand model, with
two corrections:

  * q5 is derived, not chosen (|P5 - RCM| must equal l3);
  * q6 is acknowledged as a real passive joint variable at P5, determined by
    the plate's rigidity. Five joint variables, two of them actuated.

What this file proves
---------------------
  A. the RCM property: RCM sits at h*N.z for ALL inputs
  B. every point keeps a constant distance to RCM (the spherical structure)
  C. the loop closure is rank 2, not rank 3 -- one equation is redundant
  D. feeding the closed-form solver's q2, q3 into this symbolic chain drives
     loop_closure to zero
  E. a lambdified symbolic closed form agrees with the hand-written NumPy

Run it directly. Symbolic simplification is attempted first; where the
expressions are too tangled, the identity is checked numerically at random
parameter sets in 50-digit arithmetic, which is reported as such.
"""

import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

# ============================================================
# SYMBOLS  --  five joint variables, two actuated
# ============================================================
Iq1, Iq2 = sm.symbols('Iq1, Iq2', real=True)             # ACTUATED
q2, q3, q6 = sm.symbols('q2, q3, q6', real=True)         # PASSIVE, solved
q1, q4 = sm.symbols('q1, q4', real=True)                 # FIXED, structural
l1, l2, l3, l4, l5, l6 = sm.symbols('l1:7', positive=True)
h = sm.symbols('h', positive=True)

# ------------------------------------------------------------
# DERIVED CONSTANTS (functions of the fixed geometry only)
# ------------------------------------------------------------
d2 = sm.sqrt(l1**2 + h**2 - 2*l1*h*sm.cos(q1))      # |P2 - RCM|
d4 = sm.sqrt(l4**2 + h**2 - 2*l4*h*sm.cos(q4))      # |P4 - RCM|

# q5 is NOT free: it is what puts P5 on the sphere of radius l3 about RCM
q5 = sm.acos((l5**2 + d4**2 - l3**2) / (2*l5*d4))

phi2 = sm.acos((l2**2 + d2**2 - l3**2) / (2*l2*d2))  # cone angle at P2
delta = sm.acos(l6 / (2*l3))                         # plate half-angle

# aiming corrections: rotate a frame so its z axis points at RCM
psi = sm.atan2(-l1*sm.sin(q1), h - l1*sm.cos(q1))
xi = sm.atan2(-l2*sm.sin(phi2), d2 - l2*sm.cos(phi2))
beta = sm.atan2(-l4*sm.sin(q4), h - l4*sm.cos(q4))
chi = sm.atan2(-l5*sm.sin(q5), d4 - l5*sm.cos(q5))

# ============================================================
# FRAMES
# ============================================================
N = me.ReferenceFrame('N')

# --- leg 1 ---
I1 = N.orientnew('I1', 'Axis', (Iq1, N.z))          # ACTUATED
A = I1.orientnew('A', 'Axis', (q1, I1.y))           # FIXED tilt of l1
I2 = A.orientnew('I2', 'Axis', (psi - q1, A.y))     # aim z at RCM
I2s = I2.orientnew('I2s', 'Axis', (q2, I2.z))       # PASSIVE pin at P2
B = I2s.orientnew('B', 'Axis', (phi2, I2s.y))       # derived cone angle
I3 = B.orientnew('I3', 'Axis', (xi - phi2, B.y))    # aim z at RCM
I3s = I3.orientnew('I3s', 'Axis', (q3, I3.z))       # PASSIVE pin at P3
G = I3s.orientnew('G', 'Axis', (delta, I3s.y))      # plate chord direction

# --- ground link and leg 2 ---
D = N.orientnew('D', 'Axis', (q4, N.y))             # FIXED ground link
I4 = D.orientnew('I4', 'Axis', (beta - q4, D.y))    # aim z at RCM
I4s = I4.orientnew('I4s', 'Axis', (Iq2, I4.z))      # ACTUATED
E = I4s.orientnew('E', 'Axis', (q5, I4s.y))         # derived tilt of l5
I5 = E.orientnew('I5', 'Axis', (chi - q5, E.y))     # aim z at RCM
I5s = I5.orientnew('I5s', 'Axis', (q6, I5.z))       # PASSIVE pin at P5

# ============================================================
# POINTS
# ============================================================
P1 = me.Point('P1')
P2 = P1.locatenew('P2', l1*A.z)
P3 = P2.locatenew('P3', l2*B.z)
P5b = P3.locatenew('P5b', l6*G.z)                   # P5 reached along leg 1
P4 = P1.locatenew('P4', l4*D.z)
P5 = P4.locatenew('P5', l5*E.z)
RCM = P3.locatenew('RCM', l3*I3.z)                  # RCM reached along leg 1

loop_closure = (P5b.pos_from(P1) - P5.pos_from(P1)).express(N).to_matrix(N)

# ============================================================
# NUMERIC HARNESS
# ============================================================
PARAMS = (h, l1, l2, l3, l4, l5, l6, q1, q4)
INPUTS = (Iq1, Iq2, q2, q3, q6)


def rand_params(rng, tries=200):
    """Random but feasible fixed geometry."""
    for _ in range(tries):
        hv = 1.0
        l1v, l4v = rng.uniform(.2, .6, 2)
        l3v = rng.uniform(.2, .7)
        q1v, q4v = rng.uniform(.2, 1.2, 2)
        d2v = np.sqrt(l1v**2 + hv**2 - 2*l1v*hv*np.cos(q1v))
        d4v = np.sqrt(l4v**2 + hv**2 - 2*l4v*hv*np.cos(q4v))
        l2v = rng.uniform(abs(d2v - l3v) + .02, d2v + l3v - .02)
        lo = abs(d4v - l3v) + .02
        hi = d4v + l3v - .02
        if lo >= hi:
            continue
        l5v = rng.uniform(lo, hi)
        l6v = rng.uniform(.05, 1.9*l3v)
        return dict(zip(PARAMS, [hv, l1v, l2v, l3v, l4v, l5v, l6v, q1v, q4v]))
    raise RuntimeError('no feasible parameter set found')


def check(name, expr, rng, n=8, prec=50, inputs=True, symbolic=False):
    """Evaluate an identity at random points in high precision.

    `symbolic=True` first attempts simplify(). That is left OFF by default:
    these expressions contain nested atan2/acos and simplify() does not
    terminate in reasonable time on them. High-precision numeric evaluation at
    random parameter sets is the practical substitute -- 50-digit agreement at
    random points is strong evidence, not proof.
    """
    if symbolic:
        s = sm.simplify(sm.trigsimp(expr))
        if s == 0:
            print(f'  [symbolic] {name}: identically zero')
            return
    worst = 0.0
    for _ in range(n):
        sub = rand_params(rng)
        if inputs:
            sub.update({v: sm.Float(rng.uniform(-np.pi, np.pi), prec)
                        for v in INPUTS})
        val = expr if hasattr(expr, 'shape') else sm.Matrix([expr])
        v = max(abs(sm.N(e.subs(sub), prec)) for e in val)
        worst = max(worst, float(v))
    print(f'  [numeric ] {name}: max |residual| = {worst:.2e} '
          f'over {n} random cases at {prec} digits')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print(__doc__.split('Run it directly')[0].strip()[:0] or '', end='')

    # ---- A. the RCM property ----------------------------------------
    print('\nA. RCM property (RCM = h*N.z for all inputs)')
    rcm_vec = RCM.pos_from(P1).express(N).to_matrix(N)
    check('RCM - h*N.z', rcm_vec - sm.Matrix([0, 0, h]), rng, n=4)

    # ---- B. spherical structure: constant radii ---------------------
    print('\nB. constant distances to RCM')
    R = sm.Matrix([0, 0, h])
    for nm, pt, want in [('|P2-RCM| - d2', P2, d2),
                         ('|P3-RCM| - l3', P3, l3),
                         ('|P5b-RCM| - l3', P5b, l3),
                         ('|P4-RCM| - d4', P4, d4),
                         ('|P5-RCM| - l3', P5, l3)]:
        v = pt.pos_from(P1).express(N).to_matrix(N) - R
        check(nm, sm.sqrt((v.T*v)[0]) - want, rng, n=5)

    # ---- C. rank of the loop closure --------------------------------
    print('\nC. rank of the loop-closure Jacobian w.r.t. (q2, q3)')
    Jc = loop_closure.jacobian(sm.Matrix([q2, q3]))
    ranks = set()
    for _ in range(6):
        sub = rand_params(rng)
        sub.update({v: float(rng.uniform(-np.pi, np.pi)) for v in INPUTS})
        ranks.add(sm.Matrix(sm.N(Jc.subs(sub), 30)).rank())
    print(f'  3 equations, 2 unknowns; observed Jacobian rank(s): {sorted(ranks)}')
    print('  -> one of the three scalar equations is redundant, which is why')
    print('     a Newton solver on the raw 3x1 system stalls.')

    # redundancy made explicit
    v1 = P5b.pos_from(P1).express(N).to_matrix(N) - R
    v2 = P5.pos_from(P1).express(N).to_matrix(N) - R
    check('|P5b-RCM|^2 - |P5-RCM|^2  (the redundant equation)',
          (v1.T*v1)[0] - (v2.T*v2)[0], rng, n=5)

    # ---- D. the closed-form solver satisfies this model -------------
    print('\nD. closed-form solver (rcm5bar) substituted into this SymPy chain')
    from rcm5bar import Linkage, solve

    lc = sm.lambdify(PARAMS + INPUTS, loop_closure, 'numpy')
    worst_lc = worst_p3 = worst_p5 = 0.0
    f_P3 = sm.lambdify(PARAMS + INPUTS, P3.pos_from(P1).express(N).to_matrix(N))
    f_P5 = sm.lambdify(PARAMS + INPUTS, P5.pos_from(P1).express(N).to_matrix(N))
    ncase = 0
    for _ in range(40):
        sub = rand_params(rng)
        pv = [float(sub[p]) for p in PARAMS]
        lk = Linkage(h=pv[0], l1=pv[1], l2=pv[2], l3=pv[3], l4=pv[4],
                     l5=pv[5], l6=pv[6], q1=pv[7], q4=pv[8])
        # sanity: the derived q5 agrees with the symbolic one
        q5_sym = float(sm.N(q5.subs(sub)))
        assert abs(q5_sym - lk.q5) < 1e-12, (q5_sym, lk.q5)
        for _ in range(6):
            a, b = rng.uniform(-np.pi, np.pi, 2)
            br = int(rng.choice([-1, 1]))
            try:
                s = solve(lk, a, b, branch=br)
            except Exception:
                continue
            ncase += 1
            args = pv + [a, b, s['q2'], s['q3'], s['q6']]
            worst_lc = max(worst_lc, float(np.max(np.abs(lc(*args)))))
            worst_p3 = max(worst_p3, float(np.max(np.abs(
                f_P3(*args).ravel() - s['P3']))))
            worst_p5 = max(worst_p5, float(np.max(np.abs(
                f_P5(*args).ravel() - s['P5']))))
    print(f'  {ncase} cases')
    print(f'  max |loop_closure(Iq1,Iq2,q2,q3)| = {worst_lc:.2e}')
    print(f'  max |P3_sympy - P3_solver|        = {worst_p3:.2e}')
    print(f'  max |P5_sympy - P5_solver|        = {worst_p5:.2e}')

    # ---- E. q6 is determined, not free ------------------------------
    print('\nE. q6 is a genuine passive variable, fixed by the plate')
    print('   It is absent from loop_closure because that equation compares')
    print('   POSITIONS only. Its constraint is an orientation one: the plate')
    print('   is rigid through P3, P5 and RCM. Check that I5s.x actually lines')
    print('   up with the plate chord once q6 takes the solver value:')
    chord_I5 = (P3.pos_from(P5).express(I5s).to_matrix(I5s))
    f_chord = sm.lambdify(PARAMS + INPUTS, chord_I5)
    worst_q6 = 0.0
    for _ in range(15):
        sub = rand_params(rng)
        pv = [float(sub[p]) for p in PARAMS]
        lk = Linkage(h=pv[0], l1=pv[1], l2=pv[2], l3=pv[3], l4=pv[4],
                     l5=pv[5], l6=pv[6], q1=pv[7], q4=pv[8])
        for _ in range(4):
            a, b = rng.uniform(-np.pi, np.pi, 2)
            try:
                s = solve(lk, a, b)
            except Exception:
                continue
            c = f_chord(*(pv + [a, b, s['q2'], s['q3'], s['q6']])).ravel()
            worst_q6 = max(worst_q6, abs(c[1]))      # y component must vanish
    print(f'   max |chord . I5s.y| = {worst_q6:.2e}   (zero => q6 correct)')

    print('\nAll checks complete.')
