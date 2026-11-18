"""rcm_to_dxf.py -- write the mechanism (home pose) as a 3D DXF wireframe.

Fallback after rcm_to_step.py's STEP file failed Onshape's import three
times in a row with the same "no translatable geometry" error, even after
fixing the AP214 product/shape wrapper and switching to the wireframe-
specific GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION entity. Three
different STEP structures giving the identical error suggests Onshape's STEP
translator may simply not treat a curve-only (no solid, no surface) file as
"translatable" at all -- not something more STEP tweaking is likely to fix
blindly, so this tries a structurally different format instead.

DXF (R12 / AC1009, the oldest and most widely supported variant) is a flat,
group-code based format -- no product/shape wrapper, no representation-
context linking, just entities. LINE and POINT both carry full x/y/z (group
codes 10/20/30, and 11/21/31 for a LINE's second endpoint), so this is
genuinely 3D, not flattened to a sketch plane -- AS LONG AS Onshape's DXF
importer preserves that. That is the one real remaining uncertainty here:
DXF has 2D drafting roots, and some import pipelines force everything onto
one plane regardless of what the file says. If this also fails, the likely
next step is not another file format but a small pasted FeatureScript
snippet instead, which sidesteps "will the importer recognise this" as a
question entirely, since there is no import step -- Onshape runs the
coordinates directly. Say the word if it comes to that.

Reuses build_geometry() from rcm_to_step.py so the numbers can never drift
between the two attempts.

    python rcm_to_dxf.py [--h-mm 150] [--out rcm_home.dxf]
"""

from __future__ import annotations

import argparse

from rcm_to_step import build_geometry


def _line(code, value):
    return f"{code}\n{value}\n"


def write_dxf(path, pts, rod_edges, axis_edges):
    out = []
    out.append(_line(0, "SECTION"))
    out.append(_line(2, "ENTITIES"))

    def add_line(name_a, name_b, layer):
        ax, ay, az = pts[name_a]
        bx, by, bz = pts[name_b]
        out.append(_line(0, "LINE"))
        out.append(_line(8, layer))
        out.append(_line(10, f"{ax:.6f}"))
        out.append(_line(20, f"{ay:.6f}"))
        out.append(_line(30, f"{az:.6f}"))
        out.append(_line(11, f"{bx:.6f}"))
        out.append(_line(21, f"{by:.6f}"))
        out.append(_line(31, f"{bz:.6f}"))

    def add_point(name, layer):
        x, y, z = pts[name]
        out.append(_line(0, "POINT"))
        out.append(_line(8, layer))
        out.append(_line(10, f"{x:.6f}"))
        out.append(_line(20, f"{y:.6f}"))
        out.append(_line(30, f"{z:.6f}"))

    for a, b in rod_edges:
        add_line(a, b, "STAVEN")
    for a, b in axis_edges:
        add_line(a, b, "SCHARNIERASSEN")
    for name in pts:
        add_point(name, "PUNTEN")

    out.append(_line(0, "ENDSEC"))
    out.append(_line(0, "EOF"))

    with open(path, "w") as fh:
        fh.write("".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-mm", type=float, default=150.0)
    ap.add_argument("--out", default="rcm_home.dxf")
    args = ap.parse_args()

    pts, rod_edges, axis_edges = build_geometry(args.h_mm)
    write_dxf(args.out, pts, rod_edges, axis_edges)

    print(f"wrote {args.out}  (h = {args.h_mm:.1f} mm)")
    print("3 layers: STAVEN (de 5 staven), SCHARNIERASSEN (3 assen naar "
          "RCM), PUNTEN (alle 6 punten als losse POINT-entiteiten)")
    print("\nLet op bij het importeren: check na import of de punten nog "
          "hun echte Z-hoogte hebben (bijv. RCM op z=150mm) en niet allemaal "
          "op z=0 zijn platgeslagen -- dat is de bekende zwakke plek van "
          "DXF-import.")


if __name__ == "__main__":
    main()
