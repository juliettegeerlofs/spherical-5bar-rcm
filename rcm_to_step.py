"""rcm_to_step.py -- write the mechanism (home pose) as a STEP wireframe.

Why STEP and not Onshape's own sketch/feature API directly: STEP (ISO 10303)
is a stable, documented, vendor-neutral format that any CAD tool -- including
Onshape -- can import and auto-translate into native geometry. Onshape's own
low-level feature-JSON schema changes across API versions and is easy to get
subtly wrong without testing against a live account, which this environment
cannot do. A STEP file keeps the one genuinely uncertain step (does Onshape's
importer accept this file) as small and standard as possible.

Every point and length below comes straight from rcm5bar.Linkage / solve(),
not retyped -- so this can never drift from the numbers already verified
elsewhere in this project (see rcm_bouwtekening.html for the same values).

Produces a minimal wireframe: 6 points (P1, P2, P3, P4, P5, RCM) and the 5
rod centrelines as bounded edges (P1-P2, P2-P3, P1-P4, P4-P5, P3-P5), plus
the 3 scharnier axes (P2-RCM, P3-RCM, P5-RCM) on a second, separate curve set
so you can hide them in Onshape without hiding the rods.

    python rcm_to_step.py [--h-mm 150] [--out rcm_home.step]
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

from rcm5bar import Linkage, solve
from rcm_view import best_designs, params_of, SOURCES_V2
from hole_search import NAMES, EVAL_KW


def build_geometry(h_mm: float):
    """(points_mm, rod_edges, axis_edges) for the best v2 (home_cone) design,
    at the home pose, scaled to h_mm millimetres."""
    rows = best_designs("home_cone", 1, paths=SOURCES_V2)
    p = params_of(rows[0])
    h = h_mm / 1000.0
    scaled = {n: (v * h if n.startswith("l") else v) for n, v in p.items()}
    lk = Linkage(h=h, **scaled)
    s = solve(lk, 0.0, 0.0, branch=EVAL_KW["branch"])

    pts = {k: tuple(float(c) * 1000.0 for c in s[k]) for k in
           ("P1", "P2", "P3", "P4", "P5", "RCM")}
    rod_edges = [("P1", "P2"), ("P2", "P3"), ("P1", "P4"),
                ("P4", "P5"), ("P3", "P5")]
    axis_edges = [("P2", "RCM"), ("P3", "RCM"), ("P5", "RCM")]
    return pts, rod_edges, axis_edges


def _direction(a, b):
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    n = (dx * dx + dy * dy + dz * dz) ** 0.5
    return dx / n, dy / n, dz / n


def write_step(path, pts, rod_edges, axis_edges):
    """A minimal, valid STEP AP214 file: two GEOMETRIC_CURVE_SETs (rods,
    axes) of bounded EDGE_CURVEs between named CARTESIAN_POINTs. No solids --
    this is a wireframe reference to build sketches/parts from in Onshape,
    not a finished model."""
    lines = []
    ids = {}
    nxt = [1]

    def new_id():
        i = nxt[0]
        nxt[0] += 1
        return i

    def add(entity_text):
        i = new_id()
        lines.append(f"#{i} = {entity_text};")
        return i

    point_id = {}
    vertex_id = {}
    for name, (x, y, z) in pts.items():
        pid = add(f"CARTESIAN_POINT('{name}',({x:.6f},{y:.6f},{z:.6f}))")
        point_id[name] = pid
        vertex_id[name] = add(f"VERTEX_POINT('{name}',#{pid})")

    def edge_curve(a, b):
        dx, dy, dz = _direction(pts[a], pts[b])
        dir_id = add(f"DIRECTION('',({dx:.9f},{dy:.9f},{dz:.9f}))")
        vec_id = add(f"VECTOR('',#{dir_id},1000.)")
        line_id = add(f"LINE('',#{point_id[a]},#{vec_id})")
        return add(f"EDGE_CURVE('{a}-{b}',#{vertex_id[a]},#{vertex_id[b]},"
                   f"#{line_id},.T.)")

    rod_edge_ids = [edge_curve(a, b) for a, b in rod_edges]
    axis_edge_ids = [edge_curve(a, b) for a, b in axis_edges]

    rod_set = ",".join(f"#{i}" for i in rod_edge_ids)
    axis_set = ",".join(f"#{i}" for i in axis_edge_ids)
    rod_set_id = add(f"GEOMETRIC_CURVE_SET('rods',({rod_set}))")
    axis_set_id = add(f"GEOMETRIC_CURVE_SET('scharnierassen (naar RCM)',"
                      f"({axis_set}))")

    # The standard AP214 unit + uncertainty context, as a compound instance
    # (the "( TYPE1(...) TYPE2(...) ... )" syntax): every real STEP exporter
    # emits this exact skeleton, and it's what was missing before -- a bare
    # GEOMETRIC_REPRESENTATION_CONTEXT with no units/uncertainty attached is
    # not enough for Onshape's translator to recognise the file as having
    # "translatable geometry", even though the curve data itself was valid.
    mm_id = new_id()
    lines.append(f"#{mm_id} = ( LENGTH_UNIT() NAMED_UNIT(*) "
                 f"SI_UNIT(.MILLI.,.METRE.) );")
    rad_id = new_id()
    lines.append(f"#{rad_id} = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() "
                 f"SI_UNIT($,.RADIAN.) );")
    sr_id = new_id()
    lines.append(f"#{sr_id} = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) "
                 f"SOLID_ANGLE_UNIT() );")
    unc_id = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),"
                 f"#{mm_id},'DISTANCE_ACCURACY_VALUE',"
                 f"'Maximum model space distance')")
    ctx_id = new_id()
    lines.append(f"#{ctx_id} = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
                 f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc_id})) "
                 f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm_id},#{rad_id},"
                 f"#{sr_id})) "
                 f"REPRESENTATION_CONTEXT('','3D') );")

    # the product/shape-definition wrapper every AP214 consumer expects
    # around the actual geometry
    app_ctx_id = add("APPLICATION_CONTEXT('automotive design')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard',"
        f"'automotive_design',2000,#{app_ctx_id})")
    prod_ctx_id = add(f"PRODUCT_CONTEXT('',#{app_ctx_id},'mechanical')")
    prod_id = add(f"PRODUCT('RCM_MECHANISM','RCM mechanism, home pose','',"
                  f"(#{prod_ctx_id}))")
    pdc_id = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',"
                f"#{app_ctx_id},'design')")
    pdf_id = add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod_id})")
    pd_id = add(f"PRODUCT_DEFINITION('design','',#{pdf_id},#{pdc_id})")
    pds_id = add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd_id})")
    # NOT plain SHAPE_REPRESENTATION: that's most often expected to hold a
    # solid (MANIFOLD_SOLID_BREP), and a translator that only looks for that
    # will report "no translatable geometry" for pure curve content even
    # inside an otherwise-correct product/shape wrapper -- which is exactly
    # what happened on the first attempt. This subtype exists specifically
    # to mark a representation as intentionally wireframe-only.
    rep_id = add(f"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION('',"
                f"(#{rod_set_id},#{axis_set_id}),#{ctx_id})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds_id},#{rep_id})")

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    header = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('RCM mechanism wireframe, home pose'),'2;1');
FILE_NAME('rcm_home.step','{ts}',('rcm_to_step.py'),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 3 1 1 }}'));
ENDSEC;
DATA;"""
    footer = "ENDSEC;\nEND-ISO-10303-21;\n"

    with open(path, "w") as fh:
        fh.write(header + "\n")
        fh.write("\n".join(lines))
        fh.write("\n" + footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-mm", type=float, default=150.0)
    ap.add_argument("--out", default="rcm_home.step")
    args = ap.parse_args()

    pts, rod_edges, axis_edges = build_geometry(args.h_mm)
    write_step(args.out, pts, rod_edges, axis_edges)

    print(f"wrote {args.out}  (h = {args.h_mm:.1f} mm)")
    print("punten:")
    for k, v in pts.items():
        print(f"  {k:4} ({v[0]:8.3f}, {v[1]:8.3f}, {v[2]:8.3f}) mm")
    print("\nImporteer dit bestand handmatig in Onshape (Insert -> Import) om "
          "te controleren dat de geometrie klopt, VOORDAT je "
          "push_to_onshape.py gebruikt om het automatisch te uploaden.")


if __name__ == "__main__":
    main()
