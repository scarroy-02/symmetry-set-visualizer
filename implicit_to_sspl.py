#!/usr/bin/env python3
"""
implicit_to_sspl.py
===================

Generate a .sspl file (cubic B-spline control points) from an implicit plane
curve f(x, y) = 0.  The output drops directly into index.html / test.html via
the Load button.

Examples
--------
List built-in implicit curves:
    python implicit_to_sspl.py --list

Use a predefined curve:
    python implicit_to_sspl.py --preset lemniscate -o assets/lemniscate.sspl

Pass a custom expression:
    python implicit_to_sspl.py --expr "x**2 + y**2 - 1" -o circle.sspl -n 12

Override the sampling box / resolution / control-point count:
    python implicit_to_sspl.py --expr "x**3 - 3*x*y**2 - 1" \\
        --bbox -2,2,-2,2 --res 600 -n 32 -o trefoil.sspl

Dependencies: numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

import numpy as np

# (expression, bbox, suggested control-point count)
PRESETS: dict[str, tuple[str, tuple[float, float, float, float], int]] = {
    "circle":       ("x**2 + y**2 - 1",                                            (-1.5, 1.5, -1.5, 1.5), 12),
    "ellipse":      ("(x/2)**2 + y**2 - 1",                                        (-2.5, 2.5, -1.5, 1.5), 12),
    "lemniscate":   ("(x**2 + y**2)**2 - 2*(x**2 - y**2)",                         (-2.0, 2.0, -1.2, 1.2), 24),
    "cardioid":     ("(x**2 + y**2 - 2*x)**2 - 4*(x**2 + y**2)",                   (-1.5, 4.5, -3.0, 3.0), 20),
    "astroid":      ("(x**2 + y**2 - 1)**3 + 27*x**2*y**2",                        (-1.5, 1.5, -1.5, 1.5), 16),
    "trefoil_rose": ("(x**2 + y**2)**3 - 4*x**2*y**2",                             (-1.5, 1.5, -1.5, 1.5), 24),
    "figure_eight": ("x**4 - x**2 + y**2",                                         (-1.2, 1.2, -1.0, 1.0), 20),
    "deltoid":      ("(x**2 + y**2)**2 + 18*(x**2 + y**2) - 27 - 8*x*(x**2 - 3*y**2)",
                                                                                  (-3.5, 3.5, -3.5, 3.5), 20),
    "hyperbola":    ("x**2 - y**2 - 1",                                            (-3.0, 3.0, -3.0, 3.0), 16),  # 2 open branches
    "parabola":     ("y - x**2",                                                   (-2.0, 2.0, -0.5, 4.0), 16),  # open arc
}


SAFE_NS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
    "pi": np.pi, "e": np.e,
}


def trace_zero_set(expr: str, bbox: tuple[float, float, float, float], res: int) -> list[np.ndarray]:
    """Return polylines (Nx2 arrays) tracing f(x, y) = 0 inside the bbox."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_lo, x_hi, y_lo, y_hi = bbox
    xs = np.linspace(x_lo, x_hi, res)
    ys = np.linspace(y_lo, y_hi, res)
    X, Y = np.meshgrid(xs, ys)
    ns = dict(SAFE_NS)
    ns["x"] = X
    ns["y"] = Y
    Z = eval(compile(expr, "<implicit>", "eval"), ns)  # noqa: S307 (CLI use)
    if np.isscalar(Z) or np.asarray(Z).shape != X.shape:
        raise ValueError(f"Expression did not produce a 2-D array of shape {X.shape}")

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(X, Y, Z, levels=[0.0])
    finally:
        plt.close(fig)

    segs = cs.allsegs[0] if cs.allsegs else []
    return [np.asarray(s, dtype=float) for s in segs if len(s) >= 4]


def is_closed(arr: np.ndarray, tol: float) -> bool:
    return float(np.linalg.norm(arr[0] - arr[-1])) < tol


def resample_arclen(arr: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """Return n points evenly spaced along arc length."""
    if closed:
        if np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
        arr_w = np.vstack([arr, arr[:1]])
        targets_endpoint = False
    else:
        arr_w = arr
        targets_endpoint = True

    seg = np.linalg.norm(np.diff(arr_w, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("Degenerate polyline (zero arc length)")

    targets = np.linspace(0.0, total, n, endpoint=targets_endpoint)
    out = np.empty((n, 2))
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        denom = s[j + 1] - s[j] if j + 1 < len(s) else 0.0
        a = 0.0 if denom == 0 else (t - s[j]) / denom
        out[i] = arr_w[j] * (1.0 - a) + arr_w[j + 1] * a
    return out


def fit_view(point_arrays: Iterable[np.ndarray], canvas_px: int = 900) -> dict[str, float]:
    pts = np.vstack(list(point_arrays))
    cx = float((pts[:, 0].min() + pts[:, 0].max()) / 2)
    cy = float((pts[:, 1].min() + pts[:, 1].max()) / 2)
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6))
    return {"x": -cx, "y": cy, "scale": canvas_px / (span * 1.5)}


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(t) for t in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be 'x_lo,x_hi,y_lo,y_hi'")
    x_lo, x_hi, y_lo, y_hi = parts
    if x_lo >= x_hi or y_lo >= y_hi:
        raise argparse.ArgumentTypeError("bbox must satisfy x_lo<x_hi and y_lo<y_hi")
    return (x_lo, x_hi, y_lo, y_hi)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="List built-in presets and exit.")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="Built-in implicit curve.")
    ap.add_argument("--expr", help='Custom implicit expression in x and y, e.g. "x**2 + y**2 - 1".')
    ap.add_argument("--bbox", type=parse_bbox, help="Sampling box: x_lo,x_hi,y_lo,y_hi")
    ap.add_argument("--res", type=int, default=400, help="Grid samples per axis (default 400).")
    ap.add_argument("-n", "--num-points", type=int, help="Control points per branch.")
    ap.add_argument("-o", "--out", help="Output .sspl path.")
    args = ap.parse_args()

    if args.list:
        print("Available presets:")
        for name, (expr, bbox, n) in sorted(PRESETS.items()):
            print(f"  {name:<14}  f = {expr}")
            print(f"  {'':<14}  bbox = {bbox}, suggested n = {n}")
        return 0

    if not (args.preset or args.expr):
        ap.error("one of --preset, --expr, or --list is required")
    if not args.out:
        ap.error("-o / --out is required (unless --list)")

    if args.preset:
        expr, default_bbox, default_n = PRESETS[args.preset]
    else:
        expr, default_bbox, default_n = args.expr, (-2.0, 2.0, -2.0, 2.0), 16

    bbox = args.bbox or default_bbox
    n = args.num_points or default_n
    if n < 4:
        ap.error("--num-points must be >= 4 (B-spline interpolation needs enough samples)")

    polys = trace_zero_set(expr, bbox, args.res)
    if not polys:
        print(f"No zero-set found for f(x,y) = {expr} on bbox {bbox}", file=sys.stderr)
        return 1

    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    closed_tol = span * 1e-3

    curves_out: list[list[dict[str, float]]] = []
    open_flags: list[bool] = []
    sampled_arrays: list[np.ndarray] = []
    for arr in polys:
        closed = is_closed(arr, closed_tol)
        try:
            sampled = resample_arclen(arr, n, closed)
        except ValueError as e:
            print(f"  skipping a branch: {e}", file=sys.stderr)
            continue
        curves_out.append([{"x": float(p[0]), "y": float(p[1])} for p in sampled])
        open_flags.append(not closed)
        sampled_arrays.append(sampled)

    if not curves_out:
        print("All branches were degenerate.", file=sys.stderr)
        return 1

    data = {
        "format": "symmetry-set-spline",
        "version": 1,
        "curves": curves_out,
        "curveOpen": open_flags,
        "activeCurveIdx": 0,
        "view": fit_view(sampled_arrays),
    }
    with open(args.out, "w") as fp:
        json.dump(data, fp, indent=2)

    print(f"Wrote {args.out}: {len(curves_out)} branch(es), {n} control points each.")
    for i, (c, is_open) in enumerate(zip(curves_out, open_flags)):
        print(f"  curve {i}: {'open' if is_open else 'closed'} ({len(c)} pts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
