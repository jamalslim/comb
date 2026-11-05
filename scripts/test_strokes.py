#!/usr/bin/env python3
"""Tests of comb/strokes.py on shapes whose answers are known: a ring (closed, winding +-1),
a horizontal bar (only east/west moves), two separate bars (one pen-up jump), coverage of the
skeleton by the walk on real digits, token conventions and the IDX parser."""
import sys, pathlib, gzip, struct
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits import strokes as st


def ring(r=7, w=1.6, size=28):
    yy, xx = np.indices((size, size)) - size / 2 + 0.5
    return ((np.abs(np.hypot(yy, xx) - r) < w) * 255).astype(np.uint8)


def test_ring():
    tok, start, L, P = st.image_to_tokens(ring(), d=200)
    g = st.geometry(tok[:L])
    assert (tok[:L] < 8).all(), "a ring is one stroke: no pen-up moves expected"
    assert g[1] <= 1.5, f"closure distance {g[1]}"
    assert abs(abs(g[4]) - 1) < 0.15, f"winding {g[4]}"
    print(f"  ring: {L} moves, closure {g[1]:.2f}, winding {g[4]:+.2f}, width {g[2]:.0f}, height {g[3]:.0f}")


def test_bar_and_jump():
    img = np.zeros((28, 28), np.uint8)
    img[14, 6:22] = 255
    tok, _, L, _ = st.image_to_tokens(img, d=100)
    assert set(np.unique(tok[:L]).tolist()) <= {0, 4}, np.unique(tok[:L])
    img2 = img.copy()
    img2[20, 6:22] = 255
    tok2, _, L2, _ = st.image_to_tokens(img2, d=100)
    up = (tok2[:L2] >= 8).sum()
    assert up > 0 and st.geometry(tok2[:L2])[6] == up
    print(f"  bar: {L} east/west moves; two bars: {L2} moves with {up} pen-up moves between them")


def test_coverage_on_digits(n=200):
    from scipy import ndimage
    from sklearn.datasets import load_digits
    D = load_digits()
    bad = 0
    for img8 in D.images[:n]:
        z = ndimage.zoom(img8 / 16.0, 3.0, order=3).clip(0, 1)
        x = np.zeros((28, 28))
        x[2:26, 2:26] = z
        x = (x * 255).round().astype(np.uint8)
        tok, s0, L, _ = st.image_to_tokens(x, d=10_000, threshold=110)
        pos, down = st.pen_path(tok[:L])
        pos = pos + np.array(s0)
        drawn = set()
        for k, dn in enumerate(down):
            if dn:
                drawn.add(tuple(pos[k]))
                drawn.add(tuple(pos[k + 1]))
        comps = [c for c in st._components(set(zip(*np.nonzero(st.thin(x >= 110))))) if len(c) >= 3]
        need = set().union(*comps) if comps else set()
        bad += len(need - drawn) > 0 or len(drawn - need) > 0
    assert bad == 0, f"{bad} walks do not draw exactly their skeleton"
    print(f"  coverage: {n} real digits, every walk draws exactly its skeleton")


def test_conventions_and_idx():
    assert st.K_TOKENS == 16 and st.PAD == 8 and tuple(st.DIRS[0]) == (0, 1) and tuple(st.DIRS[2]) == (-1, 0)
    img = np.zeros((28, 28))
    assert st.render(np.full(10, st.PAD)).sum() == 0, "padding must not draw"
    arr = (np.arange(2 * 3 * 3) % 256).astype(np.uint8).reshape(2, 3, 3)
    raw = struct.pack(">HBB", 0, 8, 3) + struct.pack(">III", 2, 3, 3) + arr.tobytes()
    assert np.array_equal(st.parse_idx(gzip.compress(raw)), arr) and np.array_equal(st.parse_idx(raw), arr)
    print("  conventions and IDX parser ok")


if __name__ == "__main__":
    test_ring()
    test_bar_and_jump()
    test_coverage_on_digits()
    test_conventions_and_idx()
    print("all stroke tests passed")
