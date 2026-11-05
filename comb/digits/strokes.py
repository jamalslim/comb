"""Digits as pen-move sequences: MNIST -> skeleton -> ordered walk -> 16 tokens; tokens -> image.

Token c in 0..15:  direction = c mod 8 (E, NE, N, NW, W, SW, S, SE, a unit step on the pixel grid),
pen = c < 8 (down: draws; up: moves without drawing).  Pen-up moves also pad a finished digit, so
no end symbol is needed: a padded sequence just drifts off the canvas without drawing.
K = 16 tokens = 4 work qubits; the token is the count of the round.
"""
import gzip, pathlib, struct, urllib.request
import numpy as np

K_TOKENS = 16
PAD = 8                                            # pen-up east
DIRS = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)])   # (drow, dcol)
_DIR_INDEX = {(int(r), int(c)): i for i, (r, c) in enumerate(DIRS)}
ANGLES = np.arange(8) * (np.pi / 4)

MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]
FILES = {"x_train": "train-images-idx3-ubyte.gz", "y_train": "train-labels-idx1-ubyte.gz",
         "x_test": "t10k-images-idx3-ubyte.gz", "y_test": "t10k-labels-idx1-ubyte.gz"}


# --------------------------------------------------------------------------- MNIST
def parse_idx(raw):
    """IDX file (possibly gzipped bytes) -> numpy array."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    zero, dtype, ndim = struct.unpack(">HBB", raw[:4])
    assert zero == 0 and dtype == 8, "only unsigned-byte IDX files are supported"
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(shape)


def load_mnist(cache_dir="data"):
    """Returns dict with x_train (60000, 28, 28) uint8, y_train, x_test, y_test.  Tries a cached npz,
    then keras, then torchvision, then a direct download of the IDX files."""
    cache = pathlib.Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    npz = cache / "mnist.npz"
    if npz.exists():
        z = np.load(npz)
        return {k: z[k] for k in FILES}
    out = None
    try:
        from tensorflow.keras.datasets import mnist                                         # noqa
        (xtr, ytr), (xte, yte) = mnist.load_data()
        out = dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    except Exception:
        pass
    if out is None:
        try:
            from torchvision import datasets                                                # noqa
            tr = datasets.MNIST(str(cache), train=True, download=True)
            te = datasets.MNIST(str(cache), train=False, download=True)
            out = dict(x_train=tr.data.numpy(), y_train=tr.targets.numpy(), x_test=te.data.numpy(), y_test=te.targets.numpy())
        except Exception:
            pass
    if out is None:
        out, err = {}, None
        for key, fname in FILES.items():
            for base in MIRRORS:
                try:
                    out[key] = parse_idx(urllib.request.urlopen(base + fname, timeout=60).read())
                    break
                except Exception as ex:
                    err = ex
            if key not in out:
                raise RuntimeError(f"could not load MNIST ({fname}): {err}. Put mnist.npz with keys x_train, y_train, x_test, y_test in {cache}")
    out = {k: np.asarray(v, dtype=np.uint8) for k, v in out.items()}
    np.savez_compressed(npz, **out)
    return out


# --------------------------------------------------------------------------- thinning
def thin(img):
    """Zhang-Suen thinning of a boolean image -> one-pixel-wide skeleton (boolean)."""
    I = np.pad(np.asarray(img, dtype=bool), 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P2 = I[:-2, 1:-1]
            P3 = I[:-2, 2:]
            P4 = I[1:-1, 2:]
            P5 = I[2:, 2:]
            P6 = I[2:, 1:-1]
            P7 = I[2:, :-2]
            P8 = I[1:-1, :-2]
            P9 = I[:-2, :-2]
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(ring[:8]).astype(np.int16)
            A = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = (P2 * P4 * P6 == 0), (P4 * P6 * P8 == 0)
            else:
                c1, c2 = (P2 * P4 * P8 == 0), (P2 * P6 * P8 == 0)
            kill = (I[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if kill.any():
                I[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            break
    return I[1:-1, 1:-1].astype(bool)


# --------------------------------------------------------------------------- tracing
def _neighbors(p, S):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in S]


def _components(S):
    S = set(S)
    comps = []
    while S:
        seed = min(S)
        comp = {seed}
        stack = [seed]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, S):
                if q not in comp:
                    comp.add(q)
                    stack.append(q)
        comps.append(comp)
        S -= comp
    return comps


def _dir(p, q):
    return _DIR_INDEX[(q[0] - p[0], q[1] - p[1])]


def _turn(prev_dir, d):
    if prev_dir is None:
        return 0
    return min(abs(d - prev_dir) % 8, 8 - abs(d - prev_dir) % 8)


def _walk(comp, start):
    """Depth-first walk over a connected skeleton component, retracing when it backtracks,
    stopping as soon as every pixel has been visited.  The next pixel is the unvisited neighbour
    with the fewest unvisited onward neighbours (Warnsdorff's rule), ties broken by the smallest
    turn, so staircase corners are not skipped and loops close.  Returns the direction indices."""
    moves, visited, stack, pos, prev = [], {start}, [start], start, None
    onward = lambda q: sum(1 for r in _neighbors(q, comp) if r not in visited)
    while stack and len(visited) < len(comp):
        p = stack[-1]
        cand = [q for q in _neighbors(p, comp) if q not in visited]
        if cand:
            q = min(cand, key=lambda q: (onward(q), _turn(prev, _dir(p, q))))
            d = _dir(p, q)
            moves.append(d)
            visited.add(q)
            stack.append(q)
            prev = d
            pos = q
        else:
            stack.pop()
            if stack and len(visited) < len(comp):
                q = stack[-1]
                d = _dir(p, q)
                moves.append(d)
                prev = d
                pos = q
    return moves, pos


def _jump(p, q):
    """Pen-up unit moves from p to q (Chebyshev chain)."""
    moves = []
    r, c = p
    while (r, c) != tuple(q):
        dr, dc = int(np.sign(q[0] - r)), int(np.sign(q[1] - c))
        moves.append(_DIR_INDEX[(dr, dc)] + 8)
        r, c = r + dr, c + dc
    return moves


def image_to_tokens(img, d, threshold=128, min_px=3):
    """28 x 28 uint8 (or float in [0,1]) image -> token sequence of length d (padded or truncated).
    Also returns the start pixel, the untruncated length and the skeleton pixel count."""
    a = np.asarray(img)
    mask = (a >= threshold) if a.dtype == np.uint8 else (a >= threshold / 255.0)
    skel = thin(mask)
    pts = set(zip(*np.nonzero(skel)))
    comps = [c for c in _components(pts) if len(c) >= min_px]
    comps.sort(key=lambda c: min(c))                               # top-left component first
    tokens, pos, start = [], None, None
    for comp in comps:
        ends = [p for p in comp if len(_neighbors(p, comp)) == 1]
        s = min(ends) if ends else min(comp)
        if pos is not None:
            tokens += _jump(pos, s)
        elif start is None:
            start = s
        moves, pos = _walk(comp, s)
        tokens += moves
    length = len(tokens)
    tokens = (tokens + [PAD] * d)[:d]
    return np.array(tokens, dtype=np.int8), start if start is not None else (14, 14), length, len(pts)


def encode(images, d, threshold=128, min_px=3, log=None):
    """Batch version; returns tokens (N, d) int8, starts (N, 2), lengths (N,), skeleton sizes (N,)."""
    N = len(images)
    T = np.zeros((N, d), dtype=np.int8)
    S = np.zeros((N, 2), dtype=np.int16)
    L = np.zeros(N, dtype=np.int32)
    P = np.zeros(N, dtype=np.int32)
    for i in range(N):
        T[i], S[i], L[i], P[i] = image_to_tokens(images[i], d, threshold, min_px)
        if log and i % 5000 == 0 and i:
            log(f"  encoded {i}/{N}")
    return T, S, L, P


# --------------------------------------------------------------------------- rendering
def _gauss1d(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img, sigma):
    if sigma <= 0:
        return img
    k = _gauss1d(sigma)
    r = len(k) // 2
    p = np.pad(img, r, mode="constant")
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, p)
    p = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, p)
    return p


def pen_path(tokens):
    """Positions visited (relative to the start) and a pen-down flag per move."""
    tok = np.asarray(tokens, dtype=np.int64)
    dirs = DIRS[tok % 8]
    pos = np.concatenate([[[0, 0]], np.cumsum(dirs, axis=0)])
    return pos, tok < 8


def render(tokens, size=28, canvas=64, thickness=1, blur=0.6):
    """Token sequence -> size x size image in [0, 1], drawn from the canvas centre, thickened by a
    3 x 3 dilation `thickness` times, blurred, then centred by centre of mass like MNIST."""
    pos, down = pen_path(tokens)
    img = np.zeros((canvas, canvas))
    pos = pos + canvas // 2
    for i, dn in enumerate(down):
        if dn:
            for (r, c) in (pos[i], pos[i + 1]):
                if 0 <= r < canvas and 0 <= c < canvas:
                    img[r, c] = 1.0
    for _ in range(thickness):
        p = np.pad(img, 1)
        img = np.maximum.reduce([p[1 + dr:canvas + 1 + dr, 1 + dc:canvas + 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
    img = _blur(img, blur)
    if img.max() > 0:
        img = img / img.max()
    tot = img.sum()
    if tot <= 0:
        return np.zeros((size, size))
    rr, cc = np.indices(img.shape)
    cr = int(round((rr * img).sum() / tot))
    cc_ = int(round((cc * img).sum() / tot))
    out = np.zeros((size, size))
    h = size // 2
    r0, c0 = cr - h, cc_ - h
    rs, re = max(r0, 0), min(r0 + size, canvas)
    cs, ce = max(c0, 0), min(c0 + size, canvas)
    out[rs - r0:re - r0, cs - c0:ce - c0] = img[rs:re, cs:ce]
    return np.clip(out, 0, 1)


def render_many(T, **kw):
    return np.stack([render(t, **kw) for t in T])


# --------------------------------------------------------------------------- geometry of a stroke
def geometry(tokens):
    """Descriptors of one sequence: pen-down length, closure distance, bounding box width/height,
    winding (signed turning / 2 pi), total absolute turning (rad), number of pen-up moves."""
    tok = np.asarray(tokens, dtype=np.int64)
    down = tok < 8
    pos, _ = pen_path(tok)
    idx = np.nonzero(down)[0]
    if len(idx) == 0:
        return np.zeros(7)
    pts = np.concatenate([pos[idx], pos[idx[-1] + 1:idx[-1] + 2]])
    closure = np.linalg.norm(pts[-1] - pts[0])
    w = pts[:, 1].max() - pts[:, 1].min()
    h = pts[:, 0].max() - pts[:, 0].min()
    ang = ANGLES[tok[idx] % 8]
    dang = np.angle(np.exp(1j * np.diff(ang)))
    return np.array([len(idx), closure, w, h, dang.sum() / (2 * np.pi), np.abs(dang).sum(), (~down).sum()], dtype=float)


GEOMETRY_NAMES = ["pen-down length", "closure distance", "width", "height", "winding", "total turning", "pen-up moves"]


def geometry_many(T):
    return np.stack([geometry(t) for t in T])
