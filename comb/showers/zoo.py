"""
Model construction, saving and loading from a run specification.

A spec is a flat dict, e.g.
  {"model": "comb", "basis": "pca", "n_m": 2, "n_w": 3, "n_sub": 5, "L_v": 3,
   "L_w": 2, "dephase": 0, "current": "hamming", "entangler": "cz", "seed": 0}
  {"model": "hmm", "basis": "pca", "chi": 8, "K": 16, "seed": 0}
"""
from __future__ import annotations

import json
import os

import numpy as np

from .model import CoMB, ClassicalHMM, LinearDecoded


def build(spec: dict, d: int):
    s = dict(spec)
    n_out = d                                     # pixels
    if s.get("basis", "pixel") == "pca" and s.get("pcs"):
        d = int(s["pcs"])                         # the base model generates only the first `pcs` principal components
    if s["model"] in ("comb", "qjg"):            # "qjg": run files saved before the rename
        base = CoMB(d, n_m=s["n_m"], n_w=s["n_w"], n_sub=s["n_sub"], L_v=s.get("L_v", 3), L_w=s.get("L_w", 2),
                   seed=s.get("seed", 0), dephase=bool(s.get("dephase", 0)), entangler=s.get("entangler", "cz"),
                   init_scale=s.get("init_scale", 0.6), current=s.get("current", "hamming"), wiring=s.get("wiring", "interleaved"), dense=bool(s.get("dense", False)))
    elif s["model"] == "hmm":
        base = ClassicalHMM(d, chi=s["chi"], K=s.get("K", 16), seed=s.get("seed", 0))
    elif s["model"] == "envmem":                  # general coherent memory with an environment, or its dephased twin
        from .envmem import EnvMemory
        base = EnvMemory(d, chi=s["chi"], K=s.get("K", 8), E=s.get("E"), seed=s.get("seed", 0), dephase=bool(s.get("dephase", 0)),
                         init_scale=s.get("init_scale", 1.0))
    else:
        raise ValueError(s["model"])
    return LinearDecoded(base, n_out=n_out) if s.get("basis", "pixel") == "pca" else base


def describe(spec: dict) -> str:
    s = spec
    if s["model"] == "comb":
        q = s["n_m"] + s["n_w"]
        name = (f"CoMB(chi={2 ** s['n_m']}, n_w={s['n_w']}, n_sub={s['n_sub']}, L_v={s.get('L_v', 3)})"
                f"{' dephased' if s.get('dephase') else ''} [{q} qubits]")
    elif s["model"] == "envmem":
        name = f"EnvMemory(chi={s['chi']}, K={s.get('K', 8)}, E={s.get('E')})" + (" dephased" if s.get("dephase") else "")
    else:
        name = f"HMM(chi={s['chi']}, K={s.get('K', 16)})"
    return name + ((f" PCA({s['pcs']})" if s.get("pcs") else " PCA") if s.get("basis", "pixel") == "pca" else " pixel")


def save(path: str, model, spec: dict, extra: dict | None = None):
    arrs = dict(params=model.get(), m=model.m, s=model.s)
    ro = model.base if isinstance(model, LinearDecoded) else model
    arrs.update(lo=ro.lo, hi=ro.hi)
    if isinstance(model, LinearDecoded):
        arrs.update(mu=model.mu, P=model.P, base_m=model.base.m, base_s=model.base.s)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, spec=json.dumps(spec), extra=json.dumps(extra or {}), **arrs)
    os.replace(tmp, path)


def load(path: str, d: int | None = None):
    z = np.load(path, allow_pickle=False)
    spec = json.loads(str(z["spec"]))
    if d is None:
        d = len(z["m"])
    model = build(spec, d)
    if isinstance(model, LinearDecoded):
        model.set_decoder(z["m"], z["s"], z["mu"], z["P"], z["base_m"], z["base_s"])
        model.base.lo, model.base.hi = z["lo"].copy(), z["hi"].copy()
    else:
        model.m, model.s = z["m"], z["s"]
        model.lo, model.hi = z["lo"].copy(), z["hi"].copy()
    model.set(z["params"])
    return model, spec, json.loads(str(z["extra"]))
