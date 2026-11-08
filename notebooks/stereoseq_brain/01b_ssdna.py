#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                      # noqa: E402

TIF = C.MOSTA / "Mouse_brain_Adult.tif"
if not TIF.exists():
    raise SystemExit(f"missing input: {TIF}\n  this is the MOSTA ssDNA image of the section")

X_DNB0 = int(round(2 * C.X0))
Y_DNB0 = int(round(2 * C.Y0))
NX, NY = C.NX, C.NY

img = tifffile.imread(TIF)
print(f"ssDNA image {img.shape} {img.dtype}", flush=True)
need = (X_DNB0 + 2 * C.W, Y_DNB0 + 2 * C.H)
assert img.shape[0] >= need[0] and img.shape[1] >= need[1], (
    f"image {img.shape} is smaller than the grid needs {need}; it is indexed [x_dnb, y_dnb] "
    f"and must not be transposed")

sub = img[X_DNB0:X_DNB0 + 2 * C.W, Y_DNB0:Y_DNB0 + 2 * C.H]
out = sub.reshape(NX, 10, NY, 10).mean((1, 3)).astype(np.float32)
assert out.shape == (NX, NY), out.shape
np.save(C.CACHE / "ssdna_5um.npy", out)
print(f"-> {C.CACHE / 'ssdna_5um.npy'}  {out.shape}  "
      f"nonzero {100 * (out > 0).mean():.1f}% of the grid", flush=True)
