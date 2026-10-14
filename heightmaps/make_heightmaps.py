# -*- coding: utf-8 -*-
"""
Génère les heightmaps initiales de sable dans heightmaps/<res>/<nom>.npy
(float32, forme (res, res), indexées [y, x], vent selon +x).

    python heightmaps/make_heightmaps.py            # résolution 128
    python heightmaps/make_heightmaps.py --res 256

Chargées par simulation_rans_opti.py avec --heightmap <nom>.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gaussian(res):
    """Tas gaussien unique (celui de simulation_rans_opti.py)."""
    s = res / 128
    x, y = np.meshgrid(np.arange(res, dtype=np.float32), np.arange(res, dtype=np.float32))
    cx, cy, peak, sigma = int(40 * s), res // 2, 20.0 * s, 12.0 * s
    return (peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))).astype(np.float32)


def mounds(res):
    """Plusieurs petits tas près du bord x- (ceux de simulation_stable_fluids.py)."""
    s = res / 128
    x, y = np.meshgrid(np.arange(res, dtype=np.float32), np.arange(res, dtype=np.float32))
    h = np.zeros((res, res), dtype=np.float32)
    for cx, cy, peak, sigma in [
        ( 8*s,  20*s,   7*s,  5*s),
        (12*s,  50*s,   9*s,  8*s),
        ( 6*s,  78*s,   6*s,  4*s),
        (14*s, 105*s,   8*s,  6*s),
        ( 9*s,  35*s,   5*s,  3*s),
    ]:
        h += peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return h.astype(np.float32)


def sine_dunes(res, wavelength=32.0, amplitude=6.0, bed=1.0):
    """
    Dunes transverses : crêtes parallèles à y (perpendiculaires au vent),
    profil sinusoïdal en x posé sur un lit de sable uniforme :
        h = bed + A (1 + sin(2 pi x / lambda)) / 2
    A = 6, lambda = 32 (à res 128) : pente max = pi A / lambda ≈ 30°, sous l'angle d'avalanche (35°).
    bed = 1 : il reste du sable partout, même dans les creux.
    """
    s = res / 128
    x = np.arange(res, dtype=np.float32)
    profile = bed * s + amplitude * s * 0.5 * (1.0 + np.sin(2.0 * np.pi * x / (wavelength * s)))
    return np.tile(profile[None, :], (res, 1)).astype(np.float32)


HEIGHTMAPS = {"gaussian": gaussian, "mounds": mounds, "sine_dunes": sine_dunes}


def save_png(h, name, path):
    """Image de la heightmap avec la hauteur en colorbar (même orientation que les simulations)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(h, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Hauteur de sable (cellules)")
    ax.set_title(f"{name} — {h.shape[1]}x{h.shape[0]}, max = {h.max():.2f}")
    ax.set_xlabel("X  (vent →)")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=128)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(args.res))
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in HEIGHTMAPS.items():
        h = fn(args.res)
        np.save(os.path.join(out_dir, f"{name}.npy"), h)
        save_png(h, name, os.path.join(out_dir, f"{name}.png"))
        print(f"{name:12s} -> {out_dir}/{name}.npy (+ .png)  shape={h.shape}  max={h.max():.2f}")
