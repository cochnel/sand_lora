# -*- coding: utf-8 -*-
"""
Simulation couplée vent/sable sans réseau de neurones ni module solver.
Inspiré de couplage_sans_NN.py, utilise uniquement les modules disponibles
dans windblown_sand-main (fluids/ + physics_functions/).
"""

import argparse
import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--nb_xp",         type=int,   required=True)
parser.add_argument("--res_x",         type=int,   default=128, choices=[64, 128, 256])
parser.add_argument("--steps",         type=int,   default=500)
parser.add_argument("--inflow_speed",  type=float, default=1.0)
parser.add_argument("--speed_limit",   type=float, default=0.8)
parser.add_argument("--dt_salt_base",  type=float, default=1.0)
parser.add_argument("--save_every",    type=int,   default=50)
args = parser.parse_args()

# -- Chemin vers windblown_sand-main ------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_WIND_ROOT = os.path.join(_HERE, "windblown_sand-main")
if _WIND_ROOT not in sys.path:
    sys.path.insert(0, _WIND_ROOT)

from fluids import advection, pressure, bounds, level_grid, timestep, diffusion
from physics_functions import erosion_deposition_functions, couple_simulation, height_bounds


# -- Carte de hauteur (taille adaptée à res_x) --------------------------------
res_x = args.res_x

_x = np.arange(res_x, dtype=np.float32)
_y = np.arange(res_x, dtype=np.float32)
_x, _y = np.meshgrid(_x, _y)

s = res_x / 128  # facteur d'échelle

# # EE: tas gaussien unique (original)
# center_x    = int(40 * s)
# center_y    = res_x // 2
# peak_height = 20.0 * s
# sigma       = 12.0 * s
# height_map  = (peak_height * np.exp(
#     -((_x - center_x) ** 2 + (_y - center_y) ** 2) / (2 * sigma ** 2)
# )).astype(np.float32)

# EE: plusieurs petits tas près du bord x- (entrée du vent), sigmas variés
_mounds = [
    # (cx,   cy,    peak, sigma)
    ( 8*s,  20*s,   7*s,  5*s),
    (12*s,  50*s,   9*s,  8*s),
    ( 6*s,  78*s,   6*s,  4*s),
    (14*s, 105*s,   8*s,  6*s),
    ( 9*s,  35*s,   5*s,  3*s),
]
height_map = np.zeros((_x.shape[0], _x.shape[1]), dtype=np.float32)
for cx, cy, peak, sigma in _mounds:
    height_map += peak * np.exp(
        -((_x - cx) ** 2 + (_y - cy) ** 2) / (2 * sigma ** 2)
    )



# -- Paramètres de simulation --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

res_z              = 40
inflow_speed       = args.inflow_speed
inflow_sand_height = 0.0
speed_limit        = args.speed_limit
domain_size        = 10
dt_salt_base       = args.dt_salt_base
max_dt             = 1.0

total_steps         = args.steps
start_erosion_frame = 30
save_every          = args.save_every


# -- Hauteurs initiales --------------------------------------------------------
sand_heights_np   = height_map.copy()
ground_heights_np = np.zeros_like(sand_heights_np)

sand_heights   = torch.tensor(sand_heights_np,   dtype=dtype, device=device)
ground_heights = torch.tensor(ground_heights_np, dtype=dtype, device=device)

# Facteurs physiques (mêmes calculs que simu.py)
v_xp     = 5.0
nu       = 15e-6 * res_x / (v_xp * domain_size)
dt_salt  = dt_salt_base * (res_x / domain_size) ** 2
m        = domain_size / res_x

print(f"nu      = {nu:.6f}")
print(f"dt_salt = {dt_salt:.4f}")
print(f"m       = {m:.4f}")
print(f"device  = {device}")


# -- Conditions aux limites ----------------------------------------------------
# Vent entrant en x-, y-, y+ et z+ (en haut), vitesse inflow_speed selon x
inflow_vel = torch.tensor(
    [inflow_speed, 0.0, 0.0], dtype=dtype, device=device
)[:, None, None]

boundaries_conditions = bounds.check_bounds(
    *bounds.set_dirichlet_bounds({}, {}, ["x-", "y-", "y+", "z+"], inflow_vel)
)
hm_boundaries_conditions = height_bounds.create_from_fluid_bounds(
    boundaries_conditions[0],
    {"x-": inflow_sand_height, "y-": inflow_sand_height, "y+": inflow_sand_height},
)


# -- Champ de vitesse initial et SDF ------------------------------------------
# SDF calculé avant padding de ground_heights (les deux tableaux sont (H, W))
current_sdf = couple_simulation.update_sdf(ground_heights, sand_heights, res_z)

# Padding de ground_heights (opération unique, avant la boucle)
ground_heights = height_bounds.pad_bounds_ground(
    ground_heights, hm_boundaries_conditions[0]
)

# Vitesse fluide initiale : zéros  (3, res_z+1, H+1, W+1)
vel = torch.zeros(
    (3, res_z + 1, res_x + 1, res_x + 1), dtype=dtype, device=device
)

# Gestionnaire de temps
time_handler = timestep.Time(max_dt)


# -- Dossier de sortie ---------------------------------------------------------
out_dir = os.path.join(_HERE, "results", f"sand_height_{args.nb_xp}")
os.makedirs(out_dir, exist_ok=True)


# -- Boucle de simulation principale ------------------------------------------
for step in range(total_steps):

    # --- Fractions de fluide (level-set) -------------------------------------
    fractions = level_grid.get_fractions(current_sdf[None])
    fractions = torch.where(fractions > 1e-7, fractions, torch.zeros_like(fractions))

    # --- Advection -----------------------------------------------------------
    vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)
    vel = advection.advect_mac_semi_lagrange(
        vel, vel, time_handler.set_dt(vel)
    )[..., 1:-1, 1:-1, 1:-1]
    vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))
    vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)

    # --- Diffusion (viscosité) -----------------------------------------------
    fractions_padded = 1.0 + 0.0 * F.pad(fractions, (1, 1, 1, 1, 1, 1), value=1.0)
    vel = diffusion.diffuse(
        vel, fractions_padded, dx=1, dt=time_handler.dt, nu=nu
    )[..., 1:-1, 1:-1, 1:-1]
    vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, 1:-1, 1:-1]
    vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))

    # --- Projection (pression) -----------------------------------------------
    pressure_bcs = bounds.pressure_boundaries(fractions.clone(), *boundaries_conditions)
    vel, _p = pressure.solve_pressure(*pressure_bcs, vel, cg_accu=1e-3)
    vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))
    vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, 1:-1, 1:-1]

    # --- Érosion / dépôt / avalanche -----------------------------------------
    if step >= start_erosion_frame:
        sand_heights = height_bounds.pad_bounds_sand(
            sand_heights, *hm_boundaries_conditions
        )

        vel3D = bounds.pad_boundary_conditions_mac(
            vel, *boundaries_conditions
        )[..., 1:-1, :, :]

        wind2D = bounds.pad_boundary_conditions_mac(
            erosion_deposition_functions.project_wind_on_ground(
                vel3D, ground_heights + sand_heights
            ),
            *boundaries_conditions,
        )

        sand_heights, _ = erosion_deposition_functions.erosion_deposition(
            sand_heights, wind2D, speed_limit, dt_salt, v_xp, m
        )
        sand_heights = height_bounds.pad_bounds_sand(
            sand_heights, *hm_boundaries_conditions
        )
        sand_heights = erosion_deposition_functions.avalanche_nicolas(
            ground_heights, sand_heights, nb_iterations=15, dt=0.5, angle=35
        )

        # Mise à jour du level-set (avec padding, puis on retire les bords)
        current_sdf = couple_simulation.update_sdf(
            ground_heights, sand_heights, res_z
        )[..., :, 1:-1, 1:-1]

        # Retour aux dimensions non-paddées pour la prochaine itération
        sand_heights = sand_heights[..., 1:-1, 1:-1]

        if step % 50 == 0:
            print(f"  Step {step:4d} | Sand mass: {sand_heights.sum().item():.1f}")

    else:
        # Avant le début de l'érosion : on calcule quand même wind2D
        # pour garder current_sdf cohérent si besoin
        sand_padded = height_bounds.pad_bounds_sand(
            sand_heights, *hm_boundaries_conditions
        )
        vel3D = bounds.pad_boundary_conditions_mac(
            vel, *boundaries_conditions
        )[..., 1:-1, :, :]
        wind2D = bounds.pad_boundary_conditions_mac(
            erosion_deposition_functions.project_wind_on_ground(
                vel3D, ground_heights + sand_padded
            ),
            *boundaries_conditions,
        )
        if step % 10 == 0:
            print(f"  Step {step:4d} | warming up fluid...")

    # --- Sauvegarde et visualisation -----------------------------------------
    if step % save_every == 0:
        sh_np = sand_heights.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(sh_np, cmap="viridis", origin="lower")
        axes[0].set_title(f"Sand height — step {step}")
        plt.colorbar(im0, ax=axes[0], label="Height")

        # Norme du vent projeté au sol (couche z=0)
        wind_norm = (vel[0, 0].detach().cpu() ** 2 + vel[1, 0].detach().cpu() ** 2).sqrt()
        im1 = axes[1].imshow(wind_norm.numpy(), cmap="plasma", origin="lower")
        axes[1].set_title(f"Wind norm (ground level) — step {step}")
        plt.colorbar(im1, ax=axes[1], label="|u|")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step_{step:05d}.png"), dpi=80)
        plt.close(fig)

        np.save(os.path.join(out_dir, f"sand_{step:05d}.npy"), sh_np)

        # --- Tranche verticale XZ au milieu (streamlines) ---
        y_mid = res_x // 2
        vel_np = vel.detach().cpu().numpy()  # (3, res_z+1, res_x+1, res_x+1)
        # Interpolation vers les centres de cellule
        vx_xz = 0.5 * (vel_np[0, :res_z, y_mid, :-1] + vel_np[0, :res_z, y_mid, 1:])   # (res_z, res_x)
        vz_xz = 0.5 * (vel_np[2, :-1, y_mid, :res_x] + vel_np[2, 1:, y_mid, :res_x])   # (res_z, res_x)

        x_arr = np.arange(res_x, dtype=np.float32)
        z_arr = np.arange(res_z, dtype=np.float32)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        speed_xz = np.sqrt(vx_xz ** 2 + vz_xz ** 2) + 1e-8
        strm = ax2.streamplot(x_arr, z_arr, vx_xz, vz_xz,
                              color=speed_xz, cmap="plasma",
                              linewidth=1.0, density=1.5)
        plt.colorbar(strm.lines, ax=ax2, label="|u|")

        sand_profile   = sh_np[y_mid, :]
        ground_profile = ground_heights_np[y_mid, :]
        ax2.fill_between(x_arr, ground_profile, ground_profile + sand_profile,
                         color="sandybrown", alpha=0.8, label="sable")
        ax2.fill_between(x_arr, 0, ground_profile,
                         color="sienna", alpha=0.9, label="sol")

        ax2.set_xlim(0, res_x - 1)
        ax2.set_ylim(0, res_z)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title(f"Tranche XZ (y={y_mid}) — streamlines — step {step}")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"xz_stream_{step:05d}.png"), dpi=80)
        plt.close(fig2)

    time_handler.step()


print("Simulation terminée.")
print(f"Résultats sauvegardés dans : {out_dir}")
