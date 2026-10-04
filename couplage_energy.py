# -*- coding: utf-8 -*-
"""
Simulation couplée vent/sable avec vent par minimisation d'énergie (Stokes).
Le vent est résolu en minimisant E = E_visqueux + E_incompressibilité
via Adam, comme dans le notebook couplage_sans_NN.
Le transport de sable est identique à simulation_stable_fluids.py.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--nb_xp",         type=int,   required=True)
parser.add_argument("--steps",         type=int,   default=500)
parser.add_argument("--inflow_speed",  type=float, default=1.0)
parser.add_argument("--speed_limit",   type=float, default=0.8)
parser.add_argument("--dt_salt_base",  type=float, default=1.0)
parser.add_argument("--dt_salt",       type=float, default=None,
                    help="Écrase dt_salt_base*(res_x/domain_size)² si fourni")
parser.add_argument("--nu_wind",       type=float, default=0.05)
parser.add_argument("--lambda_p",      type=float, default=50.0)
parser.add_argument("--lr_wind",       type=float, default=0.005)
parser.add_argument("--n_iter_init",   type=int,   default=3000)
parser.add_argument("--n_iter_reopti", type=int,   default=30)
parser.add_argument("--reopti_every",  type=int,   default=10)
parser.add_argument("--save_every",    type=int,   default=50)
args = parser.parse_args()

_HERE = os.path.dirname(os.path.abspath(__file__))
_WIND_ROOT = os.path.join(_HERE, "windblown_sand-main")
if _WIND_ROOT not in sys.path:
    sys.path.insert(0, _WIND_ROOT)

from fluids import bounds, level_grid
from physics_functions import erosion_deposition_functions, couple_simulation, height_bounds


# ---------------------------------------------------------------------------
# Énergie de Stokes : E = nu*|∇v|² + lambda*(div v)²
# Les BCs Dirichlet sont appliquées par masquage avant le calcul.
# ---------------------------------------------------------------------------

def _wind_energy(vel, boundaries_conditions, fractions, nu, lambda_p):
    """
    vel       : (3, D, H, W) — vitesse intérieure (sans padding)
    fractions : (3, D, H, W) — fractions de fluide issues du SDF
    BCs appliquées via pad_boundary_conditions_mac pour le terme visqueux.
    """
    # vel_bc : (3, D+2, H+2, W+2) — avec valeurs BC aux bords
    vel_bc = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)

    # --- terme visqueux : gradients sur vel_bc complet (inclut les bords BC) ---
    E_V = 0.0
    for c in range(3):
        v = vel_bc[c]
        E_V = E_V + (v[..., 1:] - v[..., :-1]).pow(2).sum()
        E_V = E_V + (v[..., 1:, :] - v[..., :-1, :]).pow(2).sum()
        E_V = E_V + (v[1:] - v[:-1]).pow(2).sum()
    E_V = nu * E_V

    # --- terme incompressibilité : (div v)² sur l'intérieur ---
    # vel_bc[..., 1:-1, 1:-1, 1:-1] == vel (le padding intérieur n'est pas modifié)
    # fractions et vel ont la même shape → formule identique à pressure.py
    div = (
        fractions[0:1, :-1, :-1, 1:] * vel[0:1, :-1, :-1, 1:]
        - fractions[0:1, :-1, :-1, :-1] * vel[0:1, :-1, :-1, :-1]
        + fractions[1:2, :-1, 1:, :-1] * vel[1:2, :-1, 1:, :-1]
        - fractions[1:2, :-1, :-1, :-1] * vel[1:2, :-1, :-1, :-1]
        + fractions[2:3, 1:, :-1, :-1] * vel[2:3, 1:, :-1, :-1]
        - fractions[2:3, :-1, :-1, :-1] * vel[2:3, :-1, :-1, :-1]
    )
    E_P = lambda_p * div.pow(2).sum()

    return E_V + E_P


def opti_vel(vel_init, boundaries_conditions, fractions,
             inflow_speed, nu, lambda_p, n_iter, lr=0.005, log_every=500):
    """
    Minimise l'énergie de Stokes par Adam.
    vel_init  : (3, D, H, W)
    fractions : (3, D, H, W) — même shape que vel_init
    Retourne vel détaché de même shape.
    """
    solid_mask = fractions <= 1e-7  # (3, D, H, W) — True = cellule solide

    vel = vel_init.detach().clone()
    vel[solid_mask] = 0.0
    vel = vel.requires_grad_(True)
    optimizer = torch.optim.Adam([vel], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()
        e = _wind_energy(vel, boundaries_conditions, fractions, nu, lambda_p)
        e.backward()
        optimizer.step()
        # Force les cellules solides à rester à zéro
        with torch.no_grad():
            vel[solid_mask] = 0.0
        if log_every and i % log_every == 0:
            print(f"    wind opt {i:5d}/{n_iter} | E={e.item():.4f}")

    return vel.detach()


# ---------------------------------------------------------------------------
# Carte de hauteur initiale
# ---------------------------------------------------------------------------
height_map_size = (128, 128)
center_x, center_y = 40, 64
peak_height = 20.0
sigma = 12.0

_x = np.arange(height_map_size[1], dtype=np.float32)
_y = np.arange(height_map_size[0], dtype=np.float32)
_x, _y = np.meshgrid(_x, _y)
height_map = (peak_height * np.exp(
    -((_x - center_x) ** 2 + (_y - center_y) ** 2) / (2 * sigma ** 2)
)).astype(np.float32)


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

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
reopti_every        = args.reopti_every
n_iter_init         = args.n_iter_init
n_iter_reopti       = args.n_iter_reopti

nu_wind  = args.nu_wind
lambda_p = args.lambda_p
lr_wind  = args.lr_wind

sand_heights_np   = height_map.copy()
ground_heights_np = np.zeros_like(sand_heights_np)

sand_heights   = torch.tensor(sand_heights_np,   dtype=dtype, device=device)
ground_heights = torch.tensor(ground_heights_np, dtype=dtype, device=device)

res_x = ground_heights.shape[0]  # 128

v_xp    = 5.0
nu_salt = 15e-6 * res_x / (v_xp * domain_size)
dt_salt = args.dt_salt if args.dt_salt is not None else dt_salt_base * (res_x / domain_size) ** 2
m       = domain_size / res_x

print(f"nu_salt = {nu_salt:.6f}")
print(f"dt_salt = {dt_salt:.4f}")
print(f"m       = {m:.4f}")
print(f"device  = {device}")


# ---------------------------------------------------------------------------
# Conditions aux limites (identiques à couplage_without_NN.py)
# ---------------------------------------------------------------------------
inflow_vel = torch.tensor([inflow_speed, 0.0, 0.0], dtype=dtype, device=device)[:, None, None]

boundaries_conditions = bounds.check_bounds(
    *bounds.set_dirichlet_bounds({}, {}, ["x-", "y-", "y+", "z+"], inflow_vel)
)
hm_boundaries_conditions = height_bounds.create_from_fluid_bounds(
    boundaries_conditions[0],
    {"x-": inflow_sand_height, "y-": inflow_sand_height, "y+": inflow_sand_height},
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fractions_from_sdf(sdf):
    fracs = level_grid.get_fractions(sdf[None])
    return torch.where(fracs > 1e-7, fracs, torch.zeros_like(fracs))


def make_wind2d(vel, boundaries_conditions, ground_h, sand_h):
    """Projette le champ 3D sur le sol, comme dans couplage_without_NN."""
    vel3D = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, :, :]
    wind2D_raw = erosion_deposition_functions.project_wind_on_ground(
        vel3D, ground_h + sand_h
    )
    return bounds.pad_boundary_conditions_mac(wind2D_raw, *boundaries_conditions)


# ---------------------------------------------------------------------------
# Initialisation du vent
# SDF calculé AVANT padding de ground_heights (les deux sont encore 128×128)
# ---------------------------------------------------------------------------
current_sdf = couple_simulation.update_sdf(ground_heights, sand_heights, res_z)

# Padding unique de ground_heights, avant la boucle
ground_heights = height_bounds.pad_bounds_ground(ground_heights, hm_boundaries_conditions[0])

fractions = fractions_from_sdf(current_sdf)

vel = torch.zeros((3, res_z + 1, res_x + 1, res_x + 1), dtype=dtype, device=device)

print(f"\nOptimisation initiale du vent ({n_iter_init} it.)…")
vel = opti_vel(vel, boundaries_conditions, fractions,
               inflow_speed, nu_wind, lambda_p,
               n_iter=n_iter_init, lr=lr_wind, log_every=500)
print("Vent initial prêt.\n")


# ---------------------------------------------------------------------------
# Dossier de sortie
# ---------------------------------------------------------------------------
out_dir = os.path.join(_HERE, "results", f"sand_height_{args.nb_xp}")
os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
for step in range(total_steps):

    # fractions calculées à partir de la SDF du pas précédent (ou initiale)
    fractions = fractions_from_sdf(current_sdf)

    # --- Re-optimisation périodique du vent ---
    if step % reopti_every == 0:
        vel = opti_vel(vel, boundaries_conditions, fractions,
                       inflow_speed, nu_wind, lambda_p,
                       n_iter=n_iter_reopti, lr=lr_wind, log_every=0)

    # --- Érosion / dépôt / avalanche ---
    if step >= start_erosion_frame:
        sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)

        wind2D = make_wind2d(vel, boundaries_conditions, ground_heights, sand_heights)

        if step == start_erosion_frame:
            print(f"  [debug] wind2D max={wind2D.abs().max().item():.4f}  mean={wind2D.abs().mean().item():.4f}")
            print(f"  [debug] dt_salt={dt_salt:.2f}  speed_limit={speed_limit}  v_xp={v_xp}  m={m:.4f}")
            sh_before = sand_heights.sum().item()

        sand_heights, _ = erosion_deposition_functions.erosion_deposition(
            sand_heights, wind2D, speed_limit, dt_salt, v_xp, m
        )

        if step == start_erosion_frame:
            print(f"  [debug] sand mass before={sh_before:.1f}  after={sand_heights.sum().item():.1f}")

        sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)
        sand_heights = erosion_deposition_functions.avalanche_nicolas(
            ground_heights, sand_heights, nb_iterations=15, dt=0.5, angle=35
        )

        # Mise à jour SDF avec les deux hauteurs paddées, puis on enlève le bord
        current_sdf = couple_simulation.update_sdf(
            ground_heights, sand_heights, res_z
        )[..., :, 1:-1, 1:-1]

        sand_heights = sand_heights[..., 1:-1, 1:-1]

        if step % 50 == 0:
            print(f"  Step {step:4d} | Sand mass: {sand_heights.sum().item():.1f}")
    else:
        if step % 10 == 0:
            print(f"  Step {step:4d} | warming up wind…")

    # --- Sauvegarde ---
    if step % save_every == 0:
        sh_np = sand_heights.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(sh_np, cmap="viridis", origin="lower")
        axes[0].set_title(f"Sand height — step {step}")
        plt.colorbar(im0, ax=axes[0], label="Height")

        vel_bc = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)
        wind_norm = (vel_bc[0, 1, 1:-1, 1:-1].detach().cpu() ** 2 +
                     vel_bc[1, 1, 1:-1, 1:-1].detach().cpu() ** 2).sqrt()
        im1 = axes[1].imshow(wind_norm.numpy(), cmap="plasma", origin="lower")
        axes[1].set_title(f"Wind norm (z=1) — step {step}")
        plt.colorbar(im1, ax=axes[1], label="|u|")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step_{step:05d}.png"), dpi=80)
        plt.close(fig)
        np.save(os.path.join(out_dir, f"sand_{step:05d}.npy"), sh_np)


print("Simulation terminée.")
print(f"Résultats dans : {out_dir}")
