# -*- coding: utf-8 -*-
"""
Simulation couplée vent/sable — énergie RANS variationnelle.

Ref : Section 5 du papier (Eq. 5.4–5.9)
  E = E_viscosity + E_pressure + E_advection

  E_visc  = (nu/2) * integral |grad u|^2   (Eq. 5.7)
  E_press = alpha_p * integral (div u)^2    (Eq. 5.8)
  E_adv   : pas de forme fermée, mais grad_u(E_adv) = u . grad(u)  (Eq. 5.9)

L'astuce : on calcule E_visc + E_press via autograd, puis on injecte
manuellement le gradient d'advection u.grad(u) dans vel.grad avant le
pas Adam. Adam utilise alors le gradient total (visc + press + adv).

u.grad(u) est approximé via l'advection semi-lagrangienne :
  u . grad(u)  ≈  (vel - advect(vel, vel, dt)) / dt

Viscosité : nu_eff = nu_mol + nu_turb, papier pose nu_turb = 0.5.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--nb_xp",          type=int,   required=True)
parser.add_argument("--res_x",          type=int,   default=128, choices=[64, 128, 256])
parser.add_argument("--steps",          type=int,   default=500)
parser.add_argument("--inflow_speed",   type=float, default=1.0)
parser.add_argument("--speed_limit",    type=float, default=0.8)
parser.add_argument("--dt_salt_base",   type=float, default=1.0)
parser.add_argument("--save_every",     type=int,   default=50)
# --- paramètres vent ---
parser.add_argument("--nu_wind",        type=float, default=0.5,
                    help="Viscosité effective (nu_mol + nu_turb). Papier: nu_turb=0.5")
parser.add_argument("--lambda_p",       type=float, default=10.0,
                    help="Pénalité divergence (alpha_p). Papier: 10")
parser.add_argument("--alpha_adv",      type=float, default=1.0,
                    help="Poids du gradient d'advection injecté dans Adam")
parser.add_argument("--lr_wind",        type=float, default=0.005)
parser.add_argument("--n_iter_init",    type=int,   default=3000)
parser.add_argument("--n_iter_reopti",  type=int,   default=30)
parser.add_argument("--reopti_every",   type=int,   default=10)
parser.add_argument("--log_every",      type=int,   default=500)
parser.add_argument("--save_every_opti",type=int,   default=500)
args = parser.parse_args()

_HERE = os.path.dirname(os.path.abspath(__file__))
_WIND_ROOT = os.path.join(_HERE, "windblown_sand-main")
if _WIND_ROOT not in sys.path:
    sys.path.insert(0, _WIND_ROOT)

from fluids import bounds, level_grid, advection as fluid_advection, timestep as fluid_timestep
from physics_functions import erosion_deposition_functions, couple_simulation, height_bounds


# ---------------------------------------------------------------------------
# Énergie variationnelle RANS (viscosité + pression, sans advection)
# L'advection est ajoutée manuellement comme gradient (Eq. 5.9).
# ---------------------------------------------------------------------------

def _stokes_energy(vel, boundaries_conditions, fractions, nu, lambda_p):
    """E_visc + E_press — seules les termes dont on peut calculer une énergie fermée."""
    vel_bc = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)

    # E_viscosity = nu/2 * integral |grad u|^2  (Eq. 5.7)
    E_V = 0.0
    for c in range(3):
        v = vel_bc[c]
        E_V = E_V + (v[..., 1:] - v[..., :-1]).pow(2).sum()   # d/dx
        E_V = E_V + (v[..., 1:, :] - v[..., :-1, :]).pow(2).sum()  # d/dy
        E_V = E_V + (v[1:] - v[:-1]).pow(2).sum()              # d/dz
    E_V = (nu / 2.0) * E_V

    # E_pressure = alpha_p * integral (div u)^2  (Eq. 5.8)
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


def _advection_gradient(vel, boundaries_conditions, fractions):
    """
    Gradient de l'énergie d'advection (Eq. 5.9) :
      grad_u(E_adv) = u . grad(u)

    Approximation via semi-lagrangienne :
      u . grad(u)  ≈  (vel - advect(vel, vel, dt)) / dt

    Retourne un tenseur de même forme que vel, sans graphe de gradient.
    """
    time_h = fluid_timestep.Time(max_dt=1.0)
    v_padded = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)
    dt = time_h.set_dt(v_padded)
    v_adv = fluid_advection.advect_mac_semi_lagrange(
        v_padded, v_padded, dt
    )[..., 1:-1, 1:-1, 1:-1]
    v_adv = torch.where(fractions > 1e-7, v_adv, torch.zeros_like(v_adv))
    return (vel - v_adv) / dt


def opti_vel(vel_init, boundaries_conditions, fractions,
             nu, lambda_p, alpha_adv, n_iter, lr,
             log_every=500, save_every=0, save_fn=None):
    """
    Minimise E = E_visc + E_press + E_adv via Adam.
    E_visc et E_press sont calculées via autograd.
    Le gradient de E_adv est injecté manuellement dans vel.grad (Eq. 5.9).
    """
    solid_mask = fractions <= 1e-7
    vel = vel_init.detach().clone()
    vel[solid_mask] = 0.0
    vel = vel.requires_grad_(True)
    optimizer = torch.optim.Adam([vel], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()

        # Gradient autograd : E_visc + E_press
        e = _stokes_energy(vel, boundaries_conditions, fractions, nu, lambda_p)
        e.backward()

        # Injection manuelle du gradient d'advection (Eq. 5.9)
        if alpha_adv > 0.0:
            with torch.no_grad():
                adv_grad = _advection_gradient(vel.detach(), boundaries_conditions, fractions)
                vel.grad = vel.grad + alpha_adv * adv_grad

        optimizer.step()
        with torch.no_grad():
            vel[solid_mask] = 0.0

        if log_every and i % log_every == 0:
            print(f"    wind opt {i:5d}/{n_iter} | E={e.item():.4f}")
        if save_every and save_fn and i % save_every == 0:
            save_fn(vel.detach(), i)

    return vel.detach()


# ---------------------------------------------------------------------------
# Carte de hauteur initiale
# ---------------------------------------------------------------------------
res_x = args.res_x
_x = np.arange(res_x, dtype=np.float32)
_y = np.arange(res_x, dtype=np.float32)
_x, _y = np.meshgrid(_x, _y)
s = res_x / 128

# Tas gaussien unique
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


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32

res_z              = 40
inflow_speed       = args.inflow_speed
inflow_sand_height = 0.0
speed_limit        = args.speed_limit
domain_size        = 10
dt_salt_base       = args.dt_salt_base

total_steps         = args.steps
start_erosion_frame = 30
save_every          = args.save_every
reopti_every        = args.reopti_every
n_iter_init         = args.n_iter_init
n_iter_reopti       = args.n_iter_reopti

nu_wind   = args.nu_wind
lambda_p  = args.lambda_p
alpha_adv = args.alpha_adv
lr_wind   = args.lr_wind

sand_heights_np   = height_map.copy()
ground_heights_np = np.zeros_like(sand_heights_np)
sand_heights   = torch.tensor(sand_heights_np,   dtype=dtype, device=device)
ground_heights = torch.tensor(ground_heights_np, dtype=dtype, device=device)

v_xp    = 5.0
dt_salt = dt_salt_base * (res_x / domain_size) ** 2
m       = domain_size / res_x

print(f"nu_wind   = {nu_wind}")
print(f"lambda_p  = {lambda_p}")
print(f"alpha_adv = {alpha_adv}")
print(f"dt_salt   = {dt_salt:.4f}  m={m:.4f}  device={device}")


# ---------------------------------------------------------------------------
# Conditions aux limites
# NOTE : le papier utilise des CL libres (Neumann) sur y- et y+.
# On garde Dirichlet sur y-, y+ pour compatibilité avec le module bounds actuel.
# À revisiter si le module supporte d'autres types de CL.
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
# Initialisation
# ---------------------------------------------------------------------------
def fractions_from_sdf(sdf):
    fracs = level_grid.get_fractions(sdf[None])
    return torch.where(fracs > 1e-7, fracs, torch.zeros_like(fracs))


def make_wind2d(vel, boundaries_conditions, ground_h, sand_h):
    """Projette le vent 3D sur la surface du sable (+1 pour éviter face fraction=0)."""
    vel3D = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, :, :]
    wind2D_raw = erosion_deposition_functions.project_wind_on_ground(
        vel3D, ground_h + sand_h
    )
    return bounds.pad_boundary_conditions_mac(wind2D_raw, *boundaries_conditions)


current_sdf = couple_simulation.update_sdf(ground_heights, sand_heights, res_z)
ground_heights = height_bounds.pad_bounds_ground(ground_heights, hm_boundaries_conditions[0])
fractions = fractions_from_sdf(current_sdf)
vel = torch.zeros((3, res_z + 1, res_x + 1, res_x + 1), dtype=dtype, device=device)

out_dir = os.path.join(_HERE, "results", f"sand_height_{args.nb_xp}")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "params.json"), "w") as _f:
    json.dump(vars(args), _f, indent=2)


def _save_wind_opti(vel_, iter_):
    if not args.save_every_opti:
        return
    vel_np_ = vel_.cpu().numpy()
    vx_ = 0.5 * (vel_np_[0, :res_z, res_x//2, :-1] + vel_np_[0, :res_z, res_x//2, 1:])
    vz_ = 0.5 * (vel_np_[2, :-1, res_x//2, :res_x] + vel_np_[2, 1:, res_x//2, :res_x])
    fig_, ax_ = plt.subplots(figsize=(10, 5))
    spd_ = np.sqrt(vx_**2 + vz_**2) + 1e-8
    ax_.streamplot(np.arange(res_x, dtype=np.float32), np.arange(res_z, dtype=np.float32),
                   vx_, vz_, color=spd_, cmap="plasma", linewidth=1.0, density=1.5)
    ax_.set_title(f"Vent RANS — opti initiale iter {iter_}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"wind_opti_{iter_:05d}.png"), dpi=80)
    plt.close(fig_)


print(f"\nOptimisation initiale du vent RANS ({n_iter_init} it.)...")
vel = opti_vel(vel, boundaries_conditions, fractions,
               nu_wind, lambda_p, alpha_adv,
               n_iter=n_iter_init, lr=lr_wind,
               log_every=args.log_every,
               save_every=args.save_every_opti, save_fn=_save_wind_opti)
print("Vent initial prêt.\n")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
for step in range(total_steps):

    fractions = fractions_from_sdf(current_sdf)

    if step % reopti_every == 0:
        vel = opti_vel(vel, boundaries_conditions, fractions,
                       nu_wind, lambda_p, alpha_adv,
                       n_iter=n_iter_reopti, lr=lr_wind,
                       log_every=args.log_every)

    if step >= start_erosion_frame:
        sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)
        wind2D = make_wind2d(vel, boundaries_conditions, ground_heights, sand_heights)

        if step == start_erosion_frame:
            print(f"  [debug] wind2D max={wind2D.abs().max().item():.4f}  "
                  f"mean={wind2D.abs().mean().item():.4f}")

        sand_heights, _ = erosion_deposition_functions.erosion_deposition(
            sand_heights, wind2D, speed_limit, dt_salt, v_xp, m
        )
        sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)
        sand_heights = erosion_deposition_functions.avalanche_nicolas(
            ground_heights, sand_heights, nb_iterations=15, dt=0.5, angle=35
        )
        current_sdf = couple_simulation.update_sdf(
            ground_heights, sand_heights, res_z
        )[..., :, 1:-1, 1:-1]
        sand_heights = sand_heights[..., 1:-1, 1:-1]

        if step % 50 == 0:
            print(f"  Step {step:4d} | Sand mass: {sand_heights.sum().item():.1f}")
    else:
        if step % 10 == 0:
            print(f"  Step {step:4d} | warming up wind...")

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

        y_mid = res_x // 2
        vel_np = vel.detach().cpu().numpy()
        vx_xz = 0.5 * (vel_np[0, :res_z, y_mid, :-1] + vel_np[0, :res_z, y_mid, 1:])
        vz_xz = 0.5 * (vel_np[2, :-1, y_mid, :res_x] + vel_np[2, 1:, y_mid, :res_x])
        x_arr = np.arange(res_x, dtype=np.float32)
        z_arr = np.arange(res_z, dtype=np.float32)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        speed_xz = np.sqrt(vx_xz ** 2 + vz_xz ** 2) + 1e-8
        strm = ax2.streamplot(x_arr, z_arr, vx_xz, vz_xz,
                              color=speed_xz, cmap="plasma", linewidth=1.0, density=1.5)
        plt.colorbar(strm.lines, ax=ax2, label="|u|")

        sand_profile   = sh_np[y_mid, :]
        ground_profile = ground_heights_np[y_mid, :]
        ax2.fill_between(x_arr, ground_profile, ground_profile + sand_profile,
                         color="sandybrown", alpha=0.8, label="sable")
        ax2.fill_between(x_arr, 0, ground_profile,
                         color="sienna", alpha=0.9, label="sol")
        ax2.set_xlim(0, res_x - 1)
        ax2.set_ylim(0, res_z)
        ax2.set_xlabel("X"); ax2.set_ylabel("Z")
        ax2.set_title(f"Tranche XZ (y={y_mid}) — step {step}")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"xz_stream_{step:05d}.png"), dpi=80)
        plt.close(fig2)


print("Simulation terminée.")
print(f"Résultats dans : {out_dir}")
