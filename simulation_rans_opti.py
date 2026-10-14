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

--grid regular  : grille cartésienne fixe + SDF/fractions (implémentation d'origine)
--grid deformed : grille qui épouse le terrain (thèse Sec. 5.2.3, sigma_grid.py),
                  dérivées totales Eq. 5.13, plus de cellules dans le sable.
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
# --- choix de la grille du vent ---
parser.add_argument("--grid",           type=str,   default="regular", choices=["regular", "deformed"],
                    help="regular: grille fixe + SDF (d'origine) | deformed: grille qui épouse le terrain")
parser.add_argument("--top",            type=float, default=None,
                    help="[deformed] Altitude du plafond de la grille (défaut: res_z)")
parser.add_argument("--wind_height",    type=float, default=1.0,
                    help="[deformed] Hauteur au-dessus du sable où le vent est lu pour l'érosion")
parser.add_argument("--volume_weight",  action="store_true",
                    help="[deformed] Pondère chaque terme par le volume de sa cellule (absent de la thèse)")
# --- choix de la loi de transport du sable ---
parser.add_argument("--transport",      type=str,   default="legacy", choices=["legacy", "paper"],
                    help="legacy: lib windblown_sand d'origine, inchangée | paper: Eq. 2, 3, 5 de l'article (transport_paper.py)")
parser.add_argument("--epsilon",        type=float, default=0.1,
                    help="[paper] Hauteur (m) au-dessus du sable où le vent est lu (Eq. 3)")
# --- heightmap initiale ---
parser.add_argument("--heightmap",      type=str,   default=None,
                    help="Nom d'une heightmap de heightmaps/<res_x>/ (gaussian, mounds, sine_dunes...). "
                         "Défaut : tas gaussien défini ci-dessous")
args = parser.parse_args()

_HERE = os.path.dirname(os.path.abspath(__file__))
_WIND_ROOT = os.path.join(_HERE, "windblown_sand-main")
if _WIND_ROOT not in sys.path:
    sys.path.insert(0, _WIND_ROOT)

from fluids import bounds, level_grid, advection as fluid_advection, timestep as fluid_timestep
from physics_functions import erosion_deposition_functions, couple_simulation, height_bounds
from sigma_grid import SigmaGrid
import transport_paper


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


def opti_vel_deformed(vel_init, grid, nu, lambda_p, alpha_adv, n_iter, lr,
                      volume_weight=False, log_every=500, save_every=0, save_fn=None):
    """
    Même chose que opti_vel, sur la grille déformée : les valeurs imposées
    (sol, plafond, entrée) sont gérées par SigmaGrid, pas besoin de masque.
    """
    vel = vel_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([vel], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()

        e = grid.energy(*grid.fields(vel), nu, lambda_p, volume_weight)
        e.backward()

        if alpha_adv > 0.0:
            with torch.no_grad():
                vel.grad = vel.grad + alpha_adv * grid.advection_grad(vel.detach(), volume_weight)

        optimizer.step()

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
center_x    = int(40 * s)
center_y    = res_x // 2
peak_height = 20.0 * s
sigma       = 12.0 * s
height_map  = (peak_height * np.exp(
    -((_x - center_x) ** 2 + (_y - center_y) ** 2) / (2 * sigma ** 2)
)).astype(np.float32)

# EE: plusieurs petits tas près du bord x- (entrée du vent), sigmas variés
# _mounds = [
#     # (cx,   cy,    peak, sigma)
#     ( 8*s,  20*s,   7*s,  5*s),
#     (12*s,  50*s,   9*s,  8*s),
#     ( 6*s,  78*s,   6*s,  4*s),
#     (14*s, 105*s,   8*s,  6*s),
#     ( 9*s,  35*s,   5*s,  3*s),
# ]
# height_map = np.zeros((_x.shape[0], _x.shape[1]), dtype=np.float32)
# for cx, cy, peak, sigma in _mounds:
#     height_map += peak * np.exp(
#         -((_x - cx) ** 2 + (_y - cy) ** 2) / (2 * sigma ** 2)
#     )

# --heightmap <nom> : remplace la carte ci-dessus par heightmaps/<res_x>/<nom>.npy
if args.heightmap is not None:
    _hm_path = os.path.join(_HERE, "heightmaps", str(res_x), f"{args.heightmap}.npy")
    if not os.path.exists(_hm_path):
        _dir = os.path.dirname(_hm_path)
        _dispo = sorted(f[:-4] for f in os.listdir(_dir) if f.endswith(".npy")) if os.path.isdir(_dir) else []
        raise FileNotFoundError(f"{_hm_path} introuvable. Disponibles en {res_x} : {_dispo} "
                                f"(générer avec : python heightmaps/make_heightmaps.py --res {res_x})")
    height_map = np.load(_hm_path).astype(np.float32)
    if height_map.shape != (res_x, res_x):
        raise ValueError(f"{_hm_path} : forme {height_map.shape}, attendu ({res_x}, {res_x})")
    print(f"heightmap = {_hm_path}")


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

deformed = args.grid == "deformed"
top      = args.top if args.top is not None else float(res_z)
paper    = args.transport == "paper"
eps_cells = args.epsilon / m          # [paper] epsilon en cellules

print(f"grid      = {args.grid}"
      + (f"  top={top}  wind_height={args.wind_height}  volume_weight={args.volume_weight}" if deformed else ""))
print(f"transport = {args.transport}"
      + (f"  epsilon={args.epsilon} m = {eps_cells:.2f} cellules" if paper else ""))
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
    if paper:
        # [paper] vent lu à f + epsilon (Eq. 3), interpolation linéaire
        wind2D_raw = transport_paper.project_wind_at_height(vel3D, ground_h + sand_h, eps_cells)
    else:
        wind2D_raw = erosion_deposition_functions.project_wind_on_ground(
            vel3D, ground_h + sand_h
        )
    return bounds.pad_boundary_conditions_mac(wind2D_raw, *boundaries_conditions)


def make_grid(ground_h, sand_h):
    """[deformed] Grille qui épouse ground_h + sand_h (non paddés)."""
    return SigmaGrid(ground_h + sand_h, res_z, top, inflow_speed)


def make_wind2d_deformed(vel, grid):
    """[deformed] Vent au-dessus du sable, même format que make_wind2d.
    legacy : à wind_height, sans w | paper : à epsilon (Eq. 3), avec w pour la norme 3D."""
    if paper:
        wind2D_raw = grid.ground_wind(vel, eps_cells, with_w=True)
    else:
        wind2D_raw = grid.ground_wind(vel, args.wind_height)
    return bounds.pad_boundary_conditions_mac(wind2D_raw, *boundaries_conditions)


if deformed:
    grid = make_grid(ground_heights, sand_heights)
current_sdf = couple_simulation.update_sdf(ground_heights, sand_heights, res_z)
ground_heights = height_bounds.pad_bounds_ground(ground_heights, hm_boundaries_conditions[0])
fractions = fractions_from_sdf(current_sdf)
vel = torch.zeros((3, res_z + 1, res_x + 1, res_x + 1), dtype=dtype, device=device)

out_dir = os.path.join(_HERE, "results", f"sand_height_{args.nb_xp}")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "params.json"), "w") as _f:
    json.dump(vars(args), _f, indent=2)


def plot_xz_deformed(vel_, grid_, title, path, sand_np=None):
    """[deformed] Tranche XZ (y milieu) : vent rééchantillonné sur z régulier + lignes de la grille."""
    y_mid = res_x // 2
    uc, _, wc, Zc = grid_.centered(vel_)
    uc = uc[:, y_mid].cpu().numpy()
    wc = wc[:, y_mid].cpu().numpy()
    Zc = Zc[:, y_mid].cpu().numpy()

    x_arr = np.arange(res_x, dtype=np.float32) + 0.5
    z_arr = np.linspace(0, top, 2 * res_z, dtype=np.float32)
    vx = np.full((z_arr.size, res_x), np.nan, dtype=np.float32)
    vz = np.full_like(vx, np.nan)
    for i in range(res_x):
        vx[:, i] = np.interp(z_arr, Zc[:, i], uc[:, i], left=np.nan, right=np.nan)
        vz[:, i] = np.interp(z_arr, Zc[:, i], wc[:, i], left=np.nan, right=np.nan)
    vx, vz = np.ma.masked_invalid(vx), np.ma.masked_invalid(vz)

    fig_, ax_ = plt.subplots(figsize=(10, 5))
    Zn = grid_.Zn[:, y_mid].cpu().numpy()
    xn = np.arange(res_x + 1, dtype=np.float32)
    for k in range(0, res_z + 1, 4):
        ax_.plot(xn, Zn[k], color="0.75", linewidth=0.4)

    speed = np.sqrt(vx ** 2 + vz ** 2) + 1e-8
    strm = ax_.streamplot(x_arr, z_arr, vx, vz, color=speed, cmap="plasma",
                          linewidth=1.0, density=1.5)
    plt.colorbar(strm.lines, ax=ax_, label="|u|")

    if sand_np is not None:
        ground_profile = ground_heights_np[y_mid, :]
        ax_.fill_between(x_arr, ground_profile, ground_profile + sand_np[y_mid, :],
                         color="sandybrown", alpha=0.8, label="sable")
        ax_.fill_between(x_arr, 0, ground_profile, color="sienna", alpha=0.9, label="sol")
        ax_.legend(loc="upper right")
    ax_.set_xlim(0, res_x - 1)
    ax_.set_ylim(0, top)
    ax_.set_xlabel("X"); ax_.set_ylabel("Z")
    ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig_)


def _save_wind_opti(vel_, iter_):
    if not args.save_every_opti:
        return
    if deformed:
        plot_xz_deformed(vel_, grid, f"Vent grille déformée — opti initiale iter {iter_}",
                         os.path.join(out_dir, f"wind_opti_{iter_:05d}.png"), sand_heights_np)
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


def reopti(vel, n_iter, save_every=0, save_fn=None):
    """Optimise le vent avec la grille choisie (--grid)."""
    if deformed:
        return opti_vel_deformed(vel, grid, nu_wind, lambda_p, alpha_adv,
                                 n_iter=n_iter, lr=lr_wind,
                                 volume_weight=args.volume_weight,
                                 log_every=args.log_every,
                                 save_every=save_every, save_fn=save_fn)
    return opti_vel(vel, boundaries_conditions, fractions,
                    nu_wind, lambda_p, alpha_adv,
                    n_iter=n_iter, lr=lr_wind,
                    log_every=args.log_every,
                    save_every=save_every, save_fn=save_fn)


print(f"\nOptimisation initiale du vent RANS ({n_iter_init} it.)...")
vel = reopti(vel, n_iter_init, save_every=args.save_every_opti, save_fn=_save_wind_opti)
print("Vent initial prêt.\n")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
for step in range(total_steps):

    if deformed:
        # la grille suit le sable ; vel (indexé par couche au-dessus du sol) est réutilisé tel quel
        grid = make_grid(ground_heights[1:-1, 1:-1], sand_heights)
    else:
        fractions = fractions_from_sdf(current_sdf)

    if step % reopti_every == 0:
        vel = reopti(vel, n_iter_reopti)

    if step >= start_erosion_frame:
        if deformed:
            wind2D = make_wind2d_deformed(vel, grid)
        sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)
        if not deformed:
            wind2D = make_wind2d(vel, boundaries_conditions, ground_heights, sand_heights)

        if step == start_erosion_frame:
            print(f"  [debug] wind2D max={wind2D.abs().max().item():.4f}  "
                  f"mean={wind2D.abs().mean().item():.4f}")

        if paper:
            sand_heights, _ = transport_paper.erosion_deposition(
                sand_heights, wind2D, dt_salt, v_xp, m, ground_heights + sand_heights
            )
        else:
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

        if deformed:
            w2d = grid.ground_wind(vel, args.wind_height).cpu()
            wind_norm = ((0.5 * (w2d[0, :-1, :-1] + w2d[0, :-1, 1:])) ** 2 +
                         (0.5 * (w2d[1, :-1, :-1] + w2d[1, 1:, :-1])) ** 2).sqrt()
            wind_title = f"Wind norm ({args.wind_height} au-dessus du sable) — step {step}"
        else:
            vel_bc = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)
            wind_norm = (vel_bc[0, 1, 1:-1, 1:-1].detach().cpu() ** 2 +
                         vel_bc[1, 1, 1:-1, 1:-1].detach().cpu() ** 2).sqrt()
            wind_title = f"Wind norm (z=1) — step {step}"
        im1 = axes[1].imshow(wind_norm.numpy(), cmap="plasma", origin="lower")
        axes[1].set_title(wind_title)
        plt.colorbar(im1, ax=axes[1], label="|u|")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"step_{step:05d}.png"), dpi=80)
        plt.close(fig)
        np.save(os.path.join(out_dir, f"sand_{step:05d}.npy"), sh_np)

        if deformed:
            plot_xz_deformed(vel, grid, f"Tranche XZ (y={res_x // 2}) — step {step}",
                             os.path.join(out_dir, f"xz_stream_{step:05d}.png"), sh_np)
            continue

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
