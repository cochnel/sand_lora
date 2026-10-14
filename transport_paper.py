# -*- coding: utf-8 -*-
"""
Transport du sable fidèle à l'article (Rosset et al. 2024, Sec. 3.1, Eq. 2, 3, 5).

Utilisé avec --transport paper. La lib windblown_sand-main n'est pas modifiée :
--transport legacy garde erosion_deposition_functions.erosion_deposition.

Différences avec la version legacy :
  - vent lu à f + eps (Eq. 3) par interpolation linéaire en z
    (legacy : saute entre f et f + 1 selon la partie décimale de f)
  - vitesse tangente u_t = u - n (u.n), en 3D
    (legacy : u_i - u_i n_i composante par composante)
  - u* = u_t * v_xp / 20 : même calibration que legacy (gardée volontairement)
  - Zingg (Eq. 2) : Q = k_s |u*|^2 u*, |u*| = norme 3D, k_s = 0.13
    (legacy : u_x * C * (|.|^2)^3 avec une norme qui ignore u_y -> ~u^7)
  - limitation Eq. 5 : Q_i = clamp(Q_i, -Q_max, Q_max), Q_max = h_up dx / (dt N_up)
    (legacy : pas de N_up, borne max seulement)
Le pas de temps effectif est celui de legacy (dt_salt * m), seule la loi change.

Conventions (identiques à legacy) :
  sand_heights, surface : paddés (H+2, W+2)
  wind2D : paddé (3, H+3, W+3) ; wind2D[c, 1:, 1:] = wind2D_raw[c] avec
    raw[0][j, i] = u_x sur la face x i de la ligne j    (H, W+1)
    raw[1][j, i] = u_y sur la face y j de la colonne i  (H+1, W)
    raw[2][j, i] = u_z au centre de la cellule (j, i)   (H, W)
"""

import torch

K_S = 0.13  # kg.s^2.m^-4, Eq. 2


# ---------------------------------------------------------------------------
# Vent à f + eps (Eq. 3), grille régulière
# ---------------------------------------------------------------------------
def _sample_linear(field, z_index):
    """field (1, D, Y, X), z_index (Y, X) indice continu en z -> (1, Y, X)."""
    D = field.shape[1]
    z = z_index.clamp(0, D - 1)
    k0 = z.floor().long().clamp(max=D - 2)
    t = (z - k0)[None, None]
    a0 = torch.take_along_dim(field, k0[None, None], 1)
    a1 = torch.take_along_dim(field, (k0 + 1)[None, None], 1)
    return ((1 - t) * a0 + t * a1)[:, 0]


def project_wind_at_height(vel, surface_heights, offset):
    """
    Même format d'entrée/sortie que erosion_deposition_functions.project_wind_on_ground,
    mais lit le vent exactement à surface + offset (en cellules), interpolation linéaire.
    u_x, u_y sont à z = k + 1/2, u_z à z = k (même convention que legacy).
    """
    s = surface_heights
    vx = _sample_linear(vel[0:1, :, :-1, 1:-1], 0.5 * (s[..., 1:] + s[..., :-1]) + offset - 0.5)
    vy = _sample_linear(vel[1:2, :, 1:-1, :-1], 0.5 * (s[..., 1:, :] + s[..., :-1, :]) + offset - 0.5)
    vz = _sample_linear(vel[2:3, :, :-1, :-1], s + offset)
    return torch.cat([vx[..., 1:, :], vy[..., 1:], vz[..., 1:, 1:]], dim=0)


# ---------------------------------------------------------------------------
# Érosion / dépôt (Eq. 2, 3, 5)
# ---------------------------------------------------------------------------
def _shear_at_faces(wind2D, surface, v_real):
    """u* (vecteur 3D) sur les faces x (H, W+1) et y (H+1, W)."""
    H, W = surface.shape[0] - 2, surface.shape[1] - 2
    w = wind2D[:, 1:, 1:]                     # raw + une ligne/colonne de fantômes à la fin
    wp = wind2D                               # avec les fantômes au début

    # --- faces x : (j, i), i = 0..W
    ux_x = w[0, :H, :W + 1]
    vy_p = wp[1, 1:H + 2, 0:W + 2]            # (H+1, W+2), colonne c <-> colonne réelle c-1
    uy_x = 0.25 * (vy_p[:-1, :-1] + vy_p[1:, :-1] + vy_p[:-1, 1:] + vy_p[1:, 1:])
    vz_p = wp[2, 1:H + 1, 0:W + 2]
    uz_x = 0.5 * (vz_p[:, :-1] + vz_p[:, 1:])

    # --- faces y : (j, i), j = 0..H
    uy_y = w[1, :H + 1, :W]
    vx_p = wp[0, 0:H + 2, 1:W + 2]            # (H+2, W+1), ligne r <-> ligne réelle r-1
    ux_y = 0.25 * (vx_p[:-1, :-1] + vx_p[1:, :-1] + vx_p[:-1, 1:] + vx_p[1:, 1:])
    vz_p2 = wp[2, 0:H + 2, 1:W + 1]
    uz_y = 0.5 * (vz_p2[:-1] + vz_p2[1:])

    # --- normales à la surface (dx = 1, hauteurs en cellules)
    f = surface
    dfy_c = 0.5 * (f[2:, :] - f[:-2, :])      # (H, W+2) d f / dy aux centres
    dfx_c = 0.5 * (f[:, 2:] - f[:, :-2])      # (H+2, W) d f / dx aux centres
    gx_x = f[1:-1, 1:] - f[1:-1, :-1]         # (H, W+1) d f / dx aux faces x
    gy_x = 0.5 * (dfy_c[:, :-1] + dfy_c[:, 1:])
    gy_y = f[1:, 1:-1] - f[:-1, 1:-1]         # (H+1, W) d f / dy aux faces y
    gx_y = 0.5 * (dfx_c[:-1] + dfx_c[1:])

    def tangent_shear(ux, uy, uz, gx, gy):
        nrm = torch.sqrt(1 + gx ** 2 + gy ** 2)
        nx, ny, nz = -gx / nrm, -gy / nrm, 1 / nrm
        un = ux * nx + uy * ny + uz * nz
        c = v_real / 20.0                     # calibration legacy : u* = u_t * v_xp / 20
        return c * (ux - un * nx), c * (uy - un * ny), c * (uz - un * nz)

    return (tangent_shear(ux_x, uy_x, uz_x, gx_x, gy_x),
            tangent_shear(ux_y, uy_y, uz_y, gx_y, gy_y))


def erosion_deposition(sand_heights, wind2D, dt_salt, v_real, m, surface):
    """
    sand_heights, surface : paddés (H+2, W+2) ; renvoie (sand_heights (H, W), div Q).
    Même pas de temps effectif que legacy : dh = - dt_salt * m * div Q.
    """
    (sx_x, sy_x, sz_x), (sx_y, sy_y, sz_y) = _shear_at_faces(wind2D, surface, v_real)

    # Eq. 2 : Q = k_s |u*|^2 u*  (composante horizontale sur chaque face)
    Qx = K_S * (sx_x ** 2 + sy_x ** 2 + sz_x ** 2) * sx_x      # (H, W+1)
    Qy = K_S * (sx_y ** 2 + sy_y ** 2 + sz_y ** 2) * sy_y      # (H+1, W)

    # Eq. 5 : Q_max = h_up dx / (dt N_up), clamp(Q, -Q_max, Q_max)
    dt_eff = dt_salt * m
    h_p = sand_heights
    N = ((Qx[:, 1:] > 0).to(Qx.dtype) + (Qx[:, :-1] < 0).to(Qx.dtype)
         + (Qy[1:, :] > 0).to(Qx.dtype) + (Qy[:-1, :] < 0).to(Qx.dtype)).clamp(min=1)
    Qmax_p = h_p / dt_eff                                       # fantômes : N = 1
    Qmax_p = Qmax_p.clone()
    Qmax_p[1:-1, 1:-1] = h_p[1:-1, 1:-1] / (dt_eff * N)

    up_x = torch.where(Qx > 0, Qmax_p[1:-1, :-1], Qmax_p[1:-1, 1:])
    up_y = torch.where(Qy > 0, Qmax_p[:-1, 1:-1], Qmax_p[1:, 1:-1])
    Qx = torch.maximum(torch.minimum(Qx, up_x), -up_x)
    Qy = torch.maximum(torch.minimum(Qy, up_y), -up_y)

    # Eq. 1 : conservation de la masse
    divQ = (Qx[:, 1:] - Qx[:, :-1]) + (Qy[1:, :] - Qy[:-1, :])
    new_h = (h_p[1:-1, 1:-1] - dt_eff * divQ).clamp(0)
    return new_h, divQ
