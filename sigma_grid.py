# -*- coding: utf-8 -*-
"""
Grille déformée (terrain-following) pour le vent — thèse Sec. 5.2.3.

La grille est régulière en (x, y) et chaque colonne est étirée entre la
surface du terrain h(x, y) et un plafond horizontal fixe `top` :

    z(k, y, x) = h(x, y) + (k / nz) * (top - h(x, y)),    k = 0..nz

Les hauteurs z sont stockées aux coins des cellules (Fig. 5.5). Les vitesses
(composantes cartésiennes u, v, w) sont sur une grille décalée :
    u : faces normales à x   -> (nz,   ny,   nx+1)
    v : faces normales à y   -> (nz,   ny+1, nx  )
    w : faces sigma (k)      -> (nz+1, ny,   nx  )
On garde le rangement de windblown_sand : vel = (3, nz+1, ny+1, nx+1),
les cases en trop sont ignorées.

Dérivées totales le long de la grille (Eq. 5.13), avec dx = dy = 1 :
    du/dx = Δu/Δx - du/dz * Δz/Δx
    du/dy = Δu/Δy - du/dz * Δz/Δy
    du/dz = Δu/Δz
où du/dz est interpolé au point de la différence horizontale (Fig. 5.5).

Advection (Eq. 5.9) : même schéma semi-lagrangien que la version régulière,
    u.grad(u) ≈ (vel - advect(vel)) / dt, trajectoires remontées en (x, y, z).

Conditions aux limites (thèse p. 71) :
    x-  : Dirichlet (vent entrant U, 0, 0)
    z+  : Dirichlet (U, 0, 0)
    sol : vitesse nulle (adhérence)
    x+, y-, y+ : libres (Neumann)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first(a, dim):
    return a.narrow(dim, 0, 1)


def _last(a, dim):
    return a.narrow(dim, a.shape[dim] - 1, 1)


def _avg(a, dim):
    """Moyenne de deux voisins le long de dim (n -> n-1)."""
    n = a.shape[dim]
    return 0.5 * (a.narrow(dim, 0, n - 1) + a.narrow(dim, 1, n - 1))


def _mid_to_points(m, dim):
    """Valeurs aux milieux (n-1) -> valeurs aux points (n), bords répliqués."""
    m = torch.cat([_first(m, dim), m, _last(m, dim)], dim)
    return _avg(m, dim)


def _ddz_points(A, Z):
    """du/dz aux points de A, le long des colonnes (dim 0)."""
    return _mid_to_points(torch.diff(A, dim=0) / torch.diff(Z, dim=0), 0)


def _dh_mid(A, Z, Az, dim):
    """
    Dérivée horizontale (dim = 2 pour x, 1 pour y) aux milieux, Eq. 5.13 :
        du/dx = Δu - du/dz * Δz      (Δx = 1)
    """
    return torch.diff(A, dim=dim) - _avg(Az, dim) * torch.diff(Z, dim=dim)


def _interp3(A, k, j, i):
    """Interpolation trilinéaire de A (K, J, I) en indices continus, bornés au tableau."""
    def split(c, n):
        c = c.clamp(0, n - 1)
        c0 = c.floor().long().clamp(max=n - 2)
        return c0, c - c0
    k0, fk = split(k, A.shape[0])
    j0, fj = split(j, A.shape[1])
    i0, fi = split(i, A.shape[2])
    out = 0.0
    for dk, wk in ((0, 1 - fk), (1, fk)):
        for dj, wj in ((0, 1 - fj), (1, fj)):
            for di, wi in ((0, 1 - fi), (1, fi)):
                out = out + wk * wj * wi * A[k0 + dk, j0 + dj, i0 + di]
    return out


def _pad_Z(Z):
    """Positions des fantômes : répliquées en x, y ; extrapolées linéairement en z."""
    Z = torch.cat([_first(Z, 2), Z, _last(Z, 2)], 2)
    Z = torch.cat([_first(Z, 1), Z, _last(Z, 1)], 1)
    Z = torch.cat([2 * Z[:1] - Z[1:2], Z, 2 * Z[-1:] - Z[-2:-1]], 0)
    return Z


# ---------------------------------------------------------------------------
# Grille
# ---------------------------------------------------------------------------
class SigmaGrid:

    def __init__(self, heights, res_z, top, inflow_speed):
        """
        heights : (ny, nx) hauteur totale (sol + sable) aux centres des cellules
        res_z   : nombre de couches nz
        top     : altitude du plafond (horizontal)
        """
        if heights.max().item() >= top:
            raise ValueError(f"terrain ({heights.max().item():.2f}) au-dessus du plafond ({top})")

        self.nz = res_z
        self.ny, self.nx = heights.shape
        self.top = top
        self.U = inflow_speed

        # hauteurs aux coins (moyenne des 4 cellules voisines)
        hp = F.pad(heights[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        h_nodes = 0.25 * (hp[:-1, :-1] + hp[1:, :-1] + hp[:-1, 1:] + hp[1:, 1:])
        s = torch.linspace(0, 1, res_z + 1, device=heights.device, dtype=heights.dtype)[:, None, None]
        self.h_nodes = h_nodes                                         # (ny+1, nx+1)
        self.Zn = h_nodes + s * (top - h_nodes)                       # (nz+1, ny+1, nx+1)

        Zn = self.Zn
        Zu = 0.25 * (Zn[:-1, :-1] + Zn[1:, :-1] + Zn[:-1, 1:] + Zn[1:, 1:])               # (nz, ny, nx+1)
        Zv = 0.25 * (Zn[:-1, :, :-1] + Zn[1:, :, :-1] + Zn[:-1, :, 1:] + Zn[1:, :, 1:])   # (nz, ny+1, nx)
        Zw = 0.25 * (Zn[:, :-1, :-1] + Zn[:, 1:, :-1] + Zn[:, :-1, 1:] + Zn[:, 1:, 1:])   # (nz+1, ny, nx)
        self.Zw = Zw
        self.Zc = _avg(Zw, 0)                                          # centres (nz, ny, nx)

        self.Zu_p, self.Zv_p, self.Zw_p = _pad_Z(Zu), _pad_Z(Zv), _pad_Z(Zw)

        # épaisseur locale des couches aux points (pour la pondération par volume)
        self.Tu_p = _mid_to_points(torch.diff(self.Zu_p, dim=0), 0)
        self.Tv_p = _mid_to_points(torch.diff(self.Zv_p, dim=0), 0)
        self.Tw_p = _mid_to_points(torch.diff(self.Zw_p, dim=0), 0)

        # positions physiques (x, y, z) des points réels de chaque composante
        def xy(nyp, nxp, oy, ox):
            y = torch.arange(nyp, device=heights.device, dtype=heights.dtype)[:, None] + oy
            x = torch.arange(nxp, device=heights.device, dtype=heights.dtype)[None, :] + ox
            return x.expand(nyp, nxp), y.expand(nyp, nxp)
        nz, ny, nx = self.nz, self.ny, self.nx
        self.P = {
            "u": (*xy(ny, nx + 1, 0.5, 0.0), Zu),
            "v": (*xy(ny + 1, nx, 0.0, 0.5), Zv),
            "w": (*xy(ny, nx, 0.5, 0.5), Zw),
        }

    # -----------------------------------------------------------------------
    # Champs
    # -----------------------------------------------------------------------
    def zeros(self, device=None, dtype=None):
        return torch.zeros((3, self.nz + 1, self.ny + 1, self.nx + 1),
                           device=device or self.Zn.device, dtype=dtype or self.Zn.dtype)

    def unpack(self, vel):
        """vel (3, nz+1, ny+1, nx+1) -> u, v, w avec les valeurs imposées aux bords."""
        nz, ny, nx = self.nz, self.ny, self.nx
        u = vel[0, :nz, :ny, :]
        u = torch.cat([torch.full_like(u[..., :1], self.U), u[..., 1:]], -1)  # x- : vent entrant
        v = vel[1, :nz, :, :nx]
        w = vel[2, :, :ny, :nx]
        w = torch.cat([torch.zeros_like(w[:1]), w[1:-1], torch.zeros_like(w[:1])], 0)  # sol et plafond
        return u, v, w

    def pad(self, u, v, w):
        """Ajoute une couche de fantômes dans chaque direction (ordre x, y, z)."""
        U = self.U
        # u : x- valeur U, x+ libre | y libre | sol miroir (0 au sol), plafond miroir (U)
        u = torch.cat([torch.full_like(u[..., :1], U), u, u[..., -1:]], 2)
        u = torch.cat([u[:, :1], u, u[:, -1:]], 1)
        u = torch.cat([-u[:1], u, 2 * U - u[-1:]], 0)
        # v : x- miroir (0), x+ libre | y libre | sol et plafond miroir (0)
        v = torch.cat([-v[..., :1], v, v[..., -1:]], 2)
        v = torch.cat([v[:, :1], v, v[:, -1:]], 1)
        v = torch.cat([-v[:1], v, -v[-1:]], 0)
        # w : x- miroir (0), x+ libre | y libre | w = 0 sur les faces sol/plafond, fantômes antisymétriques
        w = torch.cat([-w[..., :1], w, w[..., -1:]], 2)
        w = torch.cat([w[:, :1], w, w[:, -1:]], 1)
        w = torch.cat([-w[1:2], w, -w[-2:-1]], 0)
        return u, v, w

    def fields(self, vel):
        return self.pad(*self.unpack(vel))

    # -----------------------------------------------------------------------
    # Énergie (Eq. 5.7, 5.8) sur grille déformée
    # -----------------------------------------------------------------------
    def _visc_component(self, A, Z, T, real_z_only, volume_weight):
        Az = _ddz_points(A, Z)

        sx = (slice(1, -1), slice(1, -1), slice(None))
        dx = _dh_mid(A[sx], Z[sx], Az[sx], 2)

        sy = (slice(1, -1), slice(None), slice(1, -1))
        dy = _dh_mid(A[sy], Z[sy], Az[sy], 1)

        sz = (slice(1, -1) if real_z_only else slice(None), slice(1, -1), slice(1, -1))
        dZ = torch.diff(Z[sz], dim=0)
        dz = torch.diff(A[sz], dim=0) / dZ

        if not volume_weight:
            return dx.pow(2).sum() + dy.pow(2).sum() + dz.pow(2).sum()
        return ((dx.pow(2) * _avg(T[sx], 2)).sum()
                + (dy.pow(2) * _avg(T[sy], 1)).sum()
                + (dz.pow(2) * dZ).sum())

    def divergence(self, u_p, v_p, w_p):
        """div u aux centres des cellules (nz, ny, nx), dérivées de l'Eq. 5.13."""
        r = (slice(1, -1),) * 3
        Azu = _ddz_points(u_p, self.Zu_p)[r]
        Azv = _ddz_points(v_p, self.Zv_p)[r]
        dudx = _dh_mid(u_p[r], self.Zu_p[r], Azu, 2)
        dvdy = _dh_mid(v_p[r], self.Zv_p[r], Azv, 1)
        dwdz = torch.diff(w_p[r], dim=0) / torch.diff(self.Zw, dim=0)
        return dudx + dvdy + dwdz

    def energy(self, u_p, v_p, w_p, nu, lambda_p, volume_weight=False):
        """E_visc + E_press (l'advection est ajoutée comme gradient)."""
        E_V = (self._visc_component(u_p, self.Zu_p, self.Tu_p, False, volume_weight)
               + self._visc_component(v_p, self.Zv_p, self.Tv_p, False, volume_weight)
               + self._visc_component(w_p, self.Zw_p, self.Tw_p, True, volume_weight))
        E_V = (nu / 2.0) * E_V

        div = self.divergence(u_p, v_p, w_p)
        if volume_weight:
            E_P = lambda_p * (div.pow(2) * torch.diff(self.Zw, dim=0)).sum()
        else:
            E_P = lambda_p * div.pow(2).sum()
        return E_V + E_P

    # -----------------------------------------------------------------------
    # Dérivées aux points (utilisées par l'auto-test de l'Eq. 5.13)
    # -----------------------------------------------------------------------
    def _grad_points(self, A, Z):
        """(dA/dx, dA/dy, dA/dz) aux points réels de A."""
        Az = _ddz_points(A, Z)
        sx = (slice(1, -1), slice(1, -1), slice(None))
        sy = (slice(1, -1), slice(None), slice(1, -1))
        r = (slice(1, -1),) * 3
        Ax = _avg(_dh_mid(A[sx], Z[sx], Az[sx], 2), 2)
        Ay = _avg(_dh_mid(A[sy], Z[sy], Az[sy], 1), 1)
        return Ax, Ay, Az[r]

    # -----------------------------------------------------------------------
    # Gradient d'advection (Eq. 5.9) : u . grad(u), semi-lagrangien
    # Même schéma que la version régulière (advect_mac_semi_lagrange) :
    #   dt = min(1, max|v|), remontée RK3,  u.grad(u) ≈ (vel - advect(vel)) / dt
    # La trajectoire est remontée en coordonnées physiques (x, y, z), puis
    # convertie en coordonnées de grille avec sigma = (z - h) / (top - h).
    # -----------------------------------------------------------------------
    # décalage (sigma, y, x) des points de chaque composante, en cellules
    _OFF = {"u": (0.5, 0.5, 0.0), "v": (0.5, 0.0, 0.5), "w": (0.0, 0.5, 0.5)}

    def _h_at(self, x, y):
        """Hauteur du terrain en (x, y), interpolée bilinéairement depuis les coins."""
        x = x.clamp(0, self.nx)
        y = y.clamp(0, self.ny)
        i0 = x.floor().long().clamp(max=self.nx - 1)
        j0 = y.floor().long().clamp(max=self.ny - 1)
        fx, fy = x - i0, y - j0
        h = self.h_nodes
        return ((1 - fy) * ((1 - fx) * h[j0, i0] + fx * h[j0, i0 + 1])
                + fy * ((1 - fx) * h[j0 + 1, i0] + fx * h[j0 + 1, i0 + 1]))

    def _sample(self, A_p, comp, x, y, z):
        """Valeur de la composante `comp` (champ paddé A_p) au point physique (x, y, z)."""
        oz, oy, ox = self._OFF[comp]
        h = self._h_at(x, y)
        k = (z - h) / (self.top - h) * self.nz
        return _interp3(A_p, k - oz + 1, y - oy + 1, x - ox + 1)

    def _vel_at(self, u_p, v_p, w_p, x, y, z):
        return (self._sample(u_p, "u", x, y, z),
                self._sample(v_p, "v", x, y, z),
                self._sample(w_p, "w", x, y, z))

    def _advection_terms(self, u_p, v_p, w_p):
        """(vel - advect(vel)) / dt aux points réels de u, v, w."""
        dt = min(1.0, max(u_p.abs().max().item(), v_p.abs().max().item(), w_p.abs().max().item()))
        r = (slice(1, -1),) * 3
        out = []
        for A_p, comp in ((u_p, "u"), (v_p, "v"), (w_p, "w")):
            x, y, z = self.P[comp]
            x, y = x.expand_as(z), y.expand_as(z)
            k1 = self._vel_at(u_p, v_p, w_p, x, y, z)
            k2 = self._vel_at(u_p, v_p, w_p, *(c - 0.5 * dt * a for c, a in zip((x, y, z), k1)))
            k3 = self._vel_at(u_p, v_p, w_p, *(c - 0.75 * dt * b for c, b in zip((x, y, z), k2)))
            back = [c - dt * (2.0 / 9.0 * a + 3.0 / 9.0 * b + 4.0 / 9.0 * d)
                    for c, a, b, d in zip((x, y, z), k1, k2, k3)]
            out.append((A_p[r] - self._sample(A_p, comp, *back)) / dt)
        return out

    def advection_grad(self, vel, volume_weight=False):
        """Retourne u.grad(u) (sans alpha) rangé comme vel, 0 aux valeurs imposées."""
        Gu, Gv, Gw = self._advection_terms(*self.fields(vel))

        r = (slice(1, -1),) * 3
        if volume_weight:
            Gu, Gv, Gw = Gu * self.Tu_p[r], Gv * self.Tv_p[r], Gw * self.Tw_p[r]

        nz, ny, nx = self.nz, self.ny, self.nx
        G = torch.zeros_like(vel)
        G[0, :nz, :ny, 1:] = Gu[..., 1:]              # u en x- imposé
        G[1, :nz, :, :nx] = Gv
        G[2, 1:nz, :ny, :nx] = Gw[1:-1]               # w au sol / plafond imposé
        return G

    # -----------------------------------------------------------------------
    # Couplage sable : vent à une hauteur fixe au-dessus de la surface
    # -----------------------------------------------------------------------
    @staticmethod
    def _sample_column(A_p, T, height, half=0.5):
        """
        A_p  : (K, Y, X) valeurs d'une composante (fantômes en z inclus)
        T    : (Y, X) épaisseur de couche
        half : le point k réel est à (k+half)T du sol (0.5 pour u, v ; 0 pour w)
        """
        p = (height / T + 1 - half).clamp(0, A_p.shape[0] - 1 - 1e-4)
        p0 = p.floor().long()
        f = p - p0
        a0 = torch.gather(A_p, 0, p0[None])[0]
        a1 = torch.gather(A_p, 0, (p0 + 1)[None])[0]
        return (1 - f) * a0 + f * a1

    def ground_wind(self, vel, height, with_w=False):
        """
        Vent à `height` au-dessus de la surface, au format wind2D_raw de
        project_wind_on_ground : (3, ny+1, nx+1), u sur faces x, v sur faces y,
        w au centre des cellules (seulement si with_w, sinon 0 comme avant).
        """
        u_p, v_p, w_p = self.fields(vel)
        us = self._sample_column(u_p[:, 1:-1, 1:-1], self.Tu_p[1, 1:-1, 1:-1], height)  # (ny, nx+1)
        vs = self._sample_column(v_p[:, 1:-1, 1:-1], self.Tv_p[1, 1:-1, 1:-1], height)  # (ny+1, nx)
        wind = torch.zeros((3, self.ny + 1, self.nx + 1), device=vel.device, dtype=vel.dtype)
        wind[0, :self.ny, :] = us
        wind[1, :, :self.nx] = vs
        if with_w:
            wind[2, :self.ny, :self.nx] = self._sample_column(
                w_p[:, 1:-1, 1:-1], self.Tw_p[1, 1:-1, 1:-1], height, half=0.0)   # (ny, nx)
        return wind

    # -----------------------------------------------------------------------
    # Visualisation : champ aux centres des cellules
    # -----------------------------------------------------------------------
    def centered(self, vel):
        u, v, w = self.unpack(vel)
        return _avg(u, 2), _avg(v, 1), _avg(w, 0), self.Zc


# ---------------------------------------------------------------------------
# Auto-test : opérateurs exacts sur des champs linéaires, grille déformée
#   python sigma_grid.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    nz, ny, nx = 10, 12, 16
    yy, xx = torch.meshgrid(torch.arange(ny) + 0.5, torch.arange(nx) + 0.5, indexing="ij")
    h = 4.0 * torch.exp(-((xx - 8) ** 2 + (yy - 6) ** 2) / 8.0)
    g = SigmaGrid(h, nz, top=10.0, inflow_speed=1.0)

    def coords(Z_p, x0, y0):
        """x, y physiques des points paddés (le pad x/y réplique les positions Z,
        on prend donc aussi x, y répliqués pour rester cohérent)."""
        K, Y, X = Z_p.shape
        x = (torch.arange(X) - 1 + x0).clamp(x0, X - 2 + x0)
        y = (torch.arange(Y) - 1 + y0).clamp(y0, Y - 2 + y0)
        return x[None, None, :].expand(K, Y, X), y[None, :, None].expand(K, Y, X)

    xu, yu = coords(g.Zu_p, 0.0, 0.5)
    xv, yv = coords(g.Zv_p, 0.5, 0.0)
    xw, yw = coords(g.Zw_p, 0.5, 0.5)

    # u = 2x + 3z, v = -y + z, w = -z  ->  div = 2 - 1 - 1 = 0
    u_p = 2 * xu + 3 * g.Zu_p
    v_p = -yv + g.Zv_p
    w_p = -g.Zw_p
    div = g.divergence(u_p, v_p, w_p)[:, 1:-1, 1:-1]
    print(f"div (attendu 0)          : max |div| = {div.abs().max().item():.2e}")

    ux, uy, uz = g._grad_points(u_p, g.Zu_p)
    print(f"du/dx (attendu 2)        : err max = {(ux[:, 1:-1, 1:-1] - 2).abs().max().item():.2e}")
    print(f"du/dz (attendu 3)        : err max = {(uz[1:-1] - 3).abs().max().item():.2e}")
    vx, vy, vz = g._grad_points(v_p, g.Zv_p)
    print(f"dv/dy (attendu -1)       : err max = {(vy[:, 1:-1, 1:-1] + 1).abs().max().item():.2e}")

    # advection semi-lagrangienne : u = a (x + z), v = w = 0  ->  u.grad(u) = a u
    # (le schéma n'est qu'approché : on vérifie l'erreur relative loin des bords)
    a = 0.01
    adv_u, _, _ = g._advection_terms(a * (xu + g.Zu_p), torch.zeros_like(g.Zv_p), torch.zeros_like(g.Zw_p))
    ref = a * a * (xu + g.Zu_p)[1:-1, 1:-1, 1:-1]
    inner = (slice(2, -2), slice(2, -2), slice(3, -3))
    rel = ((adv_u - ref)[inner].abs().max() / ref[inner].abs().max()).item()
    print(f"advection u.grad(u)      : err relative max = {rel:.2e}")

    # écoulement uniforme : l'advection doit être nulle
    adv_u, _, _ = g._advection_terms(torch.ones_like(g.Zu_p), torch.zeros_like(g.Zv_p), torch.zeros_like(g.Zw_p))
    print(f"advection uniforme       : max = {adv_u[inner].abs().max().item():.2e}")

    # terrain plat : la grille sigma doit redonner les différences régulières
    gf = SigmaGrid(torch.zeros(ny, nx), nz, top=float(nz), inflow_speed=1.0)
    vel = torch.randn(3, nz + 1, ny + 1, nx + 1)
    e1 = gf.energy(*gf.fields(vel), nu=0.5, lambda_p=10.0)
    e2 = gf.energy(*gf.fields(vel), nu=0.5, lambda_p=10.0, volume_weight=True)
    print(f"plat : E sans/avec poids volume (attendu égal) : {e1.item():.6f} / {e2.item():.6f}")
