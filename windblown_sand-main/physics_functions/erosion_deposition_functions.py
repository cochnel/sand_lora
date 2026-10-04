import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import torch
import torch.nn.functional as F
import math

sys.path.append('../')

from fluids import mac_grid


def Qmax_Glen98(wind_norm , speed_limit):
    # assumes a non 0 wind norm
    return .68*speed_limit/wind_norm*(wind_norm+speed_limit)*(wind_norm-speed_limit).clamp(0)

def Qmax_Zingg1953(wind_norm):
    # assumes a non 0 wind norm
    return (0.83*(0.31/0.25)**(3/4)*(1.2/9.81)*wind_norm**3).clamp(0)

def cell_to_upwind(ux, uy, s):

    s_x = torch.where(ux[..., :, 1:-1] > 0, s[:, :, :-1], s[:,  :, 1:])
    s_y = torch.where(uy[..., 1:-1, :] > 0, s[:, :-1, :], s[:, 1:,  :])

    return s_x, s_y

def transport_div_staggered(ux, uy, s, dx):

    s_x, s_y = cell_to_upwind(ux, uy, s)

    #h, w+1
    usx = (ux[..., :, 1:-1]*s_x)[..., 1:-1, :]
    usy = (uy[..., 1:-1, :]*s_y)[..., 1:-1]

    # compute div (u s)
    dusx = (usx[:, :, 1:] - usx[:, :, :-1])/dx #h, w
    dusy = (usy[:, 1:, :] - usy[:, :-1, :])/dx #h, w

    return dusx + dusy

def div_staggered(ux, uy, dx):

    # pad velocities so that the divergence is defined everywhere
    ux = F.pad(ux, (1, 1, 1, 1), mode = 'replicate')
    uy = F.pad(uy, (1, 1, 1, 1), mode = 'replicate')

    ux = (ux)[..., 1:-1, :]
    uy = (uy)[..., 1:-1]

    dux = (ux[:, :, 1:] - ux[:, :, :-1])/dx #h, w
    duy = (uy[:, 1:, :] - uy[:, :-1, :])/dx #h, w

    return dux + duy

def _project_wind_on_ground_scalar(field, surface_heights):

    """
    surface_heights: H, W
    Field: H+1, W+1
    """

    heights_round = torch.round(surface_heights).long()[None, None]
    heights_decimal = (surface_heights%1)[None, None] + 1e-2

    return ((1-heights_decimal) * torch.take_along_dim(field, heights_round,   1) +
               heights_decimal  * torch.take_along_dim(field, heights_round+1, 1))[:, 0]



def project_wind_on_ground(vel , surface_heights):

    """
    Input  (vel, padded in X and Y)  => (3, D, H+3, W+3)
           (surface_hieghts, padded) => (H+2, W+2)
    Output Projected vel, unpadded => (2, H+1, W+1)
    """

    vx =  _project_wind_on_ground_scalar(vel[0:1, :, :-1, 1:-1], -.5+.5*(surface_heights[..., 1:] + surface_heights[..., :-1]))
    vy =  _project_wind_on_ground_scalar(vel[1:2, :, 1:-1, :-1], -.5+.5*(surface_heights[..., 1:, :] + surface_heights[..., :-1, :]))
    vz =  _project_wind_on_ground_scalar(vel[2:3, :, :-1, :-1], surface_heights)


    wind2d = torch.cat([vx[..., 1:, :], vy[..., 1:], vz[..., 1:, 1:]], dim = 0)
    # wind2d = torch.cat([vx[..., 1:, :], vy[..., 1:]], dim = 0)
    return wind2d


## OLD VERSION
# def shear_velocity(projected_wind, v_real, epsilon):
#     # projected_wind *= (v_real/20)

#     # return (mac_grid.fast_norm(projected_wind))*(v_real/20) + epsilon
#     # return mac_grid.fast_norm(projected_wind)*(1/20)+ epsilon
#     print(mac_grid.fast_norm(projected_wind*(v_real/20)).shape)
#     return mac_grid.fast_norm(projected_wind*(v_real/20)) + epsilon

def shear_velocity(staggered_vx , staggered_vy, v_real, epsilon):

    staggered_vx = staggered_vx*(v_real/20) + epsilon
    staggered_vy = staggered_vy*(v_real/20) + epsilon
    # staggered_vx = u_ground[0:1, :-1, :]
    # staggered_vy = u_ground[1:2, :, :-1]
    staggered_u_norm_x = staggered_vx[..., 1:-1]**2 +1/4*(staggered_vy[..., :-1, :-1]**2 + staggered_vy[..., 1:, 1:]**2 + staggered_vy[..., 1:, :-1]**2 + staggered_vy[..., :-1, 1:]**2)
    staggered_u_norm_y = staggered_vy[..., 1:-1, :]**2 +1/4*(staggered_vx[..., :-1, :-1]**2 + staggered_vx[..., 1:, 1:]**2 + staggered_vx[..., 1:, :-1]**2 + staggered_vx[..., :-1, 1:]**2)
    return staggered_u_norm_x, staggered_u_norm_y

def transport_sand(ux, uy, Q_x, Q_y, dx, dt_rho, sand_heights):

    uQx = (ux[..., :, 1:-1]*Q_x)[..., 1:-1, :]
    uQy = (uy[..., 1:-1, :]*Q_y)[..., 1:-1]

    uQx = (ux[..., :, 1:-1]*Q_x)[..., 1:-1, :]
    uQy = (uy[..., 1:-1, :]*Q_y)[..., 1:-1]

    #uQx: h, w+1

    out_going = (uQx[..., 1:].clamp(0) + (-uQx[..., :-1]).clamp(0) +
                 uQy[..., 1:, :].clamp(0) + (-uQx[..., :-1]).clamp(0))

    # available = (sand_heights[..., 1:-1, 1:-1]*dx/dt_rho / out_going).clamp(max = 1)
    available = (sand_heights[..., 1:-1, 1:-1]*dx/dt_rho / out_going)
    scale_x, scale_y = cell_to_upwind(ux[..., 1:-1, 1:-1], uy[..., 1:-1, 1:-1], available)

    available2 = (sand_heights[..., 1:-1, 1:-1]*dx/dt_rho)

    scale_x, scale_y = cell_to_upwind(ux[..., 1:-1, 1:-1], uy[..., 1:-1, 1:-1], available2.unsqueeze(0))

    uQx = torch.clamp(uQx[..., 1:-1], max=scale_x)
    uQy = torch.clamp(uQy[..., 1:-1, :], max=scale_y)

    duQx = (uQx[:, :, 1:] - uQx[:, :, :-1])/dx #h, w
    duQy = (uQy[:, 1:, :] - uQy[:, :-1, :])/dx #h, w

    duQx = duQx[..., 1:-1, :]
    duQy = duQy[..., 1:-1]

    F_salt = duQx[0] + duQy[1]  # EE: corrigé duQx[0]+duQx[0] → duQx[0]+duQy[1]
    F_salt = F.pad(F_salt , (1,1,1,1) , "constant")

    return F_salt

def compute_normals(height_map):

    nrx_x = (height_map[:, 1:]-height_map[:, :-1])
    nry_y = (height_map[1:, :]-height_map[:-1, :])
    nrx_x_pad , nry_y_pad = F.pad(nrx_x, (1, 1, 0, 0), "constant", 0) , F.pad(nry_y, (0 ,0 , 1, 1), "constant", 0)

    nry_x_pad = (1/4)*(nry_y_pad[1:, 1:] + nry_y_pad[1:, :-1] + nry_y_pad[:-1, 1:] + nry_y_pad[:-1, :-1])
    nrx_y_pad = (1/4)*(nrx_x_pad[1:, 1:] + nrx_x_pad[1:, :-1] + nrx_x_pad[:-1, 1:] + nrx_x_pad[:-1, :-1])

    nry_x_pad , nrx_y_pad = F.pad(nry_x_pad, (1, 1, 0, 0), "constant", 0) , F.pad(nrx_y_pad, (0 ,0 , 1, 1), "constant", 0)

    nrz_x_pad , nrz_y_pad = torch.ones_like(nry_x_pad) , torch.ones_like(nrx_y_pad)

    normal_magnitude_x = torch.sqrt(nrx_x_pad**2 + nry_x_pad**2 + nrz_x_pad**2)
    nrx_x_pad /= normal_magnitude_x
    nry_x_pad /= normal_magnitude_x
    nrz_x_pad /= normal_magnitude_x

    normal_magnitude_y = torch.sqrt(nrx_y_pad**2 + nry_y_pad**2 + nrz_y_pad**2)
    nrx_y_pad /= normal_magnitude_y
    nry_y_pad /= normal_magnitude_y
    nrz_y_pad /= normal_magnitude_y

    return torch.cat([nrx_x_pad.unsqueeze(0), nry_x_pad.unsqueeze(0), nrz_x_pad.unsqueeze(0)], dim = 0) , torch.cat([nrx_y_pad.unsqueeze(0), nry_y_pad.unsqueeze(0), nrz_y_pad.unsqueeze(0)], dim = 0)


def erosion_deposition(sand_heights, wind2D, sp_limit, dt_salt, v_real, m):

    dt = dt_salt
    dx = 1
    rho = 1
    epsilon = 1e-3

    speed_limit = sp_limit

    # nx, ny, nz = compute_normals(sand_heights)
    staggered_vx = wind2D[:, :-1, :]
    staggered_vy = wind2D[:, :, :-1]

    normals_staggered_x , normals_staggered_y = compute_normals(sand_heights)

    staggered_vx = staggered_vx - staggered_vx*normals_staggered_x
    staggered_vy = staggered_vy - staggered_vy*normals_staggered_y

    shear_vel_x , shear_vel_y = shear_velocity(staggered_vx, staggered_vy, v_real, epsilon)

    Qmax_x , Qmax_y = Qmax_Zingg1953(shear_vel_x) , Qmax_Zingg1953(shear_vel_y)

    F_salt  = transport_sand(staggered_vx, staggered_vy, Qmax_x, Qmax_y, dx, dt/rho , sand_heights)
    F_salt *= m

    added_sand = (dt/rho)*F_salt

    # plt.imshow(added_sand.detach().cpu())
    # plt.colorbar()
    # plt.show()

    sand_heights = (sand_heights[..., 1:-1, 1:-1]-added_sand).clamp(0)


    return sand_heights, F_salt

def avalanche_nicolas(ground_heights, sand_heights, nb_iterations , dt , angle):

    tan = math.tan(math.radians(angle))

    dx=1
    dt=.1
    rho=1
    eps = 1e-3
    # angle = .3
    # threshold = np.tan(angle)*(1/128)

    heights_sum = sand_heights + ground_heights

    for step in range(nb_iterations):
        drx = (heights_sum[:, 1:] - heights_sum[:, :-1])
        threshold_x = tan*torch.ones_like(drx)
        dry = (heights_sum[1:, :] - heights_sum[:-1, :])
        threshold_y = tan*torch.ones_like(dry)

        drx_pad , dry_pad = F.pad(drx, (1, 1, 0, 0), "constant", 1) , F.pad(dry, (0 ,0 , 1, 1), "constant", 1)

        drx_norm = torch.sqrt(drx**2 + (1/4)*(dry_pad[1:, 1:]**2 + dry_pad[1:, :-1]**2 + dry_pad[:-1, 1:]**2 + dry_pad[:-1, :-1]**2))+eps
        dry_norm = torch.sqrt(dry**2 + (1/4)*(drx_pad[1:, 1:]**2 + drx_pad[1:, :-1]**2 + drx_pad[:-1, 1:]**2 + drx_pad[:-1, :-1]**2))+eps

        # CHECK TORCH.MAX OR MAXIMUM

        critical_slope_x = torch.maximum(drx_norm-threshold_x , torch.zeros_like((drx_norm)))
        s_upwind_x = torch.where(drx > 0, sand_heights[:, 1:] , sand_heights[:, :-1])
        sign_x= drx/(torch.abs(drx)+eps)
        staggered_rx = sign_x*torch.minimum(torch.abs(drx)*critical_slope_x/drx_norm , s_upwind_x/dt)

        critical_slope_y = torch.maximum(dry_norm-threshold_y , torch.zeros_like((dry_norm)))
        s_upwind_y = torch.where(dry > 0, sand_heights[1:,:] , sand_heights[:-1, :])
        sign_y = dry/(torch.abs(dry)+eps)
        staggered_ry = sign_y*torch.minimum(torch.abs(dry)*critical_slope_y/dry_norm , s_upwind_y/dt)

        F_avalanche = div_staggered(staggered_rx[None] , staggered_ry[None] , dx)[0]
        delta_sand = (dt/rho)*F_avalanche
        sand_heights = (sand_heights+delta_sand).clamp(0)
        heights_sum = sand_heights+ground_heights

    return sand_heights
