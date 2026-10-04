import torch
import torch.nn.functional as F

from . import utils
from . import mac_grid

def advect_real_semi_lagrange(vel, rho, dt, rk_order = 3):
    return _advect_semi_lagrange_single_channel(
        vel=vel, rho=rho, dt=dt, rk_order=rk_order, sample_pos="center"
    )

def advect_node_semi_lagrange(vel, rho, dt, rk_order = 3):
    return _advect_semi_lagrange_single_channel(
        vel=vel, rho=rho, dt=dt, rk_order=rk_order, sample_pos="node"
    )

def advect_mac_semi_lagrange( vel, rho, dt, rk_order = 3):
    # advect in x direction
    rho_adv_x = _advect_semi_lagrange_single_channel(
        vel=vel, rho=rho[0:1, ...], dt=dt, rk_order=rk_order, sample_pos="mac_x"
    )
    # advect in y direction
    rho_adv_y = _advect_semi_lagrange_single_channel(
        vel=vel, rho=rho[1:2, ...], dt=dt, rk_order=rk_order, sample_pos="mac_y"
    )
    # concatenate back advected values into the same tensor
    rho_adv = torch.cat((rho_adv_x, rho_adv_y), dim=0)
    # advect in z direction in 3d case
    if len(vel.shape) == 4:
        rho_adv_z = _advect_semi_lagrange_single_channel(
            vel=vel, rho=rho[2:3, ...], dt=dt, rk_order=rk_order, sample_pos="mac_z"
        )
        rho_adv = torch.cat((rho_adv, rho_adv_z), dim=0)
    return rho_adv

def _advect_semi_lagrange_single_channel(vel, rho, dt, rk_order, sample_pos):
    # create mesh grid
    mgrid = utils.create_mesh_grid(
        rho.size(), device=rho.device, dtype=rho.dtype
    )
    # use runge kutta or euler method to create the back trace field
    back_trace = _runge_kutta(vel, mgrid, dt, rk_order, sample_pos)
    # use warped grid sample function from PyTorch to interpolate the grid
    # and bring the interporlated values to the current position
    rho_adv = utils.grid_sample(rho, back_trace)
    return rho_adv

def _euler(vel, mgrid, dt, sample_pos):
    if sample_pos == "node":
        raise NotImplementedError(
            "Euler step not implemented " "for node data advection."
        )
    get_pos_fcn = _map_sample_pos(sample_pos, rk=False)
    return mgrid - get_pos_fcn(vel) * dt

def _runge_kutta(vel, mgrid, dt, rk_order, sample_pos):
    grid_sample_fcn = _map_sample_pos(sample_pos, rk=True)
    if rk_order == 1:
        # return mgrid - grid_sample_fcn(vel, mgrid)*dt
        return _euler(vel, mgrid, dt, sample_pos)
    elif rk_order == 2:
        # find the mid point position using half of time step
        # back_trace_mid = mgrid - grid_sample_fcn(vel, mgrid)*dt*0.5
        back_trace_mid = _euler(vel, mgrid, 0.5 * dt, sample_pos)
        # find the mid point velocity
        return mgrid - grid_sample_fcn(vel, back_trace_mid) * dt
    elif rk_order == 3:
        # k1 = grid_sample_fcn(vel, mgrid)
        k1 = (mgrid - _euler(vel, mgrid, dt, sample_pos)) / dt
        k2 = grid_sample_fcn(vel, mgrid - 0.5 * dt * k1)
        k3 = grid_sample_fcn(vel, mgrid - 0.75 * dt * k2)
        return (
            mgrid - 2.0 / 9.0 * dt * k1 - 3.0 / 9.0 * dt * k2 - 4.0 / 9.0 * dt * k3
        )
    else:
        raise ValueError("Runge-kutta only support order 1, 2 or 3.")

def _map_sample_pos( sample_pos, rk):
    if sample_pos == "center":
        return mac_grid.grid_sample_center if rk else mac_grid.get_centered
    elif sample_pos == "node":
        return mac_grid.grid_sample_node
    elif sample_pos == "mac_x":
        return mac_grid.grid_sample_x if rk else mac_grid.get_mac_x
    elif sample_pos == "mac_y":
        return mac_grid.grid_sample_y if rk else mac_grid.get_mac_y
    elif sample_pos == "mac_z":
        return mac_grid.grid_sample_z if rk else mac_grid.get_mac_z
    else:
        raise ValueError("sample_pos {} not recognized.".format(sample_pos))

