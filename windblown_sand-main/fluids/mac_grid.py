import torch
import torch.nn.functional as F

from . import utils

def get_mac_x(grid):
    u_mac_x = grid[0:1, ...]

    v_mac_y = grid[1:2, ...]
    pad_x_neg = (1, 0, 0, 0) if len(grid.shape) == 3 else (1, 0, 0, 0, 0, 0)
    v_mac_y_pad_x = F.pad(v_mac_y, pad=pad_x_neg)
    v_centered = (v_mac_y_pad_x[..., 1:] + v_mac_y_pad_x[..., :-1]) * 0.5
    v_mac_x = (v_centered[..., 1:, :] + v_centered[..., :-1, :]) * 0.5
    pad_y_pos = (0, 0, 0, 1) if len(grid.shape) == 3 else (0, 0, 0, 1, 0, 0)
    v_mac_x = F.pad(v_mac_x, pad=pad_y_pos)
    vel_mac_x = torch.cat((u_mac_x, v_mac_x), dim=0)
    if len(grid.shape) == 4:
        w_mac_z = grid[2:3, ...]
        w_mac_z_pad_x = F.pad(w_mac_z, pad=pad_x_neg)
        w_centered = (w_mac_z_pad_x[..., 1:] + w_mac_z_pad_x[..., :-1]) * 0.5
        w_mac_x = (w_centered[:, 1:, ...] + w_centered[:, :-1, ...]) * 0.5
        pad_z_pos = (0, 0, 0, 0, 0, 1)
        w_mac_x = F.pad(w_mac_x, pad=pad_z_pos)
        vel_mac_x = torch.cat((vel_mac_x, w_mac_x), dim=0)

    return vel_mac_x

def get_mac_y(grid):
    v_mac_y = grid[1:2, ...]

    u_mac_x = grid[0:1, ...]
    pad_y_neg = (0, 0, 1, 0) if len(grid.shape) == 3 else (0, 0, 1, 0, 0, 0)
    u_mac_x_pad_y = F.pad(u_mac_x, pad=pad_y_neg)
    u_centered = (u_mac_x_pad_y[..., 1:, :] + u_mac_x_pad_y[..., :-1, :]) * 0.5
    u_mac_y = (u_centered[..., 1:] + u_centered[..., :-1]) * 0.5
    pad_x_pos = (0, 1, 0, 0) if len(grid.shape) == 3 else (0, 1, 0, 0, 0, 0)
    u_mac_y = F.pad(u_mac_y, pad=pad_x_pos)
    vel_mac_y = torch.cat((u_mac_y, v_mac_y), dim=0)

    if len(grid.shape) == 4:
        w_mac_z = grid[2:3, ...]
        w_mac_z_pad_y = F.pad(w_mac_z, pad=pad_y_neg)
        w_centered = (w_mac_z_pad_y[..., 1:, :] + w_mac_z_pad_y[..., :-1, :]) * 0.5
        w_mac_y = (w_centered[:, 1:, ...] + w_centered[:, :-1, ...]) * 0.5
        pad_z_pos = (0, 0, 0, 0, 0, 1)
        w_mac_y = F.pad(w_mac_y, pad=pad_z_pos)
        vel_mac_y = torch.cat((vel_mac_y, w_mac_y), dim=0)

    return vel_mac_y

def get_mac_z(grid):
    if len(grid.shape) == 3:
        raise RuntimeError("MACGridOperator.get_mac_z only supports 3D case.")
    u_mac_x = grid[0:1, ...]
    v_mac_y = grid[1:2, ...]
    pad_z_neg = (0, 0, 0, 0, 1, 0)
    u_mac_x_pad_z = F.pad(u_mac_x, pad=pad_z_neg)
    u_centered = (u_mac_x_pad_z[:, 1:, ...] + u_mac_x_pad_z[:, :-1, ...]) * 0.5
    u_mac_z = (u_centered[..., 1:] + u_centered[..., :-1]) * 0.5
    pad_x_pos = (0, 1, 0, 0, 0, 0)
    u_mac_z = F.pad(u_mac_z, pad=pad_x_pos)

    v_mac_y_pad_z = F.pad(v_mac_y, pad=pad_z_neg)
    v_centered = (v_mac_y_pad_z[:, 1:, ...] + v_mac_y_pad_z[:, :-1, ...]) * 0.5
    v_mac_z = (v_centered[..., 1:, :] + v_centered[..., :-1, :]) * 0.5
    pad_y_pos = (0, 0, 0, 1, 0, 0)
    v_mac_z = F.pad(v_mac_z, pad=pad_y_pos)

    w_mac_z = grid[2:3, ...]
    vel_mac_z = torch.cat((u_mac_z, v_mac_z, w_mac_z), dim=0)
    return vel_mac_z

def _mac_to_real_size(grid):
    return torch.Size([1] + [s-1 for s in grid.shape[1:]])

def grid_sample_center(grid, back_trace=None):
    """Interpolate velocity field onto cell center position."""
    if back_trace is None:  
        back_trace = utils.create_mesh_grid(
            _mac_to_real_size(grid),
            device=grid.device,
            dtype=grid.dtype,
        )
    # WARNING: might be inaccurate at non-solid boundaries,
    # since one valid boundary value is cut off.
    if len(grid.shape) == 3: # is 2D
        grid = grid[:, :-1, :-1]
    else:
        grid = grid[:, :-1, :-1, :-1]
    vel_x = utils.grid_sample(
        grid[0:1, ...],
        torch.cat(
            (
                back_trace[0:1, ...] + 0.5,
                back_trace[1:2, ...],
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    vel_y = utils.grid_sample(
        grid[1:2, ...],
        torch.cat(
            (
                back_trace[0:1, ...],
                back_trace[1:2, ...] + 0.5,
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    vel = torch.cat((vel_x, vel_y), dim=0)
    if len(grid.shape) == 4: # is 3D
        vel_z = utils.grid_sample(
            grid[2:3, ...],
            torch.cat(
                (
                    back_trace[0:1, ...],
                    back_trace[1:2, ...],
                    back_trace[2:3, ...] + 0.5,
                ),
                dim=0,
            ),
        )
        vel = torch.cat((vel, vel_z), dim=0)
    return vel

def grid_sample_node(grid, back_trace=None):
    """Interpolate velocity field onto cell node(corner) position."""
    if back_trace is None:
        back_trace = utils.create_mesh_grid(
            _mac_to_real_size(grid),
            device=grid.device,
            dtype=grid.dtype,
        )
    # WARNING: might be inaccurate at non-solid boundaries,
    # since one valid boundary value is cut off.
    vel_x = utils.grid_sample(
        grid[0:1, ...],
        torch.cat(
            (
                back_trace[0:1, ...],
                back_trace[1:2, ...] - 0.5,
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    vel_y = utils.grid_sample(
        grid[1:2, ...],
        torch.cat(
            (
                back_trace[0:1, ...] - 0.5,
                back_trace[1:2, ...],
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    vel = torch.cat((vel_x, vel_y), dim=0)
    if len(grid.shape) == 4: # is 3D
        raise NotImplementedError(
            "MACGridOperator.grid_sample_node:" "3D case not supported yet."
        )
    return vel

def grid_sample_x(grid, back_trace=None):
    """Interpolate velocity field onto mac_x position."""
    if back_trace is None:
        back_trace = utils.create_mesh_grid(
            grid.size(), device=grid.device, dtype=grid.dtype
        )
    vel_x = utils.grid_sample(grid[0:1, ...], back_trace)
    vel_y = _grid_sample_x(grid[1:2, ...], back_trace)
    vel = torch.cat((vel_x, vel_y), dim=0)
    if len(grid.shape) == 4: # is 3D
        vel_z = utils.grid_sample(
            grid[2:3, ...],
            torch.cat(
                (
                    back_trace[0:1, ...] - 0.5,
                    back_trace[1:2, ...],
                    back_trace[2:3, ...] + 0.5,
                ),
                dim=0,
            ),
        )
        vel = torch.cat((vel, vel_z), dim=0)
    return vel

def _grid_sample_x(grid, back_trace=None):
    """Interpolate velocity field component onto mac_x position."""
    if back_trace is None:
        back_trace = utils.create_mesh_grid(grid.size(), grid.dtype)
    grid_macx = utils.grid_sample(
        grid,
        torch.cat(
            (
                back_trace[0:1, ...] - 0.5,
                back_trace[1:2, ...] + 0.5,
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    return grid_macx

def grid_sample_y(grid, back_trace=None):
    """Interpolate velocity field onto mac_y position."""
    if back_trace is None:
        back_trace = utils.create_mesh_grid(
            grid.size(), device=grid.device, dtype=grid.dtype
        )
    vel_x = _grid_sample_y(grid[0:1, ...], back_trace)
    vel_y = utils.grid_sample(grid[1:2, ...], back_trace)
    vel = torch.cat((vel_x, vel_y), dim=0)
    if len(grid.shape) == 4: # is 3D
        vel_z = utils.grid_sample(
            grid[2:3, ...],
            torch.cat(
                (
                    back_trace[0:1, ...],
                    back_trace[1:2, ...] - 0.5,
                    back_trace[2:3, ...] + 0.5,
                ),
                dim=0,
            ),
        )
        vel = torch.cat((vel, vel_z), dim=0)
    return vel

def _grid_sample_y(grid, back_trace=None):
    """Interpolate velocity field component onto mac_y position."""
    if back_trace is None:
        back_trace = utils.create_mesh_grid(
            grid.size(), device=grid.device, dtype=grid.dtype
        )
    grid_macy = utils.grid_sample(
        grid,
        torch.cat(
            (
                back_trace[0:1, ...] + 0.5,
                back_trace[1:2, ...] - 0.5,
                back_trace[2:3, ...],
            ),
            dim=0,
        ),
    )
    return grid_macy

def grid_sample_z(grid, back_trace=None):
    if back_trace is None:
        back_trace = utils.create_mesh_grid(
            grid.size(), device=grid.device, dtype=grid.dtype
        )
    if len(grid.shape) == 3: # is 2D:
        raise RuntimeError("2d data does not support z direction.")
    vel_x = utils.grid_sample(
        grid[0:1, ...],
        torch.cat(
            (
                back_trace[0:1, ...] + 0.5,
                back_trace[1:2, ...],
                back_trace[2:3, ...] - 0.5,
            ),
            dim=0,
        ),
    )
    vel_y = utils.grid_sample(
        grid[1:2, ...],
        torch.cat(
            (
                back_trace[0:1, ...],
                back_trace[1:2, ...] + 0.5,
                back_trace[2:3, ...] - 0.5,
            ),
            dim=0,
        ),
    )
    vel_z = utils.grid_sample(grid[2:3, ...], back_trace)
    vel = torch.cat((vel_x, vel_y, vel_z), dim=0)
    return vel



def fast_centered(vel):
    out = []
    out.append(vel[0:1, ..., :-1, 1:] + vel[0:1, ..., :-1, :-1])
    out.append(vel[1:2, ..., 1:, :-1] + vel[1:2, ..., :-1, :-1])

    if vel.shape[0] == 3:
        if len(vel.shape) == 4:
            out = [
                out[0][:, :-1], 
                out[1][:, :-1], 
                (vel[2:3, 1:, :-1, :-1] + vel[1:2, :-1, :-1, :-1])
            ]
        else:
            out.append(2*vel[2:3, :-1, :-1])

    return .5 * torch.cat(out, dim = 0)

def fast_norm(vel):
    return fast_centered(vel**2).sum(dim = 0).sqrt()
    