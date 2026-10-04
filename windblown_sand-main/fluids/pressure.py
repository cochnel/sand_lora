import torch
from . import level_grid
from . import ls_solver
import torch.nn.functional as F


def solve_pressure(pressure_fractions, fractions, vel, precond = False, cg_accu=1e-6, limit=6000):

    """
    pressure_fractions is the area of free fluid for computing pressure matrix
    fraction further includes inflow area for computing divergence including inflow vel

    """

    rhs = _compute_pressure_rhs(fractions, vel)
    pressure = _solve_poisson_system(pressure_fractions, rhs, precond, cg_accu, limit=6000)

    vel = _correct_velocity(vel, pressure)
    
    return vel, pressure

def _compute_pressure_rhs(fractions, vel):

    if len(vel.shape) == 3: # is 2D
        rhs = -(
                fractions[0:1, :-1, 1:] * vel[0:1, :-1, 1:]
                - fractions[0:1, :-1, :-1] * vel[0:1, :-1, :-1]
                + fractions[1:2, 1:, :-1] * vel[1:2, 1:, :-1]
                - fractions[1:2, :-1, :-1] * vel[1:2, :-1, :-1]
            )
    else:
        rhs = -(
                fractions[0:1, :-1, :-1, 1:] * vel[0:1, :-1, :-1, 1:]
                - fractions[0:1, :-1, :-1, :-1] * vel[0:1, :-1, :-1, :-1]
                + fractions[1:2, :-1, 1:, :-1] * vel[1:2, :-1, 1:, :-1]
                - fractions[1:2, :-1, :-1, :-1] * vel[1:2, :-1, :-1, :-1]
                + fractions[2:3, 1:, :-1, :-1] * vel[2:3, 1:, :-1, :-1]
                - fractions[2:3, :-1, :-1, :-1] * vel[2:3, :-1, :-1, :-1]
            )
    return rhs

def _solve_poisson_system(fractions, rhs, precond, cg_accu, limit):

    A = ls_solver.build_pressure_matrix(fractions)

    PInv = ls_solver.build_preconditioner(A) if precond else None

    return ls_solver.solve_linear_system(
        rhs, A, PInv, direct=False, tol=cg_accu, limit = limit
    )

def _get_gradient(grid):
        pad = (1, 1, 1, 1) if len(grid.shape) == 3 else (1, 1, 1, 1, 1, 1)
        grid_pad = F.pad(grid.unsqueeze(0), pad=pad, mode="replicate").squeeze(0)
        if len(grid.shape) == 3:
            grad_x = grid_pad[:, 1:, 1:] - grid_pad[:, 1:, :-1]
            grad_y = grid_pad[:, 1:, 1:] - grid_pad[:, :-1, 1:]
            grad = torch.cat((grad_x, grad_y), dim=0)
        else:
            grad_x = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, 1:, 1:, :-1]
            grad_y = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, 1:, :-1, 1:]
            grad_z = grid_pad[:, 1:, 1:, 1:] - grid_pad[:, :-1, 1:, 1:]
            grad = torch.cat((grad_x, grad_y, grad_z), dim=0)
        return grad

def _correct_velocity(vel, pressure):
    pressure_grad = _get_gradient(pressure)
    vel_corrected = vel - pressure_grad
    return vel_corrected
