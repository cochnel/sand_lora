import torch
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F

from . import level_grid

def _is_2d(phi):
    return len(phi.shape) == 3

def _is_3d(phi):
    return len(phi.shape) == 4

def build_pressure_matrix(fractions):

    if _is_2d(fractions):
        Adiag_fluid = (
            fractions[0:1, :-1, :-1]
            + fractions[0:1, :-1, 1:]
            + fractions[1:2, :-1, :-1]
            + fractions[1:2, 1:, :-1]
        )
        Aplusx_fluid = -fractions[0:1, :-1, 1:]
        Aplusy_fluid = -fractions[1:2, 1:, :-1]
        pad_x_pos = (0, 1, 0, 0)
        pad_y_pos = (0, 0, 0, 1)
    else:
        Adiag_fluid = (
            fractions[0:1, :-1, :-1, :-1]
            + fractions[0:1, :-1, :-1, 1:]
            + fractions[1:2, :-1, :-1, :-1]
            + fractions[1:2, :-1, 1:, :-1]
            + fractions[2:3, :-1, :-1, :-1]
            + fractions[2:3, 1:, :-1, :-1]
        )
        Aplusx_fluid = -fractions[0:1, :-1, :-1, 1:]
        Aplusy_fluid = -fractions[1:2, :-1, 1:, :-1]
        Aplusz_fluid = -fractions[2:3, 1:, :-1, :-1]
        pad_x_pos = (0, 1, 0, 0, 0, 0)
        pad_y_pos = (0, 0, 0, 1, 0, 0)
        pad_z_pos = (0, 0, 0, 0, 0, 1)
 
    is_fluid = ~level_grid.is_obs(fractions)

    Adiag = torch.where(
        is_fluid,
        Adiag_fluid,
        torch.ones(is_fluid.size(), device=fractions.device, dtype=fractions.dtype),
    )
    # is_fluid_pad_x_pos = F.pad(is_fluid, pad=pad_x_pos)
    # Aplusx = torch.where(
    #     (
    #         (is_fluid)
    #         * (is_fluid_pad_x_pos[..., 1:])
    #     ),
    #     Aplusx_fluid,
    #     torch.zeros(is_fluid.size(), device=fractions.device, dtype=fractions.dtype),
    # )
    # is_fluid_pad_y_pos = F.pad(is_fluid, pad=pad_y_pos)
    # Aplusy = torch.where(
    #     (
    #         (is_fluid)
    #         * (is_fluid_pad_y_pos[..., 1:, :])
    #     ),
    #     Aplusy_fluid,
    #     torch.zeros(is_fluid.size(), device=fractions.device, dtype=fractions.dtype),
    # )
    Aplusx = Aplusx_fluid
    Aplusy = Aplusy_fluid
    
    Adiag = Adiag + 1e-3
    A = torch.stack([Adiag, Aplusx, Aplusy])
    if _is_3d(fractions):
        is_fluid_pad_z_pos = F.pad(is_fluid, pad=pad_z_pos)
        # Aplusz = torch.where(
        #     (
        #         (is_fluid)
        #         * (is_fluid_pad_z_pos[:, 1:, ...])
        #     ),
        #     Aplusz_fluid,
        #     torch.zeros(is_fluid.size(), device=fractions.device, dtype=fractions.dtype),
        # )
        Aplusz = Aplusz_fluid
        A = torch.cat((A, Aplusz.unsqueeze(0)), dim=0)
    return A

# overload
def build_preconditioner(A):
    """Build Incomplete Poisson Preconditioner from A matrices. The output
    is a torch.sparse.FloatTensor matrix.

    Assumes that the boundary of the scene never contains fluids

    """
    Adiag = A[0, ...]
    Aplusx = A[1, ...]
    Aplusy = A[2, ...]
    AdiagInv = torch.where(
        Adiag == 0,
        torch.zeros(
            Adiag.size(), device=A.device, dtype=A.dtype
        ),
        1.0 / Adiag,
    )
    PInvplusx = -Aplusx * AdiagInv
    PInvplusy = -Aplusy * AdiagInv
    PInvplusz = None
    pad_x_neg = (1, 0, 0, 0) if _is_2d(Adiag) else (1, 0, 0, 0, 0, 0)
    pad_y_neg = (0, 0, 1, 0) if _is_2d(Adiag) else (0, 0, 1, 0, 0, 0)
    PInvplusx_pad_neg = F.pad(PInvplusx, pad=pad_x_neg)
    PInvplusy_pad_neg = F.pad(PInvplusy, pad=pad_y_neg)
    PInvdiag = (
        1.0
        + PInvplusx_pad_neg[..., :-1] * PInvplusx_pad_neg[..., :-1]
        + PInvplusy_pad_neg[..., :-1, :] * PInvplusy_pad_neg[..., :-1, :]
    )
    PInv = torch.stack([PInvdiag, PInvplusx, PInvplusy])
    if _is_3d(Adiag):
        Aplusz = A[3, ...]
        PInvplusz = -Aplusz * AdiagInv
        pad_z_neg = (0, 0, 0, 0, 1, 0)
        PInvplusz_pad_neg = F.pad(PInvplusz, pad=pad_z_neg)
        PInvdiag += PInvplusz_pad_neg[:, :-1, ...] * PInvplusz_pad_neg[:, :-1, ...]
        PInv = torch.stack([PInvdiag, PInvplusx, PInvplusy, PInvplusz])

    return PInv

def solve_linear_system(rhs, A, PInv = None, tol=1e-6, limit=6000, direct=False):
    if direct:
        solve_linear_system_fcn = _solve_linear_system_direct
    else:
        solve_linear_system_fcn = _solve_linear_system

    class SolveLinearSystemOverwriter(Function):
        @staticmethod
        def forward(ctx, rhs, A):
            p = solve_linear_system_fcn(rhs, A, PInv, tol, limit)
            ctx.save_for_backward(A, p)
            return p

        @staticmethod
        def backward(ctx, grad_output):
            # use the fact that transpose(A)=A
            A, p = ctx.saved_tensors
            grad_rhs = solve_linear_system_fcn(grad_output, A, tol=1e-10)
            grad_Adiag = -grad_rhs * p
            padx = (0, 1, 0, 0) if _is_2d(p) else (0, 1, 0, 0, 0, 0)
            pady = (0, 0, 0, 1) if _is_2d(p) else (0, 0, 0, 1, 0, 0)
            padz = None if _is_2d(p) else (0, 0, 0, 0, 0, 1)
            p_padx = F.pad(p, pad=padx)
            grad_rhs_padx = F.pad(grad_rhs, pad=padx)
            p_pady = F.pad(p, pad=pady)
            grad_rhs_pady = F.pad(grad_rhs, pad=pady)
            grad_Aplusx = -grad_rhs * p_padx[..., 1:] - grad_rhs_padx[..., 1:] * p
            grad_Aplusy = (
                -grad_rhs * p_pady[..., 1:, :] - grad_rhs_pady[..., 1:, :] * p
            )
            grad_A = torch.stack([grad_Adiag, grad_Aplusx, grad_Aplusy])
            if _is_3d(p):
                p_padz = F.pad(p, pad=padz)
                grad_rhs_padz = F.pad(grad_rhs, pad=padz)
                grad_Aplusz = (
                    -grad_rhs * p_padz[:, 1:, ...] - grad_rhs_padz[:, 1:, ...] * p
                )
                grad_A = torch.cat((grad_A, grad_Aplusz.unsqueeze(0)), dim=0)
            return grad_rhs, grad_A

    solve = SolveLinearSystemOverwriter.apply
    p = solve(rhs, A)
    return p

def _solve_linear_system(rhs, A, PInv = None, tol=1e-6, limit=6000):

    if rhs.size() != A[0].size():
        raise ValueError("rhs and Amatrices have different sizes.")
    # initialization
    p = torch.zeros(rhs.size(), device=rhs.device, dtype=rhs.dtype)
    r = rhs.clone()  # residual vector, initialized as rhs
    if _norm(r) <= tol:
        # print('# of CG itr: {}'.format(0))
        return p
    z = _apply_preconditioner(r, PInv)
    s = z.clone()  # search vector
    sigma = _dot_product(z, r)  # intermediate value to compute step size alpha
    # main loop
    for t in range(limit):
        z = _Mv(A, s)
        alpha = sigma / _dot_product(z, s)
        p = p + alpha * s
        r = r - alpha * z
        if _norm(r) <= tol:
            # print('# of CG itr: {}, residual {}'.format(t, self._norm(r)))
            return p
        z = _apply_preconditioner(r, PInv)
        sigma_new = _dot_product(z, r)
        beta = sigma_new / sigma
        s = z + beta * s
        sigma = sigma_new
    print(
        ("Iteration {} exceeded, returned results with residual " "{}").format(
            limit, _norm(r)
        )
    )
    return p


def _solve_linear_system_direct(rhs, A, PInv = None, tol=None, limit=None):
    if _is_3d(rhs):
        raise NotImplementedError(
            "LSSolver._solve_linear_system_direct " "does not support 3D."
        )
    A = _convert_sparse_to_dense(A)
    H, W = rhs.size(1), rhs.size(2)
    nonzero_rows = (A != 0).any(dim=-2).nonzero().squeeze()
    A_nonzero_rows = A[nonzero_rows, :]
    A_nonzero_rows = A_nonzero_rows[:, nonzero_rows]
    # put RHS into vector form
    rhs_vec = rhs.contiguous().view(-1, 1)
    b = rhs_vec[nonzero_rows, :]
    # direct LU solver to compute pressure
    p, _ = torch.solve(b, A_nonzero_rows)
    # put pressure back into matrix form
    nonzero_idx = np.unravel_index(nonzero_rows.cpu().numpy(), [H, W])
    pressure = torch.zeros_like(rhs)
    pressure[:, nonzero_idx[0], nonzero_idx[1]] = p.permute(1, 0)
    return pressure

def _convert_sparse_to_dense(A):
    if _is_3d(A[0]):
        raise NotImplementedError(
            "LSSolver._convert_sparse_to_dense " "does not support 3D."
        )
    Adiag, Aplusx, Aplusy = A[0], A[1], A[2]
    H, W = Adiag.size(1), Adiag.size(2)
    A = torch.zeros(H * W, H * W, device=A.device, dtype=Adiag.dtype)
    y = np.repeat(np.arange(H - 1), W - 1)
    x = np.tile(np.arange(W - 1), H - 1)
    idx_x_y = np.ravel_multi_index([[y], [x]], dims=(H, W))
    idx_x1_y = np.ravel_multi_index([[y], [x + 1]], dims=(H, W))
    idx_x_y1 = np.ravel_multi_index([[y + 1], [x]], dims=(H, W))
    A[idx_x_y, idx_x_y] = Adiag[:, :-1, :-1].contiguous().view((H - 1) * (W - 1))
    A[idx_x_y, idx_x1_y] = Aplusx[:, :-1, :-1].contiguous().view((H - 1) * (W - 1))
    A[idx_x1_y, idx_x_y] = Aplusx[:, :-1, :-1].contiguous().view((H - 1) * (W - 1))
    A[idx_x_y, idx_x_y1] = Aplusy[:, :-1, :-1].contiguous().view((H - 1) * (W - 1))
    A[idx_x_y1, idx_x_y] = Aplusy[:, :-1, :-1].contiguous().view((H - 1) * (W - 1))
    return A

def _apply_preconditioner(r, PInv):
    """Apply preconditioner on input vector r: z = PInv r"""
    if PInv is not None:
        z = _Mv(PInv, r)
        return z
    else:
        return r

def _Mv(A, b):
    """Matrix-vector multiplication of A and b

    Args:
        A(list): (5-/7- point lapacian) matrix, containing diag, plusx,
                    plusy, plusz
        b(torch.Tensor): grid-represented vector, should be of the same
                            size as Aidag
    """
    Adiag = A[0, ...]
    Aplusx = A[1, ...]
    Aplusy = A[2, ...]
    if Adiag.size() != b.size():
        raise ValueError("Mismatch of sizes between matrix and vector.")
    # pad to get Aplusx(i-1,j,k), Aplusy(i,j-1,k) and Aplusz(i,j,k-1)
    pad_x_neg = (1, 0, 0, 0) if _is_2d(b) else (1, 0, 0, 0, 0, 0)
    pad_y_neg = (0, 0, 1, 0) if _is_2d(b) else (0, 0, 1, 0, 0, 0)
    pad_z = None if _is_2d(b) else (0, 0, 0, 0, 1, 0)
    Aplusx_pad_neg = F.pad(Aplusx, pad=pad_x_neg)
    Aplusy_pad_neg = F.pad(Aplusy, pad=pad_y_neg)
    if _is_3d(b):
        Aplusz = A[3, ...]
        Aplusz_pad_neg = F.pad(Aplusz, pad=pad_z)
    # pad to get b(i-1, j, k), b(i+1, j, k) and so on
    pad_x = (1, 1, 0, 0) if _is_2d(b) else (1, 1, 0, 0, 0, 0)
    pad_y = (0, 0, 1, 1) if _is_2d(b) else (0, 0, 1, 1, 0, 0)
    pad_z = None if _is_2d(b) else (0, 0, 0, 0, 1, 1)
    b_pad_x = F.pad(b, pad=pad_x)
    b_pad_y = F.pad(b, pad=pad_y)
    if _is_3d(b):
        b_pad_z = F.pad(b, pad=pad_z)
    # 'Matrix' A multiplies 'vector' b
    Ab = (
        Adiag * b  # Adiag(i,j,k)*b(i,j,k)
        + Aplusx * b_pad_x[..., 2:]  # Aplusx(i,j,k)*b(i+1,j,k)
        + Aplusx_pad_neg[..., :-1] * b_pad_x[..., :-2]  # Aplusx(i-1,j,k)*b(i-1,j,k)
        + Aplusy * b_pad_y[..., 2:, :]  # Aplusy(i,j,k)*b(i,j+1,k)
        + Aplusy_pad_neg[..., :-1, :] * b_pad_y[..., :-2, :]
    )  # Aplusy(i,j-1,k)*b(i,j-1,k)
    if _is_3d(b):
        Ab = (
            Ab
            + Aplusz * b_pad_z[:, 2:, ...]  # Aplusz(i,j,k)*b(i,j,k+1)
            + Aplusz_pad_neg[:, :-1, ...] * b_pad_z[:, :-2, ...]
        )  # Aplusz(i,j-1,k)*b(i,j-1,k)
    return Ab

def _dot_product(x, y):
    """dot product of two grid-represented vectors x and y."""
    return torch.sum(x * y)

def _norm(x, p=np.inf):
    """defines the norm of a grid-represented vector.

    Args:
        x(torch.Tensor): input grid-represented vector
        p: order of norm, can be int,float,inf,-inf,'fro','nuc'
    """
    x_vec = x.contiguous().view(-1)
    norm = torch.norm(x_vec, p=p)
    return norm



