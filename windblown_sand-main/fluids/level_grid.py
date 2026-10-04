import torch
import torch.nn.functional as F


def _is_2d(phi):
    return len(phi.shape) == 3

def _is_3d(phi):
    return len(phi.shape) == 4

def _get_line_fractions(phi, direction="z"):
    """Computes the line fraction of fluid for each cell given level set

    Args:
        phi(torch.Tensor): LevelGrid of size [C, (D+1), H+1, W+1]
        direction(string): surface normal direction to choose which
            surface to compute line fraction, should be one of 'x', 'y', 'z'
    """
    if direction == "x":
        assert _is_3d(phi) # is 3D
        left = phi[..., :-1, :]
        right = phi[..., 1:, :]
        bottom = phi[:, :-1, ...]
        up = phi[:, 1:, ...]
        pad_vertical_pos = (0, 0, 0, 0, 0, 1)
        pad_horizontal_pos = (0, 0, 0, 1, 0, 0)
    elif direction == "y":
        assert _is_3d(phi) # is 3D
        left = phi[..., :-1]
        right = phi[..., 1:]
        bottom = phi[:, :-1, ...]
        up = phi[:, 1:, ...]
        pad_vertical_pos = (0, 0, 0, 0, 0, 1)
        pad_horizontal_pos = (0, 1, 0, 0, 0, 0)
    elif direction == "z":
        left = phi[..., :-1]
        right = phi[..., 1:]
        bottom = phi[..., :-1, :]
        up = phi[..., 1:, :]
        pad_vertical_pos = (
            (0, 0, 0, 1) if _is_2d(phi) else (0, 0, 0, 1, 0, 0)
        )
        pad_horizontal_pos = (
            (0, 1, 0, 0) if _is_2d(phi) else (0, 1, 0, 0, 0, 0)
        )
    else:
        raise ValueError("direction should be one of x, y, z.")
    max_ub = torch.max(up, bottom)
    min_ub = torch.min(up, bottom)
    fraction_horizontal = max_ub / (max_ub - min_ub + 1e-7)
    fraction_horizontal = torch.clamp(fraction_horizontal, 0.0, 1.0)
    fraction_horizontal = F.pad(
        fraction_horizontal.unsqueeze(0), pad=pad_vertical_pos, mode="replicate"
    ).squeeze(0)
    max_lr = torch.max(left, right)
    min_lr = torch.min(left, right)
    fraction_vertical = max_lr / (max_lr - min_lr + 1e-7)
    fraction_vertical = torch.clamp(fraction_vertical, 0.0, 1.0)
    fraction_vertical = F.pad(
        fraction_vertical.unsqueeze(0), pad=pad_horizontal_pos, mode="replicate"
    ).squeeze(0)
    fraction = torch.cat((fraction_horizontal, fraction_vertical), dim=0)
    return fraction

def _get_face_fractions(line_fractions, phi, direction):
    """Computes face fractions given line fractions for a plane.

    Args:
        line_fractions(torch.Tensor): MACGrid of size [C, D+1, H+1, W+1]
        phi(torch.Tensor): LevelGrid of size [C, D+1, H+1, W+1]
        direction: indicates the surface normal of the line_fractions
    """
    assert _is_3d(phi)
    if direction == "x":
        left = line_fractions[0:1, :-1, :-1, :]
        bottom = line_fractions[1:2, :-1, :-1, :]
        right = line_fractions[0:1, :-1, 1:, :]
        up = line_fractions[1:2, 1:, :-1, :]
        phi_00 = phi[:, :-1, :-1, :]
        phi_01 = phi[:, 1:, :-1, :]
        phi_11 = phi[:, 1:, 1:, :]
        pad_to_mac = (0, 0, 0, 1, 0, 1)
    elif direction == "y":
        left = line_fractions[0:1, :-1, :, :-1]
        bottom = line_fractions[1:2, :-1, :, :-1]
        right = line_fractions[0:1, :-1, :, 1:]
        up = line_fractions[1:2, 1:, :, :-1]
        phi_00 = phi[:, :-1, :, :-1]
        phi_01 = phi[:, 1:, :, :-1]
        phi_11 = phi[:, 1:, :, 1:]
        pad_to_mac = (0, 1, 0, 0, 0, 1)
    elif direction == "z":
        left = line_fractions[0:1, :, :-1, :-1]
        bottom = line_fractions[1:2, :, :-1, :-1]
        right = line_fractions[0:1, :, :-1, 1:]
        up = line_fractions[1:2, :, 1:, :-1]
        phi_00 = phi[:, :, :-1, :-1]
        phi_01 = phi[:, :, 1:, :-1]
        phi_11 = phi[:, :, 1:, 1:]
        pad_to_mac = (0, 1, 0, 1, 0, 0)
    face_fraction = torch.where(
        phi_01 < 1e-7,
        0.5 * ((left + right) * bottom + (right - left) * up),
        torch.zeros(phi_01.size(), device=phi.device, dtype=left.dtype),
    )
    face_fraction = torch.where(
        (phi_01 >= 1e-7) * (phi_11 < 1e-7),
        0.5 * ((left + right) * bottom + (left - right) * up),
        face_fraction,
    )
    face_fraction = torch.where(
        (phi_01 >= 1e-7) * (phi_11 >= 1e-7) * (phi_00 < 1e-7),
        0.5 * ((left + right) * up + (right - left) * bottom),
        face_fraction,
    )
    face_fraction = torch.where(
        (phi_01 >= 1e-7) * (phi_11 >= 1e-7) * (phi_00 >= 1e-7),
        0.5 * ((left + right) * up + (left - right) * bottom),
        face_fraction,
    )
    face_fraction = F.pad(face_fraction, pad=pad_to_mac, value = 1)
    return face_fraction

def get_fractions(grid):
    if _is_2d(grid):
        return _get_line_fractions(grid, direction="z")
    else:
        line_fractions_x = _get_line_fractions(grid, direction="x")
        line_fractions_y = _get_line_fractions(grid, direction="y")
        line_fractions_z = _get_line_fractions(grid, direction="z")
        face_fraction_x = _get_face_fractions(
            line_fractions_x, grid, direction="x"
        )
        face_fraction_y = _get_face_fractions(
            line_fractions_y, grid, direction="y"
        )
        face_fraction_z = _get_face_fractions(
            line_fractions_z, grid, direction="z"
        )
        face_fraction = torch.cat(
            (face_fraction_x, face_fraction_y, face_fraction_z), dim=0
        )
        return face_fraction
    

def is_obs(fractions):
    if _is_2d(fractions):
        is_obs = (
            (fractions[0:1, :-1, :-1] < 1e-6)
            * (fractions[0:1, :-1, 1:] < 1e-6)
            * (fractions[1:2, :-1, :-1] < 1e-6)
            * (fractions[1:2, 1:, :-1] < 1e-6)
        )
    else:
        is_obs = (
            (fractions[0:1, :-1, :-1, :-1] < 1e-6)
            * (fractions[0:1, :-1, :-1, 1:] < 1e-6)
            * (fractions[1:2, :-1, :-1, :-1] < 1e-6)
            * (fractions[1:2, :-1, 1:, :-1] < 1e-6)
            * (fractions[1:2, :-1, :-1, :-1] < 1e-6)
            * (fractions[1:2, 1:, :-1, :-1] < 1e-6)
        )
    return is_obs