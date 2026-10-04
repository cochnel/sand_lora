import torch.nn.functional as F

def create_from_fluid_bounds(bound_types, hm_bound_values):

    bound_types = {k:v for k, v in bound_types.items() if k in ['x-', 'x+', 'y-', 'y+']}

    for k in hm_bound_values:
        if bound_types[k] != 'dirichlet':
            raise ValueError("setting 'hm_bound_values' is only supported for dirichlet boundaries")

    for k in bound_types:
        if bound_types[k] == 'dirichlet' and k not in hm_bound_values:
            hm_bound_values[k] = 0

    return bound_types, hm_bound_values


def pad_bounds_ground(ground_height, bound_types):

    ground_height = F.pad(ground_height[None, None], (1,1,1,1), mode = 'replicate')[0, 0]

    for k in ['x-', 'x+', 'y-', 'y+']:
        if bound_types[k] == 'circular':
            if k == 'x-':
                ground_height[..., 0:1] = ground_height[..., -2:-1]
            if k == 'x+':
                ground_height[..., -1:] = ground_height[..., 1:2]
            if k == 'y-':
                ground_height[..., 0:1, :] = ground_height[..., -2:-1, :]
            if k == 'y+':
                ground_height[..., -1:, :] = ground_height[..., 1:2, :]

    return ground_height


def pad_bounds_sand(sand_height, bound_types, hm_bound_values):

    sand_height = pad_bounds_ground(sand_height, bound_types)

    for k in ['x-', 'x+', 'y-', 'y+']:
        if bound_types[k] == 'dirichlet':
            if k == 'x-':
                sand_height[..., 0:2] = hm_bound_values[k]
            if k == 'x+':
                sand_height[..., -2:] = hm_bound_values[k]
            if k == 'y-':
                sand_height[..., 0:2, :] = hm_bound_values[k]
            if k == 'y+':
                sand_height[..., -2:, :] = hm_bound_values[k]

    return sand_height
