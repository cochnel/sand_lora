import torch
import torch.nn.functional as F


def set_dirichlet_bounds(bound_types, bound_values, bounds, value = None):
    bound_types.update({
        k:'dirichlet' for k in bounds
    })
    if value is not None:
        bound_values.update({
            k:value for k in bounds
        })
    return bound_types, bound_values


def check_bounds(bound_types, bound_values = None):

    """
    Check the validity of the boundary conditions and update them if necessary. All bounds not given will be set to Neumann
    Bound types are in the form {'x-' : 'dirichlet', 'x+': 'neumann', 'y+': 'circular', ...}
    """

    if bound_values is None:
        bound_values = {}

    # bound validity
    for k in bound_types:
        if k not in ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']:
            raise ValueError(f"Bound unknown: '{k}'")
        
    # bound type validity
    for v in bound_types.values():
        if v not in ['circular', 'dirichlet', 'neumann']:
            raise ValueError(f"Bound type unknown: '{v}'")
    
    # check and set circular bounds
    for t in [['x-', 'x+'], ['x+', 'x-'], ['y-', 'y+'], ['y+', 'y-'], ['z-', 'z+'], ['z+', 'z-']]:
        if t[0] in bound_types and bound_types[t[0]] == 'circular':
            assert t[1] not in bound_types or bound_types[t[1]] == 'circular', f"Incompatible bound types: {t[0]}' is circular while '{t[1]}' is {bound_types[t[1]]}"
            bound_types[t[1]] == 'circular'

    # only bound values for dirichlet
    for k in bound_values:
        if bound_types[k] != 'dirichlet':
            raise ValueError("setting 'bound_values' is only supported for dirichlet boundaries")
    
    # set 0 dirichlet bounds if not provided
    for k in bound_types:
        if bound_types[k] == 'dirichlet' and k not in bound_values:
            bound_values[k] = 0

    # set all unset bounds to neumann
    for k in ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']:
        if k not in bound_types:
            bound_types[k] = 'neumann'

    return bound_types, bound_values
    


def pad_boundary_conditions_mac(vel, bound_types, bound_values):

    """
    In MAC grid the last row is often a paceholder, we need to move the padding one row back 
    """

    # pad the grid
    vel = F.pad(vel, (1, 1,) * (len(vel.shape)-1))

    for k in ['x-', 'x+', 'y-', 'y+'] + (['z-', 'z+'] if len(vel.shape) == 4 else []):
        if bound_types[k] == 'dirichlet':
            vel[_bound_slice('x', k)]     = bound_values[k][0:1]
            vel[_bound_slice('y', k)]     = bound_values[k][1:2]
            if vel.shape[0] == 3: #3D
                vel[_bound_slice('z', k)] = bound_values[k][2:3]

        if bound_types[k] == 'neumann':
            vel[_bound_slice('x', k)]     = vel[_neumann_copy('x', k)]
            vel[_bound_slice('y', k)]     = vel[_neumann_copy('y', k)]
            if vel.shape[0] == 3: #3D
                vel[_bound_slice('z', k)] = vel[_neumann_copy('z', k)]

        if bound_types[k] == 'circular':
            vel[_circular_to('x', k)]     = vel[_circular_from('x', k)]
            vel[_circular_to('y', k)]     = vel[_circular_from('y', k)]
            if vel.shape[0] == 3: #3D
                vel[_circular_to('z', k)] = vel[_circular_from('z', k)]

    return vel


def _make_slice(b, k0, s):
    return (
        {'x':slice(0,1), 'y':slice(1,2), 'z':slice(2,3)}[b],
        ..., s,
        *((slice(None),) * {'x':0, 'y':1, 'z':2}[k0])
    )

def _bound_slice(b, k):
    return _make_slice(b, k[0], slice(-2, None) if k[1] == '+' # we do'nt care about '-1' for the orhtogonal directions
                          else (
                            slice(0, 2) if k[0] == b else slice(0, 1)
                          )
                       )

def _neumann_copy(b, k):
     return _make_slice(b, k[0], slice(-3, -2) if k[1] == '+'
                          else (
                            slice(2, 3) if k[0] == b else slice(1, 2)
                          )
                       )

def _circular_to(b, k):
    return _make_slice(b, k[0], slice(0, 1) if k[1] == '-' 
                       else slice(-2, None)) # we copy the bounday edge only from - to + (and orth in the padding as a side effect)

def _circular_from(b, k):
    return _make_slice(b, k[0], slice(-3, -2) if k[1] == '-' 
                       else slice(1, 3)) # we copy the bounday edge only from - to + (and orth in the padding as a side effect)
    


def pressure_boundaries(fractions, bound_types, bound_values):

    pressure_fractions = fractions.clone()

    for k in ['x-', 'x+', 'y-', 'y+'] + (['z-', 'z+'] if fractions.shape[0] == 3 else []):
        if bound_types[k] == 'dirichlet':
            s =_make_slice(k[0], k[0], slice(-1, None) if k[1] == '+' else slice(0,1))
            pressure_fractions [s] = 0.0

    return pressure_fractions, fractions