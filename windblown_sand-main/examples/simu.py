import math
import torch
import torch.nn.functional as F

import sys
sys.path.append('../')

from fluids import advection, pressure, bounds, level_grid, timestep, diffusion
from physics_functions import erosion_deposition_functions, simu_utils, couple_simulation, height_bounds

import matplotlib.pyplot as plt


def simulation(
    res_z,
    total_steps,
    save_every,
    max_dt,
    nb_data_folder,
    ground_name,
    initial_sand_heights,
    start_erosion_frame,
    inflow_sand_height,
    recover,
    mask_sand_above_buildings,
    speed_limit,
    inflow_angle,
    inflow_speed,
    nu,
    dt_salt,
    domain_size

):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load and intialize fields
    time_handler = timestep.Time(max_dt)

    ground_heights, sand_heights, vel, start_step = simu_utils.load_data(device, torch.float32, res_z, nb_data_folder, ground_name, initial_sand_heights, recover , mask_sand_above_buildings)
    see_heights, wind2D = None, None
    current_sdf = couple_simulation.update_sdf(ground_heights , sand_heights, res_z)

    res_x = ground_heights.shape[0]

    v_xp = 5.0
    nu = 15*1e-6*res_x/(v_xp*domain_size)
    dt_salt *= (res_x/domain_size)**2
    m = domain_size/res_x

    print("dt_salt: "+ str(dt_salt))
    print("nu: "+ str(nu))
    print("m: "+ str(m))

    angle = math.radians(inflow_angle)
    inflow_vel = torch.tensor([inflow_speed*math.cos(angle), inflow_speed*math.sin(angle) , 0.0], device = vel.device)[:, None, None]

    boundaries_conditions = bounds.check_bounds(*bounds.set_dirichlet_bounds({}, {}, ['x-', 'y-', 'y+', 'z+'], inflow_vel))
    hm_boundaries_conditions = height_bounds.create_from_fluid_bounds(boundaries_conditions[0], {'x-': inflow_sand_height, 'y-': inflow_sand_height, 'y+': inflow_sand_height})
    # hm_boundaries_conditions = height_bounds.create_from_fluid_bounds(boundaries_conditions[0], {'x-': inflow_sand_height})
    ground_heights = height_bounds.pad_bounds_ground(ground_heights, hm_boundaries_conditions[0])

    for step in range(start_step, total_steps):

        print("demo: simulate at frame {}.".format(step))

        fractions = level_grid.get_fractions(current_sdf[None])
        fractions = torch.where(fractions > 1e-7, fractions, torch.zeros_like(fractions))

        vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)
        vel = advection.advect_mac_semi_lagrange(vel, vel, time_handler.set_dt(vel))[..., 1:-1, 1:-1, 1:-1]
        vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))
        vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)


        vel = diffusion.diffuse(vel, 1+0*F.pad(fractions, (1,1,1,1,1,1), value = 1), dx=1, dt=time_handler.dt, nu=nu)[..., 1:-1, 1:-1, 1:-1]
        vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, 1:-1, 1:-1]
        vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))

        pressure_boundaries = bounds.pressure_boundaries(fractions.clone(), *boundaries_conditions)
        vel, p = pressure.solve_pressure(*pressure_boundaries, vel, cg_accu=1e-3)
        vel = torch.where(fractions > 1e-7, vel, torch.zeros_like(vel))

        vel = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, 1:-1, 1:-1]

        """ COMPUTE EROSION DEPOSITION """
        if step>=start_erosion_frame:

            sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)

            vel3D = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, :, :] # padding in the z direction is not necessary
            wind2D = bounds.pad_boundary_conditions_mac(
                erosion_deposition_functions.project_wind_on_ground(vel3D, ground_heights + sand_heights),
                *boundaries_conditions
            )

            sand_heights, see_heights = erosion_deposition_functions.erosion_deposition(sand_heights, wind2D, speed_limit, dt_salt, v_xp, m)
            sand_heights = height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)

            # plt.imshow(sand_heights.detach().cpu())
            # plt.colorbar()
            # plt.show()

            """ AVALANCHE """
            # sand_heights = erosion_deposition_functions.avalanche_guillaume(current_sdf , sand_heights, nb_iterations=30 , dt=.1 , angle=.5)
            sand_heights = erosion_deposition_functions.avalanche_nicolas(ground_heights, sand_heights, nb_iterations=15 , dt=.5 , angle=35)

            current_sdf = couple_simulation.update_sdf(ground_heights, sand_heights, res_z)[..., :, 1:-1, 1:-1]
            # current_sdf = couple_simulation.update_sdf_upwind(ground_heights, sand_heights, res_z, wind2D)[..., :, 1:-1, 1:-1]


            sand_heights = sand_heights[..., 1:-1, 1:-1]

            print('Sand mass : '+'{:.2f}'.format(torch.sum(sand_heights).item()))

        else:
            vel3D = bounds.pad_boundary_conditions_mac(vel, *boundaries_conditions)[..., 1:-1, :, :] # padding in the z direction is not necessary
            wind2D = bounds.pad_boundary_conditions_mac(
                erosion_deposition_functions.project_wind_on_ground(vel3D, ground_heights +
                                                                    height_bounds.pad_bounds_sand(sand_heights, *hm_boundaries_conditions)
                                                                    ),
                *boundaries_conditions
            )


        if step % save_every == 0:
            simu_utils.save_step_files(ground_name, step, start_erosion_frame, nb_data_folder,
                                       vel,
                                       ground_heights[..., 1:-1, 1:-1],
                                       sand_heights , see_heights , wind2D[..., 1:-1, 1:-1])

        time_handler.step()
