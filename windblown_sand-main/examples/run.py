import numpy as np
import torch
import imageio
import argparse
import simu

def main(
    res_z: int = 64,
    total_steps: int = 100,
    save_every: int = 10,
    dt: float = 1.0,
    nb_data_folder: str = "demo_data",
    ground_name: str = "climbing_dune",
    initial_sand_heights: str = "uniform_ones_128",
    start_erosion_frame: int = 100,
    inflow_sand_height: float = 0.0,
    recover: bool = False,
    mask_sand_above_buildings: bool = False,
    speed_limit: float = 0.8,
    inflow_angle: float = 0.0,
    inflow_speed: float = 1.0,
    nu: float = 1e-4,
    dt_salt: float = 1.0,
    domain_size: int = 10

):

    simu.simulation(
        res_z,
        total_steps,
        save_every,
        dt,
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

    )


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=1000, suppress=False)
    torch.set_printoptions(precision=3, linewidth=1000, profile="full")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=50,
        help="For how many step to run the simulation",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="How many step are left between succesive saves",
    )

    parser.add_argument(
        "--res_z",
        type=int,
        default=64,
        help="Simulation height",
    )

    parser.add_argument(
        "--nb_data_folder",
        type=str,
        default="0",
        help="Name of the folder in which the data of each iteration will be stored",
    )

    parser.add_argument(
        "--ground_name",
        type=str,
        default="climbing_dune",
        help="Name of the ground heightfield name in ./ground_heights/ folder",
    )

    parser.add_argument(
        "--initial_sand_heights",
        type=str,
        default="uniform_ones_128",
        help="Name of the initial sand heights name in ./standard_sand_heights/ folder",
    )

    parser.add_argument(
        "--start_erosion_frame",
        type=int,
        default=100,
        help="Frame at which start erosion process",
    )

    parser.add_argument(
        "--inflow_sand_height",
        type=float,
        default=0.0,
        help="sand height to add as inflow at every step",
    )

    parser.add_argument(
        "--speed_limit",
        type=float,
        default=0.8,
        help="Speed limit at which saltation starts",
    )

    parser.add_argument(
        "--recover",
        type=bool,
        default=False,
        help="Flag to recover simulation which was stopped in the middle",
    )

    parser.add_argument(
        "--mask_sand_above_buildings",
        type=bool,
        default=False,
        help="Flag to determine if mask sand height above buildings",
    )

    parser.add_argument(
        "--inflow_angle",
        type=float,
        default=0.0,
        help="Angle of the inflow wind flow",
    )

    parser.add_argument(
        "--inflow_speed",
        type=float,
        default=1.0,
        help="Speed of the inflow wind flow",
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=1e-4,
        help="Viscosity",
    )

    parser.add_argument(
        "--dt_salt",
        type=float,
        default=1.0,
        help="dt for saltation",
    )

    parser.add_argument(
        "--domain_size",
        type=int,
        default=10,
        help="Real world domain size",
    )



    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
