import torch

import os
import numpy as np
import sys

from fluids import utils
import argparse
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from physics_functions import erosion_deposition_functions,  simu_utils, couple_simulation


def getfiles(dirpath):
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

def get_last_step_of_sim(dirpath):
    a = getfiles(dirpath)
    num = a[-1].split("sand_heights_")[1][:-4]
    return int(num)

def save_step_files(ground_name,  step , start_erosion_frame , nb_data_folder , vel , ground_heights, sand_heights , see_heights , wind2D):

    fname = "../out/demo_data_sdf_"+nb_data_folder+"/{:03}.png".format(step)
    fname_to_save = "../out/demo_data_sdf_"+nb_data_folder+"/{:03}.png".format(step)
    fname_to_save_proj = "../out/demo_data_sdf_"+nb_data_folder+"/z_proj_{:03}.png".format(step)
    np.save("../out/vels_"+nb_data_folder+"/vels_"+str(step) , vel.detach().cpu())
    if step>start_erosion_frame:
        np.save("../out/sand_heights_"+nb_data_folder+"_"+ground_name+"/sand_heights_"+str(step) , sand_heights.detach().cpu().numpy())
        np.save("../out/see_heights_"+nb_data_folder+"_"+ground_name+"/see_heights_"+str(step) , see_heights.detach().cpu())
        np.save("../out/wind_2D_"+nb_data_folder+"_"+ground_name+"/wind_2D_"+str(step) , wind2D.detach().cpu())

    # utils.save_3D_img_slices(vel, fname_to_save)
    utils.save_visu(vel, ground_heights, sand_heights, wind2D, fname_to_save)


def load_data(device, dtype, res_z,
              nb_data_folder: str = "demo_data",
              ground_name: str = "climbing_dune",
              initial_sand_heights: str = "uniform_ones_128",
              recover: bool = False,
              mask_initial_sand: bool = False):

    ground_heights = torch.tensor(np.load("../ground_heights/" + ground_name + ".npy"), dtype = dtype, device = device)

    if recover:
        sand_heights_path = "../out/sand_heights_"+nb_data_folder+"_"+ground_name
        start_step = simu_utils.get_last_step_of_sim(sand_heights_path)
        sand_heights = torch.tensor(np.load(sand_heights_path + "/sand_heights_" + str(start_step) + ".npy"), dtype = dtype, device = device)
        vel = torch.tensor(np.load("../out/vels_"+nb_data_folder + "/vels_" + str(start_step) + ".npy"), dtype = dtype, device = device)

    else:
        start_step = 0
        sand_heights = torch.tensor(np.load("../standard_sand_heights/" + initial_sand_heights + ".npy"), dtype = dtype, device = device)
        if mask_initial_sand:
            mask = (ground_heights>0).long()
            sand_heights *= (1-mask)
        vel = torch.zeros(torch.Size((3, res_z+1, ground_heights.shape[0]+1, ground_heights.shape[1]+1)), dtype = dtype, device = device)

    # CREATE DIRS FOR STORING FILES
    utils.mkdir("../out/demo_data_sdf_"+nb_data_folder)
    utils.mkdir("../out/vels_"+nb_data_folder)
    utils.mkdir("../out/sand_heights_"+nb_data_folder+"_"+ground_name)
    utils.mkdir("../out/wind_2D_"+nb_data_folder+"_"+ground_name)
    utils.mkdir("../out/see_heights_"+nb_data_folder+"_"+ground_name)

    return ground_heights, sand_heights, vel, start_step
