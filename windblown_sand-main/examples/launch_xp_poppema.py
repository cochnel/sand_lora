import sys
import os

res = 256
total_steps = 2000
# grounds = ["poppema_128_g_0_33" , "poppema_256_g_0_5" , "poppema_256_g_0_67" , "poppema_256_g_0_75"]
grounds = ["poppema_"+str(res)+"_g_0_33" , "poppema_"+str(res)+"_g_0_5" , "poppema_"+str(res)+"_g_0_67" ,  "poppema_"+str(res)+"_g_0_75"]

if res==128:
    # folders = [330 , 500 , 670 , 750]
    init_s_h = "two_128"
    inflow_s_h = "2.0"
    dt_salt = 1.0

elif res==256:
    folders = [330 , 500 , 670 , 750]
    folders = [331 , 501 , 671 , 751]
    init_s_h = "four_256"
    inflow_s_h = "4.0"
    dt_salt = 1.0

nb = 0
command = "python run.py --total_steps " + str(total_steps) + " --ground_name " + grounds[nb] + " --initial_sand_heights " + init_s_h + " --start_erosion_frame 100 --mask_initial_sand True --nb_data_folder " + str(folders[nb]) + " --res_z 30 --inflow_sand_height " + inflow_s_h + " --inflow_speed 1.0 --domain_size 14 --dt_salt " + str(dt_salt)
os.system(command)

nb = 1
command = "python run.py --total_steps " + str(total_steps) + " --ground_name " + grounds[nb] + " --initial_sand_heights " + init_s_h + " --start_erosion_frame 100 --mask_initial_sand True --nb_data_folder " + str(folders[nb]) + " --res_z 30 --inflow_sand_height " + inflow_s_h + " --inflow_speed 1.0 --domain_size 14 --dt_salt " + str(dt_salt)
os.system(command)

nb = 2
command = "python run.py --total_steps " + str(total_steps) + " --ground_name " + grounds[nb] + " --initial_sand_heights " + init_s_h + " --start_erosion_frame 100 --mask_initial_sand True --nb_data_folder " + str(folders[nb]) + " --res_z 30 --inflow_sand_height " + inflow_s_h + " --inflow_speed 1.0 --domain_size 14 --dt_salt " + str(dt_salt)
os.system(command)

nb = 3
command = "python run.py --total_steps " + str(total_steps) + " --ground_name " + grounds[nb] + " --initial_sand_heights " + init_s_h + " --start_erosion_frame 100 --mask_initial_sand True --nb_data_folder " + str(folders[nb]) + " --res_z 30 --inflow_sand_height " + inflow_s_h + " --inflow_speed 1.0 --domain_size 14 --dt_salt " + str(dt_salt)
os.system(command)
