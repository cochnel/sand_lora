simulations can be run in examples/ by running the run.py file. For example:

python run.py --ground_name climbing_dune --initial_snow_heights zeros_128 --total_steps 800 --nb_data_folder 6

python run.py --total_steps 1000 --ground_name poppema_128 --initial_snow_heights ones_128 --inflow_snow_height 1.0  --nb_data_folder 60 --mask_initial_sand True

you can change the file defining the simulation and import it there.
