import numpy as np
import torch
import torch.nn.functional as F

def get_first_non_zero_idxs(v_x , v_y , v_z , eps):

    # mask = (v_x==0.0).numpy()
    mask = (np.abs(v_x)<eps).numpy()
    sum = np.sum(mask , axis=0)
    last_zeros_idxs_x = sum-2

    # mask = (v_y==0.0).numpy()
    mask = (np.abs(v_y)<eps).numpy()
    sum = np.sum(mask , axis=0)
    last_zeros_idxs_y = sum-2

    # mask = (v_z==0.0).numpy()
    mask = (np.abs(v_z)<eps).numpy()
    sum = np.sum(mask , axis=0)
    last_zeros_idxs_z = sum-2

    return last_zeros_idxs_x , last_zeros_idxs_y , last_zeros_idxs_z


def fill_zeros_in_vel(v_x , v_y , v_z , vx_c , vy_c , vz_c , eps):

    print("Filing zeros ...")

    l_i_x , l_i_y , l_i_z = get_first_non_zero_idxs(v_x , v_y , v_z , eps)

    iss, jss = np.nonzero((l_i_x-vx_c)<0)

    for k in range(len(iss)):
        i , j = iss[k] , jss[k]
        last_idx = l_i_x[i,j]
        if np.abs(v_x[last_idx+1, i , j])<eps and last_idx+2<128:
            # print("a")
            v_x[last_idx+1, i , j] = v_x[last_idx+2, i , j]

    iss, jss = np.nonzero((l_i_y-vy_c)<0)

    for k in range(len(iss)):
        i , j = iss[k] , jss[k]
        last_idx = l_i_y[i,j]
        if np.abs(v_y[last_idx+1, i , j])<eps and last_idx+2<128:
            # print("b")
            v_y[last_idx+1, i , j] = v_y[last_idx+2, i , j]

    iss, jss = np.nonzero((l_i_z-vz_c)<0)
    for k in range(len(iss)):
        i , j = iss[k] , jss[k]
        last_idx = l_i_z[i,j]
        if np.abs(v_z[last_idx+1, i , j])<eps and last_idx+2<128:
            # print("c")
            v_z[last_idx+1, i , j] = v_z[last_idx+2, i , j]
    vel = torch.stack((v_x,v_y,v_z) , axis=0).cuda()

    return vel

def get_upwind_coefs(vel):

    eps_comp = 1e-4
    vx , vy = vel[0] , vel[1]

    if np.abs(vx)>np.abs(vy):
        h = np.abs(vy)/(np.abs(vx)+eps_comp)
    else:
        h = np.abs(vx)/(np.abs(vy)+eps_comp)
    if vx>0 and vy>0:
        a , b , c , d = h/2 , 0 , 1/2-h/2 , 1/2
    if vx>0 and vy<0:
        a , b , c , d = 1/2 , h/2 , 0 , 1/2-h/2
    if vx<0 and vy>0:
        a , b , c , d = 0 , 1/2-h/2 , 1/2 , h/2
    else:
        a , b , c , d = h/2 , 1/2 , 1/2-h/2 , 0

    return a , b , c , d


def update_sdf(ground_heights, sand_heights, res_z):

    total_heights = ground_heights+sand_heights

    flat = .25*(total_heights[:-1,1:] + total_heights[1:,1:] + total_heights[1:,:-1] + total_heights[:-1,:-1])
    sdf = -flat[None] + torch.linspace(0, res_z , res_z+1, device = ground_heights.device, dtype = ground_heights.dtype)[:, None, None]
    sdf = F.pad(sdf[None] , (1, 1, 1, 1, 0, 0) , mode = "replicate")[0]

    return sdf


def update_sdf_upwind(ground_heights, sand_heights, res_z, wind2d):

    total_heights = ground_heights+sand_heights

    u, v = .5*(wind2d[0, 1:-1, 1:-1] + wind2d[0, :-2, 1:-1]), .5*(wind2d[1, 1:-1, 1:-1] + wind2d[1, 1:-1, :-2])
    u, v = .5+.5*u/(u.abs() + v.abs()), .5+.5*v/(u.abs() + v.abs())
    flat =  (u    * (v * total_heights[:-1,:-1] + (1-v) * total_heights[1:,:-1])+
            (1-u) * (v * total_heights[:-1, 1:] + (1-v) * total_heights[:-1,:-1]))

    sdf = -flat[None] + torch.linspace(0, res_z , res_z+1, device = ground_heights.device, dtype = ground_heights.dtype)[:, None, None]
    sdf = F.pad(sdf[None] , (1, 1, 1, 1, 0, 0) , mode = "replicate")[0]

    return sdf
