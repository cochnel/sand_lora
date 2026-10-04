import torch
import torch.nn.functional as F

def laplace_operator_3d(input_tensor):

    laplace_kernel = torch.tensor([[[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]],

                                   [[0, 1, 0],
                                    [1, -6, 1],
                                    [0, 1, 0]],

                                   [[0, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0]]], dtype=input_tensor.dtype, device = input_tensor.device).view(1, 1, 3, 3, 3)
    
    laplace_kernel = (laplace_kernel if input_tensor.shape[1] == 1 else laplace_kernel.repeat(input_tensor.shape[1], 1, 1, 1, 1))


    return F.conv3d(input_tensor, laplace_kernel, padding=1, groups=input_tensor.shape[1])


def laplace_operator_2d(input_tensor):

    laplace_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=input_tensor.dtype, device = input_tensor.device).view(1, 1, 3, 3)

    laplace_kernel = (laplace_kernel if input_tensor.shape[1] == 1 else laplace_kernel.repeat(input_tensor.shape[1], 1, 1, 1))

    return F.conv2d(input_tensor, laplace_kernel, padding=1, groups=input_tensor.shape[1])


def diffuse(input_tensor, fractions, dx, dt, nu):

    if len(input_tensor.shape) == 3: #2d
        laplace_tensor = laplace_operator_2d((input_tensor*fractions)[None])[0]/(dx**2)/(fractions+1e-3)
    else:
        laplace_tensor = laplace_operator_3d((input_tensor*fractions)[None])[0]/(dx**2)/(fractions+1e-3)

    return input_tensor + dt*nu*laplace_tensor
