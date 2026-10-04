import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import sys
from skimage.transform import resize
import torch
import torch.nn.functional as F

# for AffineGridGenerator
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

from fluids import mac_grid

def mkdir(path):
    def mkdirs(paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                mkdir(path)
        else:
            mkdir(paths)

    if not os.path.exists(path):
        os.makedirs(path)


def save_npy(array, filename, add_redundant_channel=False):
    if not filename.endswith(".npy"):
        raise ValueError("filename should end with .npy.")
    if len(array.shape) == 3 and array.shape[2] == 2 and add_redundant_channel:
        array = np.concatenate(
            (array, np.zeros((array.shape[0], array.shape[1], 1))), axis=2
        )
    np.save(filename, array)


def load_npy(filename, remove_redundant_channel=True):
    """Loads .npy file into a np.array.

    The .npy file comes from mantaflow's saved arrays. It can be density
    or velocity in 2D or 3D.

    Args:
        filename(string): filename of the array, should end with '.npy'.
        remove_redundant_channel: since the npy files are saved in accordance
            with mantaflow, the 2D velocities have a third channel which is
            redundant. The flag is to specify whether this channel should
            be removed when loading the npy file.

    Returns:
        torch.Tensor of size [C, (D), H, W]

    """
    if not filename.endswith(".npy"):
        raise ValueError("filename should end with .npy.")
    array = np.load(filename)
    dim = 2 if len(array.shape) == 3 else 3
    # remove dummy channel in 2D velocity
    if dim == 2 and array.shape[2] == 3 and remove_redundant_channel:
        array = array[:, :, :-1]
    return array


def load_npz(filename, key):
    if not filename.endswith(".npz"):
        raise ValueError("filename should end with .npz.")
    data = np.load(filename)
    if key not in data.files:
        raise ValueError("key {} is not in npz file".format(key))
    return data[key]


def show_img(array, colormap="RGB"):
    if len(array.shape) == 3:
        if colormap == "ARROW" or colormap == "LIC":
            _vector_plot(array, save=False, plot_type=colormap)
        elif colormap == "RDBU_BAR":
            _rdbu_bar_plot(array, save=False)
        else:
            array = array[::-1]
            array = _process_img_2d(array, colormap)
            plt.figure(figsize=(10, 10))
            plt.imshow(array)
            plt.xticks([])
            plt.yticks([])
            plt.show()
    else:
        raise NotImplementedError("utils.show_img: 3D case not supported.")

def _get_filename_on_axis(filename, axis):
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    basename_axis = axis + "_" + basename
    return os.path.join(dirname, basename_axis)

def save_3D_img_slices(tensor, filename):

    tensor = (tensor.moveaxis(0, -1) / tensor.abs().max()*.5+.5)*255
    # tensor = (tensor.moveaxis(0, -1)*.5+.5)*255

    xy = tensor[tensor.shape[0]//2].cpu().numpy()
    xz = tensor[:, tensor.shape[1]//2].cpu().numpy()
    yz = tensor[:, :, tensor.shape[2]//2].cpu().numpy()



    # plt.imshow(xz[..., 0])
    # plt.colorbar()
    # plt.show()

    Image.fromarray(np.uint8(xy[::-1])).save(_get_filename_on_axis(filename, "xy"))
    Image.fromarray(np.uint8(xz[::-1])).save(_get_filename_on_axis(filename, "xz"))
    Image.fromarray(np.uint8(yz[::-1])).save(_get_filename_on_axis(filename, "yz"))


def save_visu(vel, ground_heights, snow_heights, wind2D, filename):

    vel_norm = mac_grid.fast_norm(vel)
    surface = ground_heights + snow_heights

    xy = vel_norm[vel_norm.shape[0]//2].cpu().numpy()
    xz = vel_norm[:, vel_norm.shape[1]//2].cpu().numpy()
    yz = vel_norm[:, :, vel_norm.shape[2]//2].cpu().numpy()

    cmap = "plasma"

    plt.figure()
    plt.imshow(xy , cmap=cmap, origin="lower")
    plt.colorbar()
    plt.savefig(_get_filename_on_axis(filename, "xy"))
    plt.close()

    # print(xz.shape, surface.shape)
    plt.figure()
    plt.imshow(xz , vmin = 0, cmap=cmap, origin="lower")

    plt.plot(ground_heights[surface.shape[0]//2].cpu().numpy()-.5, linewidth = .5, color = 'white',zorder=1)
    plt.plot(surface[surface.shape[0]//2].cpu().numpy() -.5, linewidth = .5, color = 'grey',zorder=1)

    plt.scatter(np.arange(surface.shape[1]),
                surface[surface.shape[0]//2].cpu().numpy()-.5,
                c = mac_grid.fast_norm(wind2D)[surface.shape[0]//2].cpu().numpy(),
                cmap = cmap, vmin = 0, vmax = xz.max(), s = .5, linewidth = 0, zorder=2)

    #

    plt.colorbar(shrink = xz.shape[0]/xz.shape[1])
    plt.savefig(_get_filename_on_axis(filename, "xz"), bbox_inches='tight', dpi = 600)
    plt.close()

    plt.figure()
    plt.imshow(yz , cmap=cmap, origin="lower")
    plt.colorbar()
    plt.savefig(_get_filename_on_axis(filename, "yz"))
    plt.close()


def save_img(array, filename, colormap="RGB", project=True, transmit=0.05):

    """Saves an array into an image.

    The array can be 2D/3D density or 2D/3D velocity field.

    Args:
        array(np.array): of size [(D), H, W, C], input array to be saved.
            Density array should be in range [0, 1].
            Velocity array is in range (-inf, inf)
        filename(string): filename of the ouput file, should end with .png.
    """
    if not filename.endswith(".png"):
        raise ValueError("filename should end with .png")
    vector_plot_colormaps = ["ARROW", "ARROW_COLOR", "LIC", "LIC_COLOR", "STREAM", "STREAM_COLOR"]
    # 2d case: directly save as image
    if len(array.shape) == 3:
        if colormap in vector_plot_colormaps:
            _vector_plot(array, filename, save=True, plot_type=colormap)
        elif colormap == "RDBU_BAR":
            array = array[::-1]  # flip the array upside down
            _rdbu_bar_plot(array, filename, save=True)
        else:
            array = array[::-1]  # flip the array upside down
            _save_img_2d(array, filename, colormap)
    # 3d case: save middle slice in all three axes as image
    elif len(array.shape) == 4:



        # for densities
        def _render_front_view(array, transmit):
            density = array[::-1]  # view density from front to back
            # plt.imshow(density)
            # plt.show()
            tau = np.exp(-transmit * np.cumsum(density, axis=0))
            rendered = np.sum(density * tau, axis=0)
            rendered = rendered / np.max(rendered)  # remap to [0, 1]
            return rendered[::-1]

        D, H, W, _ = array.shape
        filename_x_middle = _get_filename_on_axis(filename, "yz")
        filename_y_middle = _get_filename_on_axis(filename, "xz")
        filename_z_middle = _get_filename_on_axis(filename, "xy")
        # array_x_middle = np.transpose(array[:, :, int(W / 2), :], (1, 0, 2))[::-1]
        array_x_middle = np.flip(array[:, :, int(W / 2), :][::-1] , 0)

        array_y_middle = array[:, int(H / 2), :, :]
        array_z_middle = (array[int(D / 2), :, :, :])[::-1]


        if colormap in vector_plot_colormaps:
            _vector_plot(
                array_x_middle[::-1], filename_x_middle, save=True, plot_type=colormap
            )
            _vector_plot(
                array_y_middle, filename_y_middle, save=True, plot_type=colormap
            )
            _vector_plot(
                array_z_middle[::-1], filename_z_middle, save=True, plot_type=colormap
            )
        elif colormap == "RDBU_BAR":
            _rdbu_bar_plot(array_x_middle, filename_x_middle, save=True)
            _rdbu_bar_plot(array_y_middle, filename_y_middle, save=True)
            _rdbu_bar_plot(array_z_middle, filename_z_middle, save=True)
        else:
            if project:
                filename_z_proj = _get_filename_on_axis(filename, "z_proj")
                array_z_proj = _render_front_view(array, transmit)
                _save_img_2d(array_z_proj, filename_z_proj, colormap)
            else:
                norm = np.linalg.norm(array_x_middle, axis=2)
                plt.figure()
                plt.imshow(norm , cmap="seismic" , origin="lower")
                plt.savefig(filename_x_middle)
                plt.close()

                norm = np.linalg.norm(array_y_middle, axis=2)
                plt.figure()
                plt.imshow(norm , cmap="seismic", origin="lower")
                plt.savefig(filename_y_middle)
                plt.close()

                norm = np.linalg.norm(array_z_middle, axis=2)
                plt.figure()
                plt.imshow(norm , cmap="seismic", origin="lower")
                plt.savefig(filename_z_middle)
                plt.close()


    else:
        raise ValueError("input array should be in size [(D), H, W, C].")


def _save_img_2d(array, filename, colormap="RGB"):
    array = _process_img_2d(array, colormap)
    img = Image.fromarray(array)
    img.save(filename)


def _process_img_2d(array, colormap="RGB"):
    array = np.squeeze(array)
    # deal with velocity
    if len(array.shape) == 3:
        # add a zero channel to the 2d velocity
        if array.shape[2] == 2:
            array = np.concatenate(
                (array, np.expand_dims(np.zeros_like(array[..., 0]), axis=2)), axis=2
            )
    max_ = np.absolute(array.max())
    min_ = array.min() if array.min() >= 0 else -max_
    array = (array - min_) / (max_ - min_)  # [0, 1]

    norm = np.linalg.norm(array , axis=2)
    norm = np.uint8(norm)

    norm = np.stack((norm,norm,norm) , axis = 2)

    return norm

# def _process_img_2d(array, colormap="RGB"):
#     array = np.squeeze(array)
#     # deal with velocity
#     if len(array.shape) == 3:
#         # add a zero channel to the 2d velocity
#         if array.shape[2] == 2:
#             array = np.concatenate(
#                 (array, np.expand_dims(np.zeros_like(array[..., 0]), axis=2)), axis=2
#             )
#     max_ = np.absolute(array.max())
#     min_ = array.min() if array.min() >= 0 else -max_
#     array = (array - min_) / (max_ - min_)  # [0, 1]
#     if colormap == "RGB":
#         array = array * 255.0
#     elif colormap == "OrRd":
#         cmap = plt.cm.get_cmap("OrRd")
#         array = np.minimum(1, cmap(array) / cmap(0)) * 255.0
#     elif colormap in plt.colormaps():
#         cmap = plt.get_cmap(colormap)
#         array = cmap(array) * 255.0
#     else:
#         raise ValueError("Colormap {} not supported.".format(colormap))
#     array = np.uint8(array)
#     return array


def _vector_plot(
    array,
    filename=None,
    save=False,
    plot_type="ARROW",
    kernel_len=100,
    dpi=600,
    upsample=True,
    upsample_short_size=1200,
):
    """Args:
    array(np.array): 2D velocity field, in shape [H, W, C]
    save(bool): save=True means to save the plot with filename,
        save=False means to only show the plot
    kernel_len, upsample, upsample_short_size: parameters for LIC,
        do not need to change.
    """
    assert len(array.shape) == 3
    res_x, res_y, _ = array.shape
    X = np.arange(0, res_x, 1)
    Y = np.arange(0, res_y, 1)
    X, Y = np.meshgrid(X, Y)
    U = array[:, :, 0]
    V = array[:, :, 1]
    if plot_type == "ARROW":
        r = int(res_x / 32) if res_x >= 64 else 1
        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        plt.quiver(X[::r, ::r], Y[::r, ::r], U[::r, ::r], V[::r, ::r], width=0.001)
        plt.axis("off")
    elif plot_type == "ARROW_COLOR":
        r = int(res_x / 32) if res_x >= 64 else 1
        speed = np.sqrt(U ** 2 + V ** 2)
        plot = plt.quiver(
            X[::r, ::r],
            Y[::r, ::r],
            U[::r, ::r],
            V[::r, ::r],
            speed[::r, ::r],
            width=0.001,
            cmap="gist_heat",
        )
        plt.colorbar(plot, shrink=0.8)
        plt.tight_layout()
        plt.axis("off")
    elif plot_type == "STREAM":
        plt.streamplot(
            X, Y, U, V, density=2, linewidth=1, arrowsize=1, arrowstyle="->", color="k"
        )
        plt.tight_layout()
        plt.axis("off")
    elif plot_type == "STREAM_COLOR":
        speed = np.sqrt(U ** 2 + V ** 2)
        plot = plt.streamplot(
            X,
            Y,
            U,
            V,
            color=speed,
            cmap="gist_heat",
            density=2,
            linewidth=1,
            arrowsize=1,
            arrowstyle="->",
        )
        plt.colorbar(plot.lines, shrink=0.8)
        plt.tight_layout()
        plt.axis("off")
    elif plot_type == "LIC":
        img, _, _ = _licplot(upsample, upsample_short_size, res_x, res_y, U, V, kernel_len)
        plt.clf()
        plt.axis("off")
        plt.figimage(img[::-1], cmap="gray", vmin=np.min(img) / 3, vmax=np.max(img) / 3)
        plt.gcf().set_size_inches((res_x / float(dpi), res_y / float(dpi)))
        np.random.seed()
    elif plot_type == "LIC_COLOR":
        img, U, V = _licplot(upsample, upsample_short_size, res_x, res_y, U, V, kernel_len)
        colors = plt.cm.gist_heat(plt.Normalize()(np.sqrt(U*U+V*V)))
        img = plt.Normalize(np.min(img) / 3, np.max(img) / 3)(img)[:, :, None] * colors
        plt.clf()
        plt.axis("off")
        plt.figimage(img[::-1])
        plt.gcf().set_size_inches((res_x / float(dpi), res_y / float(dpi)))
        np.random.seed()
    else:
        raise ValueError("_vector_plot: plot_type not recognized.")
    if save:
        assert filename is not None
        plt.savefig(filename, dpi=dpi)
    else:
        plt.show()
    plt.close()

def _licplot(upsample, upsample_short_size, res_x, res_y, U, V, kernel_len):
    from licplot import lic_internal

    np.random.seed(4)
    if upsample:
        ratio = upsample_short_size / min(res_x, res_y)
        res_x = int(ratio * res_x)
        res_y = int(ratio * res_y)
        # assumes square resolution
        U = resize(U, output_shape=(res_x, res_y))
        V = resize(V, output_shape=(res_x, res_y))
    kernel = np.sin(np.arange(kernel_len) * np.pi / kernel_len)
    kernel = kernel.astype(np.float32)
    texture = np.random.randn(int(res_x / 4), int(res_y / 4)).astype(np.float32)
    texture = resize(texture, (res_x, res_y), order=1)
    return lic_internal.line_integral_convolution(U, V, texture, kernel), U, V



def _rdbu_bar_plot(array, filename=None, save=False, dpi=600):
    assert len(array.shape) == 3 and array.shape[2] == 1
    plt.figure()
    plot = plt.imshow(array[:, :, 0], cmap="RdBu")
    plt.axis("off")
    plt.colorbar(plot, shrink=0.8)
    if save:
        assert filename is not None
        plt.savefig(filename, dpi=dpi)
    else:
        plt.show()
    plt.close()


def create_mesh_grid(size, device="cuda", dtype=torch.float32):
    """Creates a mesh grid on device

    Args:
        size(list, torch.Tensor): in the order of [C, (D), H, W]

    Return:
        mgrid(torch.Tensor): mesh grid in range [-1, 1]
    """
    dim = 2 if len(size) == 3 else 3
    if dim == 2:
        H, W = size[1], size[2]
        y_pos, x_pos = torch.meshgrid(
            [
                torch.arange(0, H, device=device, dtype=dtype),
                torch.arange(0, W, device=device, dtype=dtype),
            ]
        )
        mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, H, W]
    else:
        D, H, W = size[1], size[2], size[3]
        z_pos, y_pos, x_pos = torch.meshgrid(
            [
                torch.arange(0, D, device=device, dtype=dtype),
                torch.arange(0, H, device=device, dtype=dtype),
                torch.arange(0, W, device=device, dtype=dtype),
            ]
        )
        mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)  # [C, D, H, W]
    return mgrid


def normalize_mesh_grid(mgrid):
    """Normalizes mesh grid to [-1, 1]

    Args:
        mgrid: mesh grid

    Return:
        mgrid_normed(torch.Tensor): mesh grid in range [-1, 1]
    """
    dim = 2 if len(mgrid.size()) == 3 else 3
    if dim == 2:
        H, W = mgrid.size(1), mgrid.size(2)
        mgrid_normed_x = 2.0 * mgrid[0:1, ...] / (W - 1.0) - 1.0
        mgrid_normed_y = 2.0 * mgrid[1:2, ...] / (H - 1.0) - 1.0
        mgrid_normed = torch.cat((mgrid_normed_x, mgrid_normed_y), dim=0)
    else:
        D, H, W = mgrid.size(1), mgrid.size(2), mgrid.size(3)
        mgrid_normed_x = 2.0 * mgrid[0:1, ...] / (W - 1.0) - 1.0
        mgrid_normed_y = 2.0 * mgrid[1:2, ...] / (H - 1.0) - 1.0
        mgrid_normed_z = 2.0 * mgrid[2:3, ...] / (D - 1.0) - 1.0
        mgrid_normed = torch.cat(
            (mgrid_normed_x, mgrid_normed_y, mgrid_normed_z), dim=0
        )
    return mgrid_normed


def grid_sample(grid, back_trace):
    """Wrapper for torch.nn.functional.grid_sample

    Args:
        grid(torch.Tensor): grid to be sampled on
        back_trace: relative position (w.r.t to current index) to sample.

    Return:
        grid_sampled(torch.Tensor): sampled grid
    """
    dim = 2 if len(grid.size()) == 3 else 3
    # normalize the field to [-1, 1]
    back_trace_normed = normalize_mesh_grid(back_trace)
    # reshape back_trace to fit the required shape
    permutation = (1, 2, 3, 0) if dim == 3 else (1, 2, 0)
    back_trace_normed = back_trace_normed.permute(permutation).unsqueeze(
        0
    )  # [N, (D), H, W, C]
    # reshape the input to fit the required shape
    grid = grid.unsqueeze(0)  # [N, C, (D), H, W]
    # go back in time and interpolate to get updated velocity
    grid_sampled = F.grid_sample(
        grid,
        back_trace_normed,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    # reshape back into standard shape
    grid_sampled = grid_sampled.squeeze(0)  # [C, (D), H, W]
    return grid_sampled


def affine_grid_generator(theta, size):
    if theta.data.is_cuda and len(size) == 4:
        if not cudnn.enabled:
            raise RuntimeError(
                "AffineGridGenerator needs CuDNN for "
                "processing CUDA inputs, but CuDNN is not enabled"
            )
        if not cudnn.is_acceptable(theta.data):
            raise RuntimeError(
                "AffineGridGenerator generator theta not acceptable for CuDNN"
            )
        N, C, H, W = size
        return torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        return AffineGridGenerator.apply(theta, size)


class AffineGridGenerator(Function):
    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError(
                "AffineGridGenerator needs CuDNN for "
                "processing CUDA inputs, but CuDNN is not enabled"
            )
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size

        if len(size) == 5:
            N, C, D, H, W = size
            ctx.size = size
            ctx.is_cuda = theta.is_cuda
            base_grid = theta.new(N, D, H, W, 4)

            w_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
            h_points = (
                torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
            ).unsqueeze(-1)
            d_points = (
                (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1]))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            base_grid[:, :, :, :, 0] = w_points
            base_grid[:, :, :, :, 1] = h_points
            base_grid[:, :, :, :, 2] = d_points
            base_grid[:, :, :, :, 3] = 1
            ctx.base_grid = base_grid
            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)

        elif len(size) == 4:
            N, C, H, W = size
            ctx.size = size
            if theta.is_cuda:
                AffineGridGenerator._enforce_cudnn(theta)
                assert False
            ctx.is_cuda = False
            base_grid = theta.new(N, H, W, 3)
            linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(
                base_grid[:, :, :, 0]
            )
            linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(
                base_grid[:, :, :, 1]
            )
            base_grid[:, :, :, 2] = 1
            ctx.base_grid = base_grid
            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        else:
            raise RuntimeError(
                "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs."
            )

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        if len(ctx.size) == 5:
            N, C, D, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, D, H, W, 3])
            assert ctx.is_cuda == grad_grid.is_cuda
            # if grad_grid.is_cuda:
            #     AffineGridGenerator._enforce_cudnn(grad_grid)
            #     assert False
            base_grid = ctx.base_grid
            grad_theta = torch.bmm(
                base_grid.view(N, D * H * W, 4).transpose(1, 2),
                grad_grid.view(N, D * H * W, 3),
            )
            grad_theta = grad_theta.transpose(1, 2)
        elif len(ctx.size) == 4:
            N, C, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, H, W, 2])
            assert ctx.is_cuda == grad_grid.is_cuda
            if grad_grid.is_cuda:
                AffineGridGenerator._enforce_cudnn(grad_grid)
                assert False
            base_grid = ctx.base_grid
            grad_theta = torch.bmm(
                base_grid.view(N, H * W, 3).transpose(1, 2), grad_grid.view(N, H * W, 2)
            )
            grad_theta = grad_theta.transpose(1, 2)
        else:
            assert False

        return grad_theta, None


def get_gaussian_kernel(
    kernel_size=3, sigma=2, channels=3, device="cuda", dtype=torch.float32, **kwargs
):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * np.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = (
        torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
            **kwargs
        )
        .to(device)
        .to(dtype)
    )

    gaussian_filter.weight.data = gaussian_kernel.to(device).to(dtype)
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
