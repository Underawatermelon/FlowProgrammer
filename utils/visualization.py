import numpy as np
import torch
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Union, Literal


def plot_tt_vf(
    tt: Union[torch.Tensor, np.ndarray], ax: mpl.axes.Axes, plot_res=[10, 10]
):
    """input tensor (the dimension of displacement [x, y] are dim 0); or
    input  array (the dimension of displ [x, y] are dim 2 )

    Args:
        tt (torch.Tensor or np.array): input array
        ax (mpl.axes.Axes): axes to plot tt on
    """
    if type(tt) == torch.Tensor:
        tt = tt.detach().cpu().numpy()
        tt = tt.transpose((1, 2, 0))
    res = tt.shape[0:2]
    ax.cla()
    steps = (np.array(res) / np.array(plot_res)).round().astype("int16")
    step_y, step_z = np.maximum(steps, 1)
    arrow_line_width = res[0] / 20 / plot_res[0]
    arrow_head_width = 5 * arrow_line_width
    for cy in range(0, res[0], step_y):
        for cz in range(0, res[1], step_z):
            ax.arrow(
                cy + tt[cy, cz, 0],
                cz + tt[cy, cz, 1],
                -tt[cy, cz, 0],
                -tt[cy, cz, 1],
                lw=0,
                width=arrow_line_width,
                head_width=arrow_head_width,
                length_includes_head=True,
            )
    ax.axis("equal")
    ax.axis([-2, res[0] + 1, -2, res[1] + 1])
    ax.tick_params(labelsize="5")
    ax.grid(False)
    return ax


def plot_tt_contour(
    tt,
    ax,
    mode: Literal["hsv", "rgb"] = "hsv",
    cmin=None,
    cmax=None,
    vmin=None,
    vmax=None,
    iscolormap=False,
):
    ax.cla()
    ax.axis("equal")
    if mode == "hsv":
        if cmin is None:
            cmin = 100
        if cmax is None:
            cmax = 255
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = tt.shape[0] * 0.5

        tt = tt.astype(np.float32)
        mag, ang = cv2.cartToPolar(tt[..., 0], tt[..., 1])
        mag = np.clip(mag, vmin, vmax)
        hsv = np.zeros(tt.shape[0:2] + (3,), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, cmin, cmax, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if iscolormap:
            mask = mag >= vmax
            rgb[mask] = [255, 255, 255]

        extent = (
            [
                tt.shape[0] * -0.5,
                tt.shape[0] * 0.5,
                tt.shape[1] * -0.5,
                tt.shape[1] * 0.5,
            ]
            if iscolormap
            else None
        )
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=extent)

    elif mode == "rgb":
        if cmin is None:
            cmin = 50
        if cmax is None:
            cmax = 255
        rgb = np.zeros(tt.shape[0:2] + (3,), dtype=np.uint8)
        rgb[..., 0] = cv2.normalize(tt[..., 0], None, cmin, cmax, cv2.NORM_MINMAX)
        rgb[..., 1] = cv2.normalize(tt[..., 1], None, cmin, cmax, cv2.NORM_MINMAX)
        rgb[..., 2] = cmin
        ax.imshow(rgb)


def plot_tt_contour_colormap(
    ax: mpl.axes.Axes, profile_size, vmin=None, vmax=None, cmin=None, cmax=None
):
    res = profile_size[0]
    if cmin is None:
        cmin = 100
    if cmax is None:
        cmax = 255
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = res * 0.5
    vrange = vmax - vmin
    mag_par1 = np.linspace(0, 2 * vrange, res) - vrange + vmin
    mag_par2 = ((np.linspace(-1, 1, res) > 0) * 2 - 1) * vmin
    x = mag_par1 + mag_par2
    y = x.copy()
    xy = np.stack(np.meshgrid(x, y), axis=2).reshape(-1, 2)
    plot_tt_contour(
        xy.reshape(res, res, 2),
        ax,
        mode="hsv",
        vmin=vmin,
        vmax=vmax,
        cmin=cmin,
        cmax=cmax,
        iscolormap=True,
    )
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(3)  # Set the linewidth of the left spine
    ax.spines["bottom"].set_linewidth(3)  # Set the linewidth of the bottom spine

    ax.tick_params(
        axis="both", which="both", labelbottom=False, labelleft=False, length=5, width=2
    )
    ax.set_xticks(np.linspace(-res / 2, res / 2, num=5))
    ax.set_yticks(np.linspace(-res / 2, res / 2, num=5))
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_title(f"vmin={vmin}, vmax={vmax}")


def plot_color_tt(
    tt: Union[torch.Tensor, np.ndarray], ax: mpl.axes.Axes, plot_res=[10, 10]
):
    if type(tt) == torch.Tensor:
        tt = tt.detach().cpu().numpy()
        tt = tt.transpose((1, 2, 0))
    res = tt.shape[0:2]
    norm = np.linalg.norm(tt, axis=2)
    max_displ = np.max(norm)
    min_displ = np.min(norm)
    ax.cla()
    ax.axis([-0.5, res[0] - 0.5, -0.5, res[1] - 0.5])
    ax.axis("equal")
    ax.tick_params(labelsize="5")
    ax.grid(False)
    step_y, step_z = (np.array(res) / np.array(plot_res)).round().astype("int16")

    # select array according to plot res
    compact_tt = tt[::step_y, ::step_z]
    compact_norm = np.linalg.norm(compact_tt, axis=2)
    y, x = np.meshgrid(np.arange(0, res[0], step_y), np.arange(0, res[1], step_z))
    # the meshgrid ouput have opposite index axis and xy axis.
    dx, dy = np.split(-compact_tt, 2, 2)
    dx, dy = dx.flatten(), dy.flatten()
    ax.quiver(
        x,
        y,
        dx,
        dy,
        compact_norm,
        pivot="tip",
        angles="uv",
        cmap="rainbow",
        norm=mpl.colors.Normalize(vmin=min_displ, vmax=max_displ),
    )
    return ax


def create_obstacle_figure():
    fig = plt.figure(figsize=(5, 2), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.axis("equal")
    return fig, ax


def plot_obstacle(
    p: Union[torch.Tensor, np.ndarray],
    pos: bool,
    ax,
    for_infer: bool = True,
):
    if type(p) == np.ndarray:
        p = torch.tensor(p)
    if bool(pos) == True:
        color = "royalblue"
    else:
        color = "orangered"
    with torch.no_grad():
        assert p.size()[0] == (18)

        x = torch.cat((p[0::3], torch.flip(p[1::3], dims=(0,))))
        y = torch.cat((p[2::3], torch.flip(p[2::3], dims=(0,))))
        xy = torch.stack((x, y), dim=1)

    if hasattr(ax, "obs_patch"):
        ax.obs_patch.set(xy=xy, color=color, linewidth=0)
    else:
        ax.obs_patch = ax.add_patch(
            mpatches.Polygon(xy, closed=True, color=color, linewidth=0)
        )
        if for_infer:
            ax.axis([-1.25 * p[17], 1.25 * p[17], 0, p[17]])
    return


def create_profile_figure():
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.axis("equal")
    return fig, ax


def plot_pf_tensor(profile_tensor, ax=None):
    """plot the profile image with profile tensor."""
    if profile_tensor.ndim == 2:
        # append r channel and b channel with all 0
        profile_tensor = np.stack(
            [
                np.zeros_like(profile_tensor),
                profile_tensor,
                np.zeros_like(profile_tensor),
            ],
            axis=2,
        )

    if ax is None:
        fig, ax = create_profile_figure()
    else:
        fig = ax.get_figure()
    if hasattr(ax, "p_img"):
        ax.p_img.set_data(profile_tensor.transpose([1, 0, 2])[::-1, :, :])
    ax.p_img = ax.imshow(profile_tensor.transpose([1, 0, 2])[::-1, :, :])
    ax.axis("off")
    return fig, ax


def fig2np(fig):
    """from mpl canvas to numpy array.
    Out matrix dtype='unit8', RGB dim = 3, 0-255
    """
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def generate_autocad_script(filename, coords=None, offset=0.8):
    """generate autocad script to draw obstacles."""
    # convert coords from m/um to mm depend on the max value of coords
    num_obs = coords.shape[0]
    if np.max(coords) < 1e-3:
        coords = coords * 1e3
    elif np.max(coords) > 1:
        coords = coords / 1e3
    coords = coords.reshape(-1, 6, 3)
    with open(filename, "w") as f:
        f.write(f"-OSNAP\n\n")
    for index in range(num_obs):
        ofs = offset * index
        obstacle = coords[index]
        front_edge = obstacle[:, [0, 2]]
        front_edge[:, 0] += ofs
        back_edge = obstacle[::-1, [1, 2]]
        back_edge[:, 0] += ofs
        points = np.concatenate((front_edge, back_edge), axis=0)
        points = np.round(points, 4)
        with open(filename, "a") as f:
            f.write(f";step{index+1}\n")
            for i in range(len(points) - 1):
                start_point = points[i]
                end_point = points[i + 1]
                f.write(
                    f"line {start_point[0]},{start_point[1]} {end_point[0]},{end_point[1]}\n\n"
                )
    with open(filename, "a") as f:
        f.write("zoom e\n")
        f.write("regen\n")


def create_cmap(color: np.ndarray):
    """create a colormap from half way to white to the given color in RGBA format."""
    color = np.array(color)
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace((color[0] / 256 + 1) / 2, color[0] / 256, N)
    vals[:, 1] = np.linspace((color[1] / 256 + 1) / 2, color[1] / 256, N)
    vals[:, 2] = np.linspace((color[2] / 256 + 1) / 2, color[2] / 256, N)
    vals[0, 3] = 1
    newcmp = ListedColormap(vals)
    return newcmp
