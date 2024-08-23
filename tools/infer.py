import os, sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.archs import *
from tools.trainer import Trainer
from data.transform import TransformObsImg
from utils.data_process import tt_convert, tt_postprocess, obs_params2imgs
from utils.visualization import (
    plot_tt_vf,
    create_obstacle_figure,
)


class TTPredictor(Trainer):
    """take raw data, apply transform, return the ready-to-use transformation tensor(s)."""

    def __init__(self, cfg):
        assert cfg.mode == "infer"
        super(TTPredictor, self).__init__(cfg)
        self.transform = TransformObsImg(self.cfg)
        self.obs_fig, self.obs_ax = create_obstacle_figure()

    def __call__(self, params, pos, transform=True, postprocess=True):
        obs_imgs = obs_params2imgs(params, pos, self.obs_ax)
        raw_tts = self.predict_from_obs_img(obs_imgs, transform, postprocess)
        tts = tt_postprocess(raw_tts)
        tts = tt_convert(tts, vert_sym=pos)
        return tts

    def predict_from_obs_img(self, img, transform=True, postprocess=True):
        """predict the transformation tensor (numpy arrary, ready-to-use) of the input image."""
        with torch.no_grad():

            if len(img.shape) == 3:
                img = img[np.newaxis, ...]
            if transform == True:
                transformed_img = []
                for i in img:
                    transformed_img.append(self.transform(i))
                img = torch.stack(transformed_img)
            img = img.to(self.cfg.device)
            output = self.model(img)
            if self.cfg.model == "gvtn":
                output = output[1]
            # squeeze if only output transforamtion tensor
            output = output.squeeze().cpu()
            if postprocess:
                output = tt_postprocess(output)
        return output


class Plotter:
    """Take x, y, yhat, plot in the same figure."""

    def __init__(
        self,
        save_dir="./tmp",
        plot_res=[40, 40],
        figsize=[5, 2],
        dpi=300,
    ):
        self.plot_res = plot_res
        self.save_dir = save_dir
        self.figsize = figsize
        self.dpi = dpi
        pass

    def __call__(self, x, y, y_hat, res_idx, namestring=""):
        fig, axes = self.plot_preds(x, y, y_hat)
        plt.savefig(self.save_dir + "/" + str(res_idx).zfill(4) + namestring + ".jpg")
        plt.close("all")

    def select_n_rand(self, x, y, n=3):
        perm = torch.randperm(len(x))
        return x[perm][:n], y[perm][:n]

    def plot_preds(self, x, y, y_hat):

        fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
        axes[0].set_title("x")
        axes[1].set_title("y")
        axes[2].set_title("y_hat")
        axes[0].imshow(x)
        axes[0].axis("off")
        plot_tt_vf(y.transpose(1, 2, 0), axes[1], plot_res=self.plot_res)
        plot_tt_vf(y_hat.transpose(1, 2, 0), axes[2], plot_res=self.plot_res)
        return fig, axes
