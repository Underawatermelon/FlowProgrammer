from typing import Optional, Union
from easydict import EasyDict
import torch
import pickle
import os
import argparse
from argparse import Namespace

SUPPORTED_MODELS = [
    "UNet",
    "AsymUNet",
    "UNet++",
    "CEyeNet",
    "gvtn",
]

MODE = ["train", "eval", "infer"]


class Config(EasyDict):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)

    def __build_default__(
        self,
        mode: str,
        model: str = "CEyeNet",
        resolution: Union[int, list] = 200,
    ):
        self.comments = ""
        self.model = model
        self.mode = mode
        self.profile_size = (
            resolution if isinstance(resolution, list) else [resolution, resolution]
        )
        assert self.mode in [
            "train",
            "eval",
            "infer",
        ], f"mode should be train/eval/infer, not '{self.mode}' ."
        assert self.model in SUPPORTED_MODELS, f"model '{self.model}' is not supported."
        self.simg_res = [200, 200]
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.amp = True if cuda else False
        self.__init_model_cfg__()
        self.__init_data_cfg__()
        if self.mode == "train":
            self.__init_train_cfg__()
            self.__init_loss_cfg__()
        if self.mode in ["train", "eval"]:
            self.__init_acc_cfg__()
        if self.mode in ["eval", "infer"]:
            self.__init_eval_cfg__()

    def __init_model_cfg__(self):
        assert self.model in SUPPORTED_MODELS, f"model '{self.model}' is not supported."
        model_cfg = EasyDict()
        if self.model == "gvtn":
            from config.gvtn_cfg import (
                conf_attn_down,
                conf_attn_same,
                conf_basic_ops,
                conf_gvtn,
                conf_attn_up,
            )

            model_cfg.gvtn = conf_gvtn
            model_cfg.basic_ops = conf_basic_ops
            model_cfg.attn_same = conf_attn_same
            model_cfg.attn_up = conf_attn_up
            model_cfg.attn_down = conf_attn_down
        else:
            model_cfg.base_num_features = (
                64  # number of output features of the first stack conv block
            )
            model_cfg.num_pool = 4  # number of downsampling/upsampling layers
            model_cfg.num_conv = 2  # number of blocks in the stack conv block
            model_cfg.kernel_size = [
                3 for i in range(model_cfg.num_conv)
            ]  # kernel size of the stacked conv layers
            model_cfg.dilation = [
                1 for i in range(model_cfg.num_conv)
            ]  # dilation of the stacked conv layers
            model_cfg.block_type = "conv"  # res/conv

            if self.model == "CEyeNet":
                model_cfg.pool_type = "avg"  # max/avg/bilinear/conv
                model_cfg.tp_order = "tile_first"  # tile_first/pool_first
                model_cfg.tile_factor = 7

            if self.model == "UNet++":
                model_cfg.unpp = EasyDict()
                model_cfg.unpp.input_channels = 1
                model_cfg.unpp.base_num_features = 64
                model_cfg.unpp.num_classes = 2
                model_cfg.unpp.num_pool = 4
                model_cfg.unpp.profile_size = [200, 200]

            if self.model == "AsymUNet":
                # extra config for decoder
                model_cfg.up_num_conv = 2
                model_cfg.up_kernel_size = [3] * model_cfg.up_num_conv
                model_cfg.up_dilation = [1] * model_cfg.up_num_conv
                model_cfg.up_block_type = "conv"
        self.model_cfg = model_cfg

    def __init_data_cfg__(self):
        assert self.mode in [
            "train",
            "eval",
            "infer",
        ], f"mode should be train/eval/infer, not '{self.mode}' ."

        assert self.model in SUPPORTED_MODELS, f"model '{self.model}' is not supported."
        data_cfg = EasyDict()

        # data augmentation control
        data_cfg.compeye = (
            False  # implement the compound eye resamling at the data transform stage
        )
        data_cfg.x_axial_flip = False
        data_cfg.x_randinv = True if self.mode == "train" else False
        data_cfg.x_rand_axial_translate = True if self.mode == "train" else False
        data_cfg.sym_agmentation = 0  # chances of lateral symmetric augmentation. 1.0 means always do it. currently not functional.

        if self.mode in ["train", "eval"]:
            data_cfg.data_root_dir = r"../dataset"
            data_cfg.dataset_size = [9000, 1000]  # dataset size for train and valid
            data_cfg.train_bs = 8
            data_cfg.valid_bs = 8
            data_cfg.workers = 4
        self.data_cfg = data_cfg

    def __init_train_cfg__(self):
        train_cfg = EasyDict()
        train_cfg.max_epoch = 300
        # log
        train_cfg.log_dir = r"../log"
        train_cfg.log_interval = max(int(9000 / self.data_cfg.train_bs // 5), 1)
        # training checkpoint path
        train_cfg.checkpoint = r""
        # learning rate (for step scheduler)
        train_cfg.lr_init = 1e-3
        train_cfg.factor = 0.1
        train_cfg.milestones = [30, 80]
        train_cfg.weight_decay = 5e-4
        train_cfg.momentum = 0.9
        # learning rate (for warmup cosine decay)
        train_cfg.is_warmup = True
        train_cfg.warmup_epochs = 1
        train_cfg.lr_final = 1e-5
        train_cfg.lr_warmup_init = 0.0
        # gradient hist (not used for now)
        train_cfg.hist_grad = False
        self.train_cfg = train_cfg

    def __init_loss_cfg__(self):
        loss_cfg = EasyDict()
        loss_cfg.components = [
            "l1",
            # "l2",
            "l1_grad",
            # "perceptual",
            # "gvtn_loss",
        ]
        # self.rae_epsilon = 0.2
        loss_cfg.weights = [1 for i in range(len(loss_cfg.components))]
        if "gvtn" in loss_cfg.components:
            from config.gvtn_cfg import conf_loss

            # the loss of gvtn keeps the original config as in the literature.
            loss_cfg.gvtn = conf_loss
        self.loss_cfg = loss_cfg

    def __init_eval_cfg__(self):
        eval_cfg = EasyDict()
        # infer weight path (for evaluation and predictor application)
        eval_cfg.infer_weight_path = r"" 
        if self.mode == "eval":
            eval_cfg.result_dir = r""
            eval_cfg.switches = {
                "tt_acc": True,
                "tt_pred_plot": True,
                "tt_label_plot": False,
            }

        self.eval_cfg = eval_cfg

    def __init_acc_cfg__(self):
        acc_cfg = EasyDict()
        acc_cfg.pix_acc_op_flags = [
            "round",
            "match",
        ]  # see full flags in tools/acc_metrics.py
        if "match" in acc_cfg.pix_acc_op_flags:
            if self.mode == "eval":
                acc_cfg.matching_error_thresholds = [
                    (0.2 - 0.025 * i).__round__(3) for i in range(9)
                ]
            else:
                acc_cfg.matching_error_thresholds = [
                    0.01,
                ]
        self.acc_cfg = acc_cfg

    def __modify__(self, key: Union[str, list], value):
        """If the key is not in the config, it will add the key to the config.
        If the key is in the config, it will update the value of the key.
        If trying to overwrite a non-dictionary with a dictionary, it will raise an error.

        Args:
            key (Union[str, list]): The key to modify in the config.
            value: The new value to assign to the key.

        Raises:
            ValueError: If the key is not a dictionary.
        """
        target = self
        for k in key[:-1]:
            if k not in target:
                target[k] = EasyDict()
            if not isinstance(target[k], EasyDict):
                raise ValueError(f"{k} is not a dictionary")
            target = target[k]
        target[key[-1]] = value

    @staticmethod
    def __list_config__(cfg: EasyDict, prefix=""):
        str_cfg = ""
        for k, v in cfg.items():
            if isinstance(v, EasyDict):
                sub_prefix = prefix + " | " + str(k)
                str_cfg += Config.__list_config__(v, sub_prefix)
            # skip functions
            elif not callable(v):
                str_cfg += f"{prefix} | {k}: {v}\n"
        str_cfg += "-----\n"
        return str_cfg

    @staticmethod
    def __save_config__(cfg: EasyDict, fname: str):
        with open(fname, "wb") as f:
            pickle.dump(cfg, f)

    @staticmethod
    def __export_config__(cfg: EasyDict, folder: str):
        path = os.path.join(folder, "cfg.txt")
        with open(path, "w") as f:
            f.write(Config.__list_config__(cfg))

    def __str__(self):
        return Config.__list_config__(self)

    def __repr__(self):
        return f"Configuration for {self.model} model in {self.mode} mode."

    def __assign_args__(self, args: Namespace):
        args_dict = vars(args)
        if args_dict.get("amp") is not None:
            if args_dict.get("amp") in ["off", "OFF", "false", "False", "0"]:
                args_dict["amp"] = False
            else:
                args_dict["amp"] = True

        for k, v in args_dict.items():
            if v is not None:
                keys = k.split(".")
                self.__modify__(keys, v)
        if args.device is not None:
            self.device = torch.device(args.device)


def create_cfg_from_args(args):
    model = args.model
    profile_size = args.profile_size if args.profile_size is not None else 200
    cfg = Config()
    cfg.__build_default__("train", model, profile_size)
    cfg.__assign_args__(args)
    return cfg


def parse_args(str_args: Optional[list] = None):
    parser = argparse.ArgumentParser(description="configure model from command line")

    parser.add_argument(
        "--comments",
        help="comments to be stored in the config",
        type=str,
    )
    parser.add_argument(
        "--model",
        help="model architecture",
        type=str,
    )
    parser.add_argument(
        "--profile_size",
        help="resolution of the output tensors",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--mode",
        help="mode of the model, can be train/eval/infer",
        type=str,
    )
    parser.add_argument(
        "--device",
        help="device to run the model",
        type=str,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--amp",
        help="use automatic mixed precision",
        type=str,
    )

    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument(
        "--base_num_features",
        help="number of output features of the first stack conv block",
        type=int,
        dest="model_cfg.base_num_features",
    )
    model_args.add_argument(
        "--num_pool",
        help="number of downsampling/upsampling layers",
        type=int,
        dest="model_cfg.num_pool",
    )
    model_args.add_argument(
        "--num_conv",
        help="number of conv layers in the stack conv block",
        type=int,
        dest="model_cfg.num_conv",
    )
    model_args.add_argument(
        "--kernel_size",
        help="kernel size of the stacked conv layers",
        type=int,
        nargs="+",
        dest="model_cfg.kernel_size",
    )
    model_args.add_argument(
        "--dilation",
        help="dilation of the stacked conv layers",
        type=int,
        nargs="+",
        dest="model_cfg.dilation",
    )
    model_args.add_argument(
        "--pool_type",
        help="pool type, max/avg/bilinear/conv",
        choices=["max", "avg", "bilinear", "conv"],
        type=str,
        dest="model_cfg.pool_type",
    )
    model_args.add_argument(
        "--tp_order",
        help="tile_first or pool_first in compound eye module implementation ",
        choices=["tile_first", "pool_first"],
        type=str,
        dest="model_cfg.tp_order",
    )
    model_args.add_argument(
        "--tile_factor", help="tile factor", type=int, dest="model_cfg.tile_factor"
    )

    model_args.add_argument(
        "--up_num_conv",
        help="number of conv layers in the upsample block",
        type=int,
        dest="model_cfg.up_num_conv",
    )

    model_args.add_argument(
        "--up_kernel_size",
        help="kernel size of conv in the upsample block",
        type=int,
        nargs="+",
        dest="model_cfg.up_kernel_size",
    )

    model_args.add_argument(
        "--up_dilation",
        help="dilation of the first conv in upsample block",
        type=int,
        nargs="+",
        dest="model_cfg.up_dilation",
    )

    model_args.add_argument(
        "--up_block_type",
        help="block type of the upsample block",
        type=str,
        dest="model_cfg.up_block_type",
    )

    data_args = parser.add_argument_group("Data Arguments")
    data_args.add_argument(
        "--compeye",
        help="implement the compound eye resampling to the input data in the data transform stage",
        dest="data_cfg.compeye",
    )
    data_args.add_argument(
        "--x_axial_flip",
        help="flip the input tensor along the axial direction",
        dest="data_cfg.x_axial_flip",
    )
    data_args.add_argument(
        "--x_randinv",
        help="randomly invert the input tensor",
        dest="data_cfg.x_randinv",
    )
    data_args.add_argument(
        "--x_rand_axial_translate",
        help="randomly translate the input tensor along the axial direction",
        dest="data_cfg.x_rand_axial_translate",
    )
    data_args.add_argument(
        "--sym_agmentation",
        help="chances of lateral symmetric augmentation",
        type=float,
        dest="data_cfg.sym_agmentation",
    )
    data_args.add_argument(
        "--data_root_dir",
        help="root directory of the dataset",
        type=str,
        dest="data_cfg.data_root_dir",
    )
    data_args.add_argument(
        "--dataset_size",
        help="dataset size for train and valid",
        type=int,
        nargs=2,
        dest="data_cfg.dataset_size",
    )
    data_args.add_argument(
        "--train_bs", help="batch size for training", type=int, dest="data_cfg.train_bs"
    )
    data_args.add_argument(
        "--valid_bs",
        help="batch size for validation",
        type=int,
        dest="data_cfg.valid_bs",
    )
    data_args.add_argument(
        "--workers",
        help="number of workers for data loading",
        type=int,
        dest="data_cfg.workers",
    )

    train_args = parser.add_argument_group("Train Arguments")
    train_args.add_argument(
        "--max_epoch",
        help="maximum number of epochs",
        type=int,
        dest="train_cfg.max_epoch",
    )
    train_args.add_argument(
        "--log_dir",
        help="directory to save the logs",
        type=str,
        dest="train_cfg.log_dir",
    )
    train_args.add_argument(
        "--log_interval",
        help="interval to log the training process",
        type=int,
        dest="train_cfg.log_interval",
    )
    train_args.add_argument(
        "--checkpoint",
        help="path to the training checkpoint",
        type=str,
        dest="train_cfg.checkpoint",
    )
    train_args.add_argument(
        "--lr_init", help="initial learning rate", type=float, dest="train_cfg.lr_init"
    )
    train_args.add_argument(
        "--factor",
        help="factor to reduce the learning rate",
        type=float,
        dest="train_cfg.factor",
    )
    train_args.add_argument(
        "--milestones",
        help="milestones to reduce the learning rate",
        type=int,
        nargs="+",
        dest="train_cfg.milestones",
    )
    train_args.add_argument(
        "--weight_decay", help="weight decay", type=float, dest="train_cfg.weight_decay"
    )
    train_args.add_argument(
        "--momentum",
        help="momentum for the optimizer",
        type=float,
        dest="train_cfg.momentum",
    )
    train_args.add_argument(
        "--is_warmup",
        help="use warmup learning rate",
        type=bool,
        default=argparse.SUPPRESS,
        dest="train_cfg.is_warmup",
    )
    train_args.add_argument(
        "--warmup_epochs",
        help="number of epochs for warmup",
        type=int,
        dest="train_cfg.warmup_epochs",
    )
    train_args.add_argument(
        "--lr_final", help="final learning rate", type=float, dest="train_cfg.lr_final"
    )
    train_args.add_argument(
        "--lr_warmup_init",
        help="initial learning rate for warmup",
        type=float,
        dest="train_cfg.lr_warmup_init",
    )
    train_args.add_argument(
        "--hist_grad",
        help="use gradient histogram",
        type=bool,
        dest="train_cfg.hist_grad",
    )

    loss_args = parser.add_argument_group("Loss Arguments")
    loss_args.add_argument(
        "--loss_components",
        help="loss components, l1/l2/l1_grad/perceptual",
        choices=["l1", "l2", "l1_grad", "perceptual"],
        type=str,
        nargs="+",
        dest="loss_cfg.components",
    )
    loss_args.add_argument(
        "--loss_weights",
        help="loss weights",
        type=float,
        nargs="+",
        dest="loss_cfg.weights",
    )

    acc_args = parser.add_argument_group("Accuracy Arguments")
    acc_args.add_argument(
        "--pix_acc_op_flags",
        help="pixel accuracy operation flags",
        choices=["IoU", "edge", "match", "round"],
        type=str,
        nargs="+",
        dest="acc_cfg.pix_acc_op_flags",
    )
    acc_args.add_argument(
        "--matching_error_thresholds",
        help="matching error limit",
        type=float,
        dest="acc_cfg.matching_error_thresholds",
        nargs="+",
    )
    acc_args.add_argument(
        "--acc_percep_with_pixel",
        help="use pixel-wise accuracy in perceptual similarity",
        type=bool,
        dest="acc_cfg.acc_percep_with_pixel",
    )
    acc_args.add_argument(
        "--perceptual_weights",
        help="perceptual weights, 5 terms",
        type=float,
        nargs="+",
        dest="acc_cfg.perceptual_weights",
    )

    eval_args = parser.add_argument_group("Evaluation Arguments")
    eval_args.add_argument(
        "--infer_weight_path",
        help="path to the inference weight",
        type=str,
        dest="eval_cfg.infer_weight_path",
    )
    eval_args.add_argument(
        "--result_dir",
        help="directory to save the results",
        type=str,
        dest="eval_cfg.result_dir",
    )
    eval_args.add_argument(
        "--tt_acc",
        help="switch for accuracy evaluation",
        type=bool,
        dest="eval_cfg.switches.tt_acc",
    )
    eval_args.add_argument(
        "--tt_pred_plot",
        help="switch for prediction plot",
        type=bool,
        dest="eval_cfg.switches.tt_pred_plot",
    )
    eval_args.add_argument(
        "--tt_label_plot",
        help="switch for label plot",
        type=bool,
        dest="eval_cfg.switches.tt_label_plot",
    )

    args = parser.parse_args(str_args)
    return args
