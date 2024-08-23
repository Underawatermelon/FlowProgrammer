import os
import pickle
from config.config import Config
from utils.utils import mkdirs
import re


def create_infer_config_from_file(infer_weight_path):
    cfg = Config()
    model_dir = os.path.dirname(infer_weight_path)
    if os.path.exists(os.path.join(model_dir, "app_cfg.pickle")):
        with open(os.path.join(model_dir, "app_cfg.pickle"), "rb") as f:
            original_cfg = pickle.load(f)
    elif os.path.exists(os.path.join(model_dir, "cfg.pickle")):
        with open(os.path.join(model_dir, "cfg.pickle"), "rb") as f:
            original_cfg = pickle.load(f)
    else:
        raise FileNotFoundError(
            "Could not find the model config file. shoud be app_cfg.pickle or cfg.pickle"
        )

    cfg.__build_default__("infer", original_cfg.model, original_cfg.profile_size)
    cfg.model_cfg = original_cfg.model_cfg
    cfg.data_cfg.compeye = original_cfg.data_cfg.compeye
    cfg.eval_cfg.infer_weight_path = infer_weight_path
    if hasattr(cfg.model_cfg, "block_type") and cfg.model_cfg.block_type == "res":
        cfg.model_cfg.dilation = [
            1,
        ] * cfg.model_cfg.num_conv
        cfg.model_cfg.kernel_size = [
            3,
        ] * cfg.model_cfg.num_conv
    return cfg


def mirror_model_dir_tree(dest_dir, model_path):
    """mirror the model directory structure in the result directory."""
    # extract each level of directory
    dirs = os.path.relpath(model_path, r"../").split(os.sep)
    save_path = os.path.join(dest_dir, *dirs[1:-1])
    return mkdirs(save_path, remove_old=False)
    # load model predictor.


def find_index(a):
    index = re.findall("(?<=-)\d+", a)
    if index:
        return int(index[0])
    else:
        return None


class FindIndex:
    def __init__(self, pattern="(?<=-)\d+") -> None:
        self.pattern = pattern

    def __call__(self, a):
        index = re.findall(self.pattern, a)
        if index:
            return int(index[0])
        else:
            return None
