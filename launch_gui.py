#!/usr/bin/env python

import sys, os
from gui.app import MainWindow
from PyQt5.QtWidgets import QApplication
from config.config import Config, parse_args
from utils.io import create_infer_config_from_file

cfg = Config()
args = parse_args()
arg_dict = vars(parse_args())
infer_weight_path = (
    arg_dict["eval_cfg.infer_weight_path"]
    if arg_dict["eval_cfg.infer_weight_path"]
    else None
)

if infer_weight_path:
    model_dir = os.path.dirname(infer_weight_path)
    cfg = create_infer_config_from_file(infer_weight_path)
    cfg.__assign_args__(args)
    print(f"Loaded model config from {infer_weight_path}.")
    print(cfg)
    cfg.__save_config__(cfg, os.path.join(model_dir, "app_cfg.pickle"))

else:
    print("No model path is given. Loading default model and config.")
    infer_weight_path = r"../log/CEyeNet/CEyeNet"
    cfg.__build_default__("infer", "CEyeNet", 200)
    cfg.eval_cfg.infer_weight_path = infer_weight_path
    cfg.__assign_args__(args)
    print(cfg)

app = QApplication(sys.argv)
window = MainWindow(cfg=cfg)
window.show()
app.exec()
