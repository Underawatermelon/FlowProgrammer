import os, sys

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 2)))
sys.path.insert(0, parent_path)
import glob
import pickle
from tqdm import tqdm
import torch
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from tools.trainer import Trainer
from utils.utils import mkdirs, Timer
from utils.io import mirror_model_dir_tree
from utils.data_process import tt_postprocess
from utils.visualization import plot_tt_contour, plot_tt_contour_colormap


def list_models(root_dir, to_file=True):
    model_paths = [
        file_dir
        for file_dir in glob.iglob(root_dir + "**/**", recursive=True)
        if os.path.isfile(file_dir)
        and (
            os.path.basename(file_dir).startswith("UNet")
            or os.path.basename(file_dir).startswith("CEyeNet")
            or os.path.basename(file_dir).startswith("GVTN")
            or os.path.basename(file_dir).startswith("AsymUNet")
        )
        and not os.path.basename(file_dir).endswith("_last")
    ]

    model_paths = sorted(list(set(model_paths)))

    if to_file:
        with open(os.path.join(root_dir, "model_path_list.txt"), "w") as f:
            f.write("\n".join(model_paths))

    return model_paths


class ModelEvaluator(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=300)
        self.switches = cfg.eval_cfg.switches
        # create metrics at different matching limits

    def evaluation(self):
        self.result_path = mirror_model_dir_tree(
            self.cfg.eval_cfg.result_dir, self.cfg.eval_cfg.infer_weight_path
        )
        # save cfg &  model
        save_cfg(self.cfg, self.result_path) 
        with open(os.path.join(self.result_path, "cfg.txt"), "w") as f:
            f.write(str(self.cfg))
        self.export_model(
            self.model,
            os.path.join(self.result_path, "model.pt"),
        )
        self.model.eval()
        stp_idx = 0
        timer = Timer(verbose=False)
        infer_time_records = []
        total_infer_time = 0
        postprcs_t_records = []
        total_t_postprcs = 0
        y_preds = []
        y_labels = []
        if self.switches["tt_acc"]:
            acc_records = []
        if self.switches["tt_pred_plot"]:
            mkdirs(os.path.join(self.result_path, "pred_ctr"))
        if self.switches["tt_label_plot"]:
            mkdirs(os.path.join(self.result_path, "label_ctr"))
        if self.switches["tt_label_plot"] or self.switches["tt_pred_plot"]:
            plot_tt_contour_colormap(self.ax, self.cfg.profile_size)
            plt.savefig(os.path.join(self.result_path, "legend.png"))
        progress_bar = tqdm(
            self.valid_loader,
            desc="eval batch",
            leave=False,
            total=len(self.valid_loader),
        )
        for data in progress_bar:
            # batch pred
            inputs, labels = data
            inputs, labels = inputs.to(self.cfg.device.type), labels.to(self.cfg.device)
            with timer:
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.cfg.device.type,
                        dtype=torch.float16,
                        enabled=self.cfg.amp,
                    ):
                        outputs = self.model(inputs)
            infer_time_records.append(timer.get_time())
            total_infer_time += infer_time_records[-1]
            if self.cfg.model == "gvtn":
                outputs = outputs[1]
            with timer:
                outputs = tt_postprocess(outputs)
                labels = tt_postprocess(labels)
            postprcs_t_records.append(timer.get_time())
            total_t_postprcs += postprcs_t_records[-1]
            y_preds.append(outputs)
            y_labels.append(labels)

            # per tt acc
            if self.switches["tt_acc"]:
                for pred, label in zip(outputs, labels):
                    # compute accuracy & log
                    acc = self.acc_metric([pred, label])
                    acc_records.append(acc)
                    if stp_idx % 50 == 0:
                        progress_bar.set_description(
                            f"sample {stp_idx} acc:  {acc_records[-1].round(3)}]"
                        )
                    stp_idx += 1
        average_infer_time = (total_infer_time / (stp_idx + 1)).__round__(3)
        std_infer_time = np.std(infer_time_records) / self.cfg.data_cfg.valid_bs
        average_postprocess_time = (total_t_postprcs / (stp_idx + 1)).__round__(3)
        std_postprocess_time = np.std(postprcs_t_records) / self.cfg.data_cfg.valid_bs
        with open(os.path.join(self.result_path, "summary.txt"), "w") as f:
            f.write("model_path:" + str(self.cfg.eval_cfg.infer_weight_path) + "\n")
            f.write("total_samples:" + str(stp_idx) + "\n")
            f.write("toal_infer_time:" + str(total_infer_time.__round__(3)) + "ms\n")
            f.write("average_infer_time:" + str(average_infer_time) + "ms\n")
            f.write("std_infer_time:" + str(std_infer_time) + "ms\n")
            f.write(
                "toal_postprocess_time:" + str(total_t_postprcs.__round__(3)) + "ms\n"
            )
            f.write(
                "average_postprocess_time:" + str(average_postprocess_time) + "ms\n"
            )
            f.write("std_postprocess_time:" + str(std_postprocess_time) + "ms\n")
        # save time records to csv file.
        np.savetxt(
            os.path.join(self.result_path, "infer_time_records.csv"),
            np.array(infer_time_records),
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self.result_path, "postprocess_time_records.csv"),
            np.array(postprcs_t_records),
            delimiter=",",
        )

        # save y_preds and y_labels
        yps = np.concatenate(y_preds)
        np.save(os.path.join(self.result_path, "y_preds.npy"), yps)
        if self.switches["tt_label_plot"]:
            yls = np.concatenate(y_labels)
            np.save(os.path.join(self.result_path, "y_labels.npy"), yls)
        if self.switches["tt_acc"]:
            # save all accuracys to csv
            acrd = np.array(acc_records).round(4)
            np.savetxt(
                os.path.join(self.result_path, "acc_records.csv"),
                acrd,
                delimiter=",",
            )
            avg_acc = np.mean(acrd, axis=0)
            std_acc = np.std(acrd, axis=0)
            with open(os.path.join(self.result_path, "summary.txt"), "a") as f:
                f.write(
                    "metric_thresholds:"
                    + str(self.cfg.acc_cfg.matching_error_thresholds)
                    + "\n"
                )
                f.write("average_acc:" + str(avg_acc) + "\n")
                f.write("std_acc:" + str(std_acc) + "\n")
        # plot and save the contour
        if self.switches["tt_pred_plot"]:
            for i in tqdm(range(yps.shape[0]), desc="plot pred contour", leave=False):
                self.save_tt_contour(yps[i], i, "pred_ctr")
            self.save_tt_contour(pred, stp_idx, "pred_ctr")
        if self.switches["tt_label_plot"]:
            for i in tqdm(range(yls.shape[0]), desc="plot label contour", leave=False):
                self.save_tt_contour(yls[i], i, "label_ctr")
        # finish
        tqdm.write(f"model {self.cfg.eval_cfg.infer_weight_path} evaluation finished.")
        if self.switches["tt_acc"]:
            tqdm.write(f"accuracies: {avg_acc}")
            tqdm.write(f"average infer time: {average_infer_time}ms")

    def save_tt_contour(self, tt, idx, foler):
        plot_tt_contour(tt, self.ax)
        self.fig.savefig(os.path.join(self.result_path, foler, f"pred_cntr{idx}.png"))

    def configure_output(self, tt_acc=False, tt_pred_plot=False, tt_label_plot=False):
        self.switches["tt_acc"] = tt_acc
        self.switches["tt_pred_plot"] = tt_pred_plot
        self.switches["tt_label_plot"] = tt_label_plot


def save_cfg(cfg, save_path):
    cfg_file = os.path.join(save_path, "cfg.pkl")
    with open(cfg_file, "wb") as f:
        pickle.dump(cfg, f)


def load_cfg(load_path):
    cfg_file = os.path.join(load_path, "cfg.pickle")
    with open(cfg_file, "rb") as f:
        cfg = pickle.load(f)
    return cfg
