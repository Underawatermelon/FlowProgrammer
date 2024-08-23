"""perceptual laccuracy modified from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
"""
import numpy as np
import torch
import cv2
import torchvision
from easydict import EasyDict
from tqdm import tqdm


class PixelwiseAccuracy(object):
    """Pixelwise accurcy metric for evaluating transforamtion tensor and profiles."""

    def __init__(self, acc_cfg: EasyDict) -> None:
        assert hasattr(
            acc_cfg, "pix_acc_op_flags"
        ), "cfg should contain an flag to dictate pixelwise accuracy operation"
        self.op_dict = {
            "IoU": self.calc_IoU,
            "edge": self.edge_detect,
            "match": self.matching_rate,
            "round": self.round,
        }

        self.acc_cfg = acc_cfg
        self.operations = []
        op_flags = acc_cfg.pix_acc_op_flags

        tqdm.write(f"pixelwise accuracy operations:{op_flags}")
        if any([op == "match" for op in acc_cfg.pix_acc_op_flags]):
            tqdm.write(
                f"Matching error thresholds: {acc_cfg.matching_error_thresholds}"
            )

        for i in op_flags:
            self.operations.append(self.op_dict[i])
        # check if parameters are given for specific operations
        if "relative_threshold" in op_flags:
            assert hasattr(
                acc_cfg, "relative_threshold"
            ), "relative threshold should be specified in cfg"

    def __call__(self, samples: list):
        """samples: list of np.ndarray, each element is a tensor of shape ((B,) H, W, C)"""
        assert (
            len(samples) == 2
        ), "samples should contain a test and a reference, in the correct sequence"
        if len(samples[0].shape) == 3:
            samples = [
                np.expand_dims(i, axis=0) for i in samples
            ]  # shape to B, H, W, C

        data = np.array(samples)  # shape to 2, B, H, W, C
        for i in self.operations:
            data = i(data)
        return data

    def calc_IoU(self, samples):
        samples = samples > 0
        union_positive = (samples[0] | samples[1]).sum(axis=(-3, -2, -1))

        intersection_positive = (samples[0] & samples[1]).sum(axis=(-3, -2, -1))
        return intersection_positive / union_positive

    def matching_rate(self, samples):
        """percentage of matched points with error below the specified threshold.
        specify the threshold of correct matching in cfg"""
        num_points = np.prod(samples.shape[1:])  # B*H*W*C
        if type(self.acc_cfg.matching_error_thresholds) is not list:
            thres = [
                self.acc_cfg.matching_error_thresholds,
            ]
        else:
            thres = self.acc_cfg.matching_error_thresholds
        limits = np.array([np.full(samples.shape[1:], limit) for limit in thres])
        normalized_abs_diff = np.abs(
            (samples[0] - samples[1]) / (samples[1] + 1e-10)
        )  # num_thres, B, H, W, C
        normalized_abs_diff = np.expand_dims(normalized_abs_diff, axis=0)
        matched = np.sum(
            normalized_abs_diff <= limits,
            axis=tuple(range(1, normalized_abs_diff.ndim)),
        )
        return (matched / num_points).squeeze()

    def edge_detect(self, samples: list):
        """current implementation is for singel channel only"""
        for i in range(len(samples)):
            samples[i] = self.find_edge(samples[i])
        return samples

    def round(self, samples: list, n=0):
        """round the samples"""
        samples = np.round(samples, n)
        return samples

    @staticmethod
    def find_edge(img: np.ndarray):
        """input shape (B, H, W, C)
        find the edge of the profile image (cv2 standard, single channel)
        Warning: current implementation only support single channel"""
        img_ = img.squeeze()  # suppress channel dimension, which should equal to 1.
        grad = np.gradient(img_, axis=(-2, -1))
        grad = (np.abs(grad[0]) + np.abs(grad[1])) > 0
        edge = grad.astype("uint8") * 255  # shape (B, H, W)
        # take B as channel for cv2.dilate to dialte multichannels together
        edge = cv2.dilate(edge.transpose(1, 2, 0), np.ones((7, 7))).transpose(2, 0, 1)
        edge = np.expand_dims(edge, axis=-1)
        return edge


class PerceptualAccuracy(torch.nn.Module):
    """perceptual accuracy metric for evaluating profiles.
    supoort only green color profiles only.
    """

    def __init__(
        self,
        acc_cfg=None,
        resize=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(PerceptualAccuracy, self).__init__()
        self.acc_cfg = acc_cfg
        self.num_block = 4
        blocks = []
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:4].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[4:9].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[9:16].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights="IMAGENET1K_V1").features[16:23].eval()
        )

        for bl in blocks:
            for param in bl.parameters():
                param.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.device = device
        self.to(device)
        if acc_cfg is None:
            self.incl_pixacc = False
            self.weights = [
                4.0,
                1.0,
                0.25,
                0.125,
            ]
        else:
            self.incl_pixacc = acc_cfg.acc_percep_with_pixel
            if self.incl_pixacc:
                self.weights = acc_cfg.perceptual_weights
            else:
                self.weights = acc_cfg.perceptual_weights[1:]

        pixcfg = EasyDict()
        pixcfg.pix_acc_op_flags = ["match"]
        pixcfg.matching_error_thresholds = 0.0
        self.pixel_metric = PixelwiseAccuracy(pixcfg)

    def __call__(self, samples, ref_cfc=False):
        """input samples should be

        Args:
            samples: a list of two np.ndarray, each of shape ((B,) H, W, C)
            ref_cfc (bool, optional): if reference is confocal image.

        Returns:
            _type_: _description_
        """
        test, ref = samples
        if len(test.shape) == 3:
            test = np.expand_dims(test, axis=0)
        if len(ref.shape) == 3:
            ref = np.expand_dims(ref, axis=0)
        accuracy = []
        processed_test = []
        for img in test:
            processed_test.append(self.pre_img_process(img))
        test = np.array(processed_test)

        processed_ref = []
        for img in ref:
            processed_ref.append(self.pre_img_process(img, cfc_img=ref_cfc))
        ref = np.array(processed_ref)
        x = self.img2tensor(test)
        y = self.img2tensor(ref)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            acc = self.pixel_metric([x.cpu(), y.cpu()])
            accuracy.append(acc)
        accuracy = np.stack(accuracy, axis=1)
        if self.incl_pixacc:
            pix_acc = self.pixel_metric([test[:, :, 1], ref[:, :, 1]])
            accuracy = np.concatenate(
                [
                    np.array(pix_acc).reshape(
                        1,
                    ),
                    accuracy,
                ]
            )
            accuracy = np.average(accuracy, weights=self.weights, axis=1)
        else:
            accuracy = np.average(accuracy, weights=self.weights, axis=1)

        return accuracy

    def pre_img_process(self, img: np.ndarray, out_size=(224, 224), cfc_img=False):
        """unify the channel and size of images before comparison.
        keep green channel, binarize, resize, prepare for network
        """
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)
        if len(img.shape) == 2:
            empty_chn = np.zeros(img.shape, dtype="uint8")
            img = cv2.merge([empty_chn, img, empty_chn])  # BGR
        elif len(img.shape) == 3:
            empty_chn = np.zeros(img[:, :, 1].shape, dtype=np.uint8)
            img = cv2.merge([empty_chn, img[:, :, 1], empty_chn])  # BGR
        if cfc_img:
            img = cv2.GaussianBlur(img, (5, 5), 3)
            cb, cg, cr = cv2.split(img)
            cr = cv2.adaptiveThreshold(
                cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
            )
            cg = cv2.adaptiveThreshold(
                cg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
            )
            cb = cv2.adaptiveThreshold(
                cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 131, -3
            )
            img = cv2.merge([cb, cg, cr])
        return img

    def img2tensor(self, img):
        """nomralize the image for vgg input."""
        # from cv img to tensor
        # swap channels and rescale to [0., 1.]， normalize
        tensor = torch.tensor(img.transpose(0, 3, 1, 2)) / 255
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        return tensor


def calc_iou(a, b):
    """calculate the intersection over union of two binary arrays
    Args:
        a (np.ndarray):  tensor of shape (..., H, W, C)
        b (np.ndarray):  tensor (..., H, W, C)
    Returns:
        np.ndarray: iou value
    """
    assert a.shape == b.shape
    axis_for_sum = tuple(range(-3, 0))
    intersection = np.logical_and(a, b).sum(axis=axis_for_sum)
    union = np.logical_or(a, b).sum(axis=axis_for_sum)
    return (intersection / union).reshape(-1, *a.shape[:-3])


def calc_matching_rate(a, b, threshold=0):
    """calculate the matching rate of two binary arrays
    Args:
        a (np.ndarray):  tensor of shape (..., H, W, C)
        b (np.ndarray):  tensor (..., H, W, C)
        threshold (int, optional): maximum allowed relative error. Defaults to 0.
    Returns:
        np.ndarray: matching rate
    """
    assert a.shape == b.shape

    axis_for_sum = tuple(range(-3, 0))
    diff = np.abs(a - b) / (b + 1e-10)
    matched = np.sum(diff <= threshold, axis=axis_for_sum)
    if len(matched.shape) == 1:
        matched = matched[np.newaxis, ...]
    return matched / np.prod(a.shape[-3:])
