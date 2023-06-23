# Source taken from: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class ColorJitter(T.ColorJitter):
    def __call__(self, image, target):
        return super().__call__(image), target


class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), torch.as_tensor(np.array(label, np.uint8, copy=False))

class ToHeatmap(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, image, *dets):
        peak, size = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, peak, size


def detections_to_heatmap(dets, shape, radius=2, device=None):
    with torch.no_grad():
        size = torch.zeros((2, shape[0], shape[1]), device=device)
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device) # HEATMAP
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(float), dtype=torch.float32, device=device)
                cx, cy = (det[:, 0] + det[:, 2] - 1) / 2, (det[:, 1] + det[:, 3] - 1) / 2
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
                mask = gaussian > peak.max(dim=0)[0]
                det_size = (det[:, 2:] - det[:, :2]).T / 2
                size[:, mask] = det_size[:, id[mask]]
                peak[i] = gaussian
        return peak, size