import dense_img_transforms

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import csv

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class SuperTuxDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        img_transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    ):
        self.dataset_path = dataset_path
        self.img_transforms = img_transform

        header = True
        self.labels = []
        with open(f"{dataset_path}/labels.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if not header:
                    self.labels.append(row)
                header = False

        self.size = len(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError

        img = Image.open(f"{self.dataset_path}/{str(idx+1).zfill(5)}.jpg")
        label = LABEL_NAMES.index(self.labels[idx][1])

        return (self.img_transforms(img), label)


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_img_transforms.ToTensor()):
        from glob import glob
        from os import path

        self.files = []
        for im_f in glob(path.join(dataset_path, "*_im.jpg")):
            self.files.append(im_f.replace("_im.jpg", ""))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + "_im.jpg")
        lbl = Image.open(b + "_seg.png")
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_img_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path

        self.files = []
        for im_f in glob(path.join(dataset_path, "*_im.jpg")):
            self.files.append(im_f.replace("_im.jpg", ""))
        self.transform = transform
        self.min_size = min_size

    def _filter(self, boxes):
        if len(boxes) == 0:
            return boxes
        return boxes[
            abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0]) >= self.min_size
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np

        b = self.files[idx]
        im = Image.open(b + "_im.jpg")
        nfo = np.load(b + "_boxes.npz")
        data = (
            im,
            self._filter(nfo["karts"]),
            self._filter(nfo["bombs"]),
            self._filter(nfo["pickup"]),
        )
        if self.transform is not None:
            data = self.transform(*data)
        return data
