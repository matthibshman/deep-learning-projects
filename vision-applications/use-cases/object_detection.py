from utils import load_detection_data, DetectionSuperTuxDataset
import dense_img_transforms

import torch
import torchvision
import torch.utils.tensorboard as tb
from os import path

TRAIN_DATA_PATH = "dense_data/train"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Detector(torch.nn.Module):
    class ConvBlock(torch.nn.Module):
        def __init__(self, input_channels, output_channels, stride):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(
                    input_channels, output_channels, kernel_size=3, padding=2, dilation=stride
                ),
                torch.nn.GroupNorm(8, output_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                torch.nn.GroupNorm(8, output_channels),
                torch.nn.ReLU(),
            )
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 1, dilation=stride),
                torch.nn.GroupNorm(8, output_channels),
            )

        def forward(self, x):
            # w/ residual (downsample to match output of network)
            return self.net(x) + self.downsample(x)

    class UpConvBlock(torch.nn.Module):
        def __init__(self, input_channels, output_channels, stride, groups, zero_weights=False):
            super().__init__()
            up_conv = torch.nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=2,
                dilation=stride,
            )
            # if zero_weights:
            #     up_conv.weight.zero_
            self.net = torch.nn.Sequential(
                up_conv,
                torch.nn.GroupNorm(groups, output_channels),
                torch.nn.ReLU(),
            )
            self.upsample = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    dilation=stride,
                ),
                torch.nn.GroupNorm(groups, output_channels),
            )

        def forward(self, x):
            # w/ residual (downsample to match output of network)
            return self.net(x) + self.upsample(x)

    def __init__(
        self,
        layer_sizes=[16, 32, 64, 128, 256],
        input_channels=3,
        num_classes=3,
        stride=2,
        input_transforms=torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.2788, 0.2657, 0.2628], std=[0.2064, 0.1944, 0.2252]
                )
            ]
        ),
    ):
        super().__init__()
        self.conv_layers = []
        self.up_conv_layers = []
        self.input_transforms = input_transforms
        layers = [
            torch.nn.Conv2d(input_channels, num_classes, kernel_size=7, padding=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        ]

        output_channels = input_channels
        for channels in layer_sizes:
            conv_block = self.ConvBlock(output_channels, channels, stride=stride)
            self.conv_layers.append(conv_block)
            groups = int(output_channels / 2)
            self.up_conv_layers = [
                self.UpConvBlock(channels, output_channels, stride, groups)
            ] + self.up_conv_layers
            output_channels = channels

        self.layers = torch.nn.ModuleList((self.conv_layers + self.up_conv_layers))
        self.first_layer = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, num_classes, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(num_classes, num_classes, 1),
        )
        self.classifier[2].weight.zero_

    def forward(self, x):
        outputs = []
        layer_input = self.first_layer(self.input_transforms(x))
        for layer in self.conv_layers:
            layer_input = layer(layer_input)
            outputs.append(layer_input)

        outputs.pop()
        for layer in self.up_conv_layers[:-1]:
            conv_output = outputs.pop()
            layer_output = layer(layer_input)
            # resize from up-conv if needed
            if layer_output.shape != conv_output.shape:
                layer_output = layer_output[:, :, : conv_output.shape[2], : conv_output.shape[3]]
            layer_input = layer_output + conv_output

        result = self.up_conv_layers[-1](layer_input)
        # resize from up-conv if needed
        if result.shape[2:] != x.shape[2:]:
            result = result[:, :, : x.shape[2], : x.shape[3]]
        return self.classifier(result)

    def extract_peak(self, heatmap, max_pool_ks=7, min_score=-5.0, max_det=100):
        """
        Extract local maxima in a 2d heatmap.
        @heatmap: H x W heatmap
        @max_pool_ks: return points larger than a max_pool_ks x max_pool_ks window around a point
        @min_score: return peaks greater than min_score
        @max_det: return no more than max_det peaks
        @return: List of peaks [(score, cx, cy), ...]
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _, width = heatmap.size()
        heatmap = heatmap.to(device)
        window_size = (max_pool_ks - 1) * 2 + 1
        max_pool = torch.nn.MaxPool2d(window_size, stride=1, padding=max_pool_ks - 1)
        max_map = max_pool(heatmap[None, None]).squeeze()

        max_equals_heatmap = torch.eq(max_map, heatmap).float()
        maxima = (heatmap * max_equals_heatmap).flatten()

        topk = min(max_det, maxima.size(0))
        maxima_vals, maxima_indices = maxima.topk(topk)

        peaks = [
            (
                maxima_vals[i],
                maxima_indices[i].remainder(width),
                torch.div(maxima_indices[i], width, rounding_mode="trunc"),
            )
            for i in range(maxima_vals.size(0))
            if maxima_vals[i] > min_score
        ]

        return peaks

    def detect(self, image):
        """
        @image: 3 x H x W image
        @return: Three list of detections [(score, cx, cy, ...], one per class
        """
        self.eval()
        forward_output = torch.sigmoid(self(image[None]).squeeze())

        detections = []
        for class_index in range(forward_output.size()[0]):
            detections.append(
                [
                    centers + (0, 0)
                    for centers in self.extract_peak(forward_output[class_index], 5, 0.5, 30)
                ]
            )
        return detections


def train(args):
    model = Detector()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
    model = model.to(device)

    class_weights = torch.tensor(compute_weight_imbalance()).to(device)
    print(class_weights)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optim = get_optimizer(model.parameters(), args.learning_rate)
    lr_scheduler = get_scheduler(optim)

    train_data_loader = load_detection_data(
        TRAIN_DATA_PATH,
        batch_size=16,
        transform=dense_img_transforms.Compose(
            [
                dense_img_transforms.ColorJitter(
                    brightness=0.9, contrast=0.9, saturation=0.9, hue=0.1
                ),
                dense_img_transforms.RandomHorizontalFlip(flip_prob=0.5),
                dense_img_transforms.ToTensor(),
                dense_img_transforms.ToHeatmap(),
            ]
        ),
    )

    global_step = 0
    for _ in range(args.epochs):
        model.train()
        for image, heatmaps, _ in iter(train_data_loader):
            train_features, train_labels = (
                image.to(device),
                heatmaps.to(device),
            )
            train_labels = torch.where(train_labels > 0.75, 1.0, 0.0)

            forward_output = model(train_features)
            loss = loss_function(forward_output, train_labels)
            train_logger.add_scalar("train/loss", loss, global_step)
            global_step += 1

            optim.zero_grad()
            loss.backward()
            optim.step()
        lr_scheduler.step()


def compute_weight_imbalance():
    train_data_loader = load_detection_data(
        TRAIN_DATA_PATH,
        transform=dense_img_transforms.Compose(
            [
                dense_img_transforms.ToTensor(),
                dense_img_transforms.ToHeatmap(),
            ]
        ),
    )

    zeros = 0.0
    total = 0.0
    for _, heatmaps, _ in iter(train_data_loader):
        binary = torch.where(heatmaps > 0.75, 1.0, 0.0)
        zeros += (binary == 0.0).sum()
        total += (binary).sum()

    return zeros / total


def get_optimizer(params, lr):
    return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=0.0005)
    # return torch.optim.Adam(params, lr, weight_decay=1e-4)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.9)
    # return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)


def compute_input_norm():
    train_data_loader = load_detection_data(
        TRAIN_DATA_PATH,
        transform=dense_img_transforms.Compose(
            [
                dense_img_transforms.ToTensor(),
                dense_img_transforms.ToHeatmap(),
            ]
        ),
    )

    x = torch.zeros(3)
    x_squared = torch.zeros(3)
    num_batches = 0.0
    for image, _, _ in iter(train_data_loader):
        x += torch.mean(image, dim=[0, 2, 3])
        x_squared += torch.mean(torch.pow(image, 2), dim=[0, 2, 3])
        num_batches += 1.0
    mean = x_squared / num_batches
    std_dev = torch.pow((x_squared / num_batches - torch.pow(mean, 2.0)), 0.5)
    print(mean)
    print(std_dev)


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--epochs", default=1, type=int)
# parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
# parser.add_argument("-inorm", "--input_norm", default=False, type=bool)
# parser.add_argument("-log", "--log_dir", default="")
# args = parser.parse_args()
# if args.input_norm:
    # compute_input_norm()
    # compute_weight_imbalance()
# train(args)