from utils import load_dense_data
from datasets import DENSE_CLASS_DISTRIBUTION
from dense_img_transforms import (
    Compose,
    ColorJitter,
    RandomHorizontalFlip,
    ToTensor,
)

import torch
import torchvision
import torch.utils.tensorboard as tb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TRAIN_DATA_PATH = "dense_data/train"
VALID_DATA_PATH = "dense_data/valid"


class FCN(torch.nn.Module):
    class ConvBlock(torch.nn.Module):
        def __init__(self, input_channels, output_channels, stride):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(
                    input_channels, output_channels, kernel_size=3, padding=1, stride=stride
                ),
                torch.nn.GroupNorm(8, output_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                torch.nn.GroupNorm(8, output_channels),
                torch.nn.ReLU(),
            )
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, output_channels, 1, stride=stride),
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
                padding=1,
                stride=stride,
                output_padding=1,
            )
            if zero_weights:
                up_conv.weight.zero_
            self.net = torch.nn.Sequential(
                up_conv,
                torch.nn.GroupNorm(groups, output_channels),
                torch.nn.ReLU(),
            )
            self.upsample = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size=1, stride=stride, output_padding=1
                ),
                torch.nn.GroupNorm(groups, output_channels),
            )

        def forward(self, x):
            # w/ residual (downsample to match output of network)
            return self.net(x) + self.upsample(x)

    def __init__(
        self,
        layer_sizes=[16, 32, 64, 128],
        input_channels=3,
        num_classes=5,
        stride=2,
        input_transforms=torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.1688, 0.1590, 0.1803], std=[0.3746, 0.3657, 0.3845]
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

        output_channels = num_classes
        for channels in layer_sizes:
            conv_block = self.ConvBlock(output_channels, channels, stride=stride)
            self.conv_layers.append(conv_block)
            groups = int(output_channels / 2)
            zero_weights = False
            # handle output convolution
            if output_channels == num_classes:
                groups = 1
                zero_weights = True
            self.up_conv_layers = [
                self.UpConvBlock(channels, output_channels, stride, groups, zero_weights)
            ] + self.up_conv_layers
            output_channels = channels

        self.layers = torch.nn.ModuleList((self.conv_layers + self.up_conv_layers))
        self.first_layer = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        @x: Tensor((Batch,3,H,W))
        @return: Tensor((Batch,5,H,W))
        """
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
        return result


def train(args):
    from os import path

    model = FCN()
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)

    model = model.to(device)

    class_weights = torch.pow(torch.tensor(DENSE_CLASS_DISTRIBUTION), -1).to(device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
    optim = get_optimizer(model.parameters(), args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max", patience=10)

    train_data_loader = load_dense_data(
        TRAIN_DATA_PATH,
        batch_size=16,
        transform=Compose(
            [
                ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2, hue=0.1),
                RandomHorizontalFlip(flip_prob=0.5),
                ToTensor(),
            ]
        ),
    )

    global_step = 0
    for _ in range(args.epochs):
        model.train()
        for _, (train_features, train_labels) in enumerate(train_data_loader):
            train_features, train_labels = (
                train_features.to(device),
                train_labels.to(device),
            )

            forward_output = model(train_features)
            loss = loss_function(forward_output, train_labels.long())
            train_logger.add_scalar("train/loss", loss, global_step)
            global_step += 1

            optim.zero_grad()
            loss.backward()
            optim.step()


def get_optimizer(params, lr):
    return torch.optim.SGD(params, lr, momentum=0.9, weight_decay=1e-4)
    # return torch.optim.Adam(params, lr, weight_decay=1e-4)


def compute_input_norm():
    train_data_loader = load_dense_data(TRAIN_DATA_PATH)

    x = torch.zeros(3)
    x_squared = torch.zeros(3)
    num_batches = 0.0
    for _, (train_features, _) in enumerate(train_data_loader):
        x += torch.mean(train_features, dim=[0, 2, 3])
        x_squared += torch.mean(torch.pow(train_features, 2), dim=[0, 2, 3])
        num_batches += 1.0
    mean = x_squared / num_batches
    std_dev = torch.pow((x_squared / num_batches - torch.pow(mean, 2.0)), 0.5)
    print(mean)
    print(std_dev)


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--epochs", default=1, type=int)
# parser.add_argument("-lr", "--learning_rate", default=0.01, type=float)
# parser.add_argument("-inorm", "--input_norm", default=False, type=bool)
# parser.add_argument("-log", "--log_dir", default="")
# args = parser.parse_args()
# if args.input_norm:
#     compute_input_norm()
# train(args)
