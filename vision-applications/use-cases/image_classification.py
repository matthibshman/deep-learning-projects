from utils import load_data

import torch
import torchvision
import torch.utils.tensorboard as tb
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
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

    def __init__(
        self,
        layer_sizes=[32, 64, 128],
        input_channels=3,
        num_classes=6,
        padding=3,
        stride=2,
        dropout_rate=0.1,
        input_transforms=torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[0.1688, 0.1590, 0.1803], std=[0.3746, 0.3657, 0.3845]
                )
            ]
        ),
    ):
        super().__init__()

        self.HEIGHT_DIM = 2
        self.WIDTH_DIM = 3
        self.input_transforms = input_transforms

        output_channels = 32
        layers = [
            torch.nn.Conv2d(
                input_channels, output_channels, kernel_size=7, padding=padding, stride=stride
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
        ]

        for channels in layer_sizes:
            layers.append(self.Block(output_channels, channels, stride=stride))
            output_channels = channels

        self.layers = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(output_channels, num_classes)
        self.classifier.weight.zero_
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        @x: Tensor((Batch,3,64,64))
        @return: Tensor((Batch,6))
        """
        x = self.input_transforms(x)
        return self.classifier(
            self.dropout(self.layers(x).mean(dim=[self.HEIGHT_DIM, self.WIDTH_DIM]))
        )


def train(args):
    from os import path

    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, "train"), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, "valid"), flush_secs=1)

    model = model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "max", patience=10)
    epochs = args.epochs

    train_data_loader = load_data(
        "data/train",
        img_transform=transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomVerticalFlip(p=0.6),
                transforms.RandomCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )
    valid_data_loader = load_data("data/valid")

    if args.input_norm:
        compute_input_norm(train_data_loader)

    global_step = 0
    best_accuracy = 0.93
    for _ in range(epochs):
        model.train()
        for _, (train_features, train_labels) in enumerate(train_data_loader):
            train_features, train_labels = (
                train_features.to(device),
                train_labels.to(device),
            )
            forward_output = model(train_features)
            loss = loss_function(forward_output, train_labels)
            train_logger.add_scalar("train/loss", loss, global_step)
            global_step += 1

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_logger.add_scalar(
            "train/accuracy", compute_accuracy(train_data_loader, model), global_step=global_step
        )
        validation_accuracy = compute_accuracy(valid_data_loader, model)
        valid_logger.add_scalar("valid/accuracy", validation_accuracy, global_step=global_step)
        lr_scheduler.step(validation_accuracy)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            # save_model(model)

    # save_model(model)


def compute_accuracy(data_loader, model):
    model.eval()
    with torch.no_grad():
        train_accuracy = 0
        num_batches = 0
        for _, (train_features, train_labels) in enumerate(data_loader):
            train_features, train_labels = (
                train_features.to(device),
                train_labels.to(device),
            )
            num_batches += 1
            forward_output = model(train_features)
            train_accuracy += torch.sum(
                (
                    train_labels == torch.argmax(torch.nn.Softmax(dim=1)(forward_output), dim=1)
                ).long()
            ) / len(train_labels)

    return train_accuracy / num_batches


def compute_input_norm(train_data_loader):
    x = torch.zeros(3)
    x_squared = torch.zeros(3)
    num_batches = 0.0
    for _, (train_features, _) in enumerate(train_data_loader):
        x += torch.mean(train_features, dim=[0, 2, 3])
        x_squared += torch.mean(torch.pow(train_features, 2), dim=[0, 2, 3])
        num_batches += 1.0
    mean = x / num_batches
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
# train(args)
