from datasets import SuperTuxDataset, DenseSuperTuxDataset, DetectionSuperTuxDataset

from torch.utils.data import DataLoader


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True
    )


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True
    )


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True
    )
