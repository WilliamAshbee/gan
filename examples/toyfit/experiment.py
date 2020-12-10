from collections import OrderedDict
from DonutDataset import DonutDataset

import torchvision

from catalyst.dl import ConfigExperiment

"""
class MNIST(torchvision.datasets.MNIST):
    MNIST Dataset with key_value __get_item__ output
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        image_key="image",
        target_key="target"
    ):
    
        :param root:
        :param train:
        :param transform:
        :param target_transform:
        :param download:
        :param image_key: key to place an image
        :param target_key: key to place target
        super().__init__(root, train, transform, target_transform, download)
        self.image_key = image_key
        self.target_key = target_key

    def __getitem__(self, index: int):
        image, target = self.data[index], self.targets[index]

        dict_ = {
            self.image_key: image,
            self.target_key: target,
        }

        if self.transform is not None:
            dict_ = self.transform(dict_)
        return dict_
"""

# data loaders & transforms
class HybridGanExperiment(ConfigExperiment):
    """
    Simple Hybrid experiment
    """
    def get_datasets(
        self, stage: str, image_key: str = "image", target_key: str = "target"
    ):
        """

        :param stage:
        :param image_key:
        :param target_key:
        :return:
        """
        datasets = OrderedDict()

        for dataset_name in ("train", "valid"):
            datasets[dataset_name] = DonutDataset(1000)

        return datasets

    def get_callbacks(self, stage: str):
        callbacks = super().get_callbacks(stage=stage)
        # Workaround vs default callbacks
        for default_cb in ("_criterion", "_optimizer", "_scheduler"):
            if default_cb in callbacks:
                del callbacks[default_cb]
        return callbacks
