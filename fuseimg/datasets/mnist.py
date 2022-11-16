"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""
from typing import Optional, Sequence
from fuse.data.datasets.dataset_wrap_seq_to_dict import DatasetWrapSeqToDict
from torchvision import transforms
from torch.utils.data import Subset, Dataset

import torchvision
import torch

from fuse.data import DatasetDefault

from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

class MNIST:
    """
    FuseMedML style of MNIST dataset: http://yann.lecun.com/exdb/mnist/
    """

    # bump whenever the static pipeline modified
    MNIST_DATASET_VER = 0

    @staticmethod
    def dataset(cache_dir: Optional[str] = None, train: bool = True, sample_ids: Sequence = None) -> DatasetDefault:
        """
        Get mnist dataset - each sample includes: 'data.image', 'data.label' and 'data.sample_id'
        :param cache_dir: optional - destination to cache mnist
        :param train: If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        :param sample_ids: Optional list of sample ids. If None, then all data is used.
        """

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # Create dataset
        torch_train_dataset = torchvision.datasets.MNIST(
            cache_dir, download=cache_dir is not None, train=train, transform=transform
        )
        # wrapping torch dataset
        if sample_ids is not None:
            torch_train_dataset = Subset(torch_train_dataset, sample_ids)
        train_str = "train" if train else "test"
        train_dataset = DatasetWrapSeqToDict(
            name=f"mnist-{train_str}", dataset=torch_train_dataset, sample_keys=("data.image", "data.label")
        )
        train_dataset.create()
        return train_dataset

class SHATZ_MNIST_CUSTOM_DATASET(torchvision.datasets.MNIST):
    """
    custom implementation of torchvision.datasets.MNIST for the purposes of also appending a string to each yielded image
    for multimodal testing.
    """

    # just override __getitem__
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        mod2 = torch.nn.functional.one_hot(torch.tensor(target), num_classes=64)
        
        return img, mod2, target


class SHATZ_MNIST:
    """
    FuseMedML style of MNIST dataset: http://yann.lecun.com/exdb/mnist/
    """

    # bump whenever the static pipeline modified
    MNIST_DATASET_VER = 0

    @staticmethod
    def dataset(cache_dir: Optional[str] = None, train: bool = True, sample_ids: Sequence = None) -> DatasetDefault:
        """
        Get mnist dataset - each sample includes: 'data.image', 'data.label' and 'data.sample_id'
        :param cache_dir: optional - destination to cache mnist
        :param train: If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        :param sample_ids: Optional list of sample ids. If None, then all data is used.
        """

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # Create dataset
        torch_train_dataset = SHATZ_MNIST_CUSTOM_DATASET(
            cache_dir, download=cache_dir is not None, train=train, transform=transform
        )
        # wrapping torch dataset
        if sample_ids is not None:
            torch_train_dataset = Subset(torch_train_dataset, sample_ids)
        train_str = "train" if train else "test"
        train_dataset = DatasetWrapSeqToDict(
            name=f"mnist-{train_str}", dataset=torch_train_dataset, sample_keys=("data.image", "data.mod2", "data.label")
        )
        train_dataset.create()
        return train_dataset
