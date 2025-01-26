import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None,
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        if train:
            data_batch_files = [
                os.path.join(base_folder, f"data_batch_{i}") for i in range(1, 6)
            ]
        else:
            data_batch_files = [os.path.join(base_folder, "test_batch")]
        X = []
        Y = []
        for data_batch_file in data_batch_files:
            data_dict = unpickle(data_batch_file)
            X.append(data_dict[b"data"])
            Y.append(data_dict[b"labels"])

        X = np.concatenate(X, axis=0)

        X = X / 255.0
        X = X.reshape((-1, 3, 32, 32))
        Y = np.concatenate(Y, axis=None)
        self.X = X
        self.Y = Y
        self.transforms = transforms
        self.length = len(self.Y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.Y[index]
        if len(imgs.shape) > 3:
            # many images
            newimages = [
                self.apply_transforms(img.transpose((1, 2, 0))).transpose((2, 0, 1))
                for img in imgs
            ]
            imgs = np.stack(newimages).reshape(imgs.shape)
        else:
            imgs = self.apply_transforms(imgs.transpose((1, 2, 0))).transpose((2, 0, 1))
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.length
        ### END YOUR SOLUTION


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
