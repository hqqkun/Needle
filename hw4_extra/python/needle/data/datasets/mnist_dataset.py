import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.length = self.images.shape[0]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        labels = self.labels[index]
        if len(imgs.shape) > 1:
            # many images
            newimages = [
                self.apply_transforms(img.reshape((28, 28, 1))).reshape(img.shape)
                for img in imgs
            ]
            imgs = np.stack(newimages).reshape(imgs.shape)
        else:
            imgs = self.apply_transforms(imgs.reshape((28, 28, 1))).reshape(imgs.shape)
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.length
        ### END YOUR SOLUTION


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as file_image:
        _, nums, row, col = struct.unpack(">IIII", file_image.read(16))
        assert row == 28 and col == 28
        images = np.reshape(
            np.frombuffer(file_image.read(), dtype=np.uint8), (nums, row * col)
        )

    with gzip.open(label_filename, "rb") as file_label:
        _, _ = struct.unpack(">II", file_label.read(8))
        labels = np.frombuffer(file_label.read(), dtype=np.uint8)

    return (np.astype(images / 255.0, np.float32), labels)
