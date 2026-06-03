import os
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir="mnist_data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    file_paths = {}

    for name, url in MNIST_URLS.items():
        filename = url.split("/")[-1]
        path = data_dir / filename
        file_paths[name] = path

        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{filename} already exists.")

    return file_paths


def read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"Invalid image file magic number: {magic}")

        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows, cols)

    return images


def read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"Invalid label file magic number: {magic}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_mnist_numpy(data_dir="mnist_data"):
    paths = download_mnist(data_dir)

    X_train = read_idx_images_gz(paths["train_images"])
    Y_train = read_idx_labels_gz(paths["train_labels"])

    X_test = read_idx_images_gz(paths["test_images"])
    Y_test = read_idx_labels_gz(paths["test_labels"])

    # (60000, 28, 28) -> (60000, 784)
    X_train = X_train.reshape(-1, 28 * 28).astype(float) / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype(float) / 255.0

    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    return X_train, Y_train, X_test, Y_test