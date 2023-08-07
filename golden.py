from basic_scheduler import scheduler
from PyOpenTCU import Tensor


import numpy as np
import os


def create_matrix(size=[256, 256], scale=20, offset=10):
    return (np.random.rand(size[0], size[1]) * scale) - offset


def save_matrix(a, filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename)
    np.save(golden_path, a)


def tiling(a, b, c):
    tensor = Tensor()
    d = scheduler(a, b, c, tensor)
    save_matrix(d, "d")


def main():
    a = create_matrix()
    b = create_matrix()
    c = create_matrix()
    save_matrix(a, "a")
    save_matrix(b, "b")
    save_matrix(c, "c")
    tiling(a, b, c)


if __name__ == "__main__":
    main()
