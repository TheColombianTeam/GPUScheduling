from basic_scheduler import scheduler
from PyOpenTCU import Tensor
from log.logger import logger


import numpy as np
import os


def create_matrix(size=[63, 63], scale=20, offset=10):
    return (np.random.rand(size[0], size[1]) * scale) - offset


def save_matrix(a, filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename)
    np.save(golden_path, a)


def tiling(a, b, c):
    tensor = Tensor()
    d = scheduler(a, b, c, tensor)
    save_matrix(d, "d")
    return d


def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename + ".npy")
    return np.load(golden_path)


def validate(d_golden, d):
    for row, columns in enumerate(d_golden):
        for column, golden in enumerate(columns):
            if not golden == d[row][column]:
                logger.warning(
                    f"Error: [{row}, {column}] does not have the correct value"
                )
                logger.warning(f"Expected {golden} obtained {d[row][column]}")


def main():
    a = read_matrix("a")
    b = read_matrix("b")
    c = read_matrix("c")
    d_ = tiling(a, b, c)
    d = np.matmul(a, b) + c
    validate(d, d_)


if __name__ == "__main__":
    main()
