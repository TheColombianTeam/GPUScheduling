from basic_scheduler import scheduler
from PyOpenTCU import Tensor
from log.logger import logger


import numpy as np
import os


def create_matrix(size=[120, 120], scale=20, offset=10):
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
    a = create_matrix()
    b = create_matrix()
    c = create_matrix()
    d_ = tiling(a, b, c)
    d = np.matmul(a, b) + c
    validate(d, d_)
    save_matrix(a,"a.npy")
    save_matrix(b,"b.npy")
    save_matrix(c,"c.npy")
    save_matrix(d_,"d.npy")



if __name__ == "__main__":
    main()
