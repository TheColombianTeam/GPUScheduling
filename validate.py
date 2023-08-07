from basic_scheduler import scheduler
from PyOpenTCU import Tensor


import numpy as np
import os


from utils.args import args
from log.logger import logger


import Schedulers as schedulers_list


def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename)
    return np.load(golden_path)


def tiling(a, b, c):
    tensor = Tensor()
    """
    TODO: Replace this function for the scheduler technique
    like:
        scheduler = getattr(schedulers_list, args.scheduler)() 
    """
    return scheduler(a, b, c, tensor) 


def main():
    a = read_matrix('a')
    b = read_matrix('b')
    c = read_matrix('c')
    d = tiling(a, b, c)


if __name__ == "__main__":
    main()
