from .models import Scheduler


import numpy as np
from utils.args import args
from log.logger import logger


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


class Mock(Scheduler.Scheduler):
    def __init__(self):
        super().__init__()
        self.__scheduler_info = []

    def scheduler_algorithm(self, a, b, c):

        self.__tiling(a, b, c)
        # TODO: Include scheduler policy
        return self.__scheduler_info

    def __complete(self, matrix,x_tiling,y_tiling):
        shape = matrix.shape
        if (shape[0] % x_tiling) > 0:
            new_shape_a = x_tiling * (shape[0] // x_tiling) + x_tiling
        else:
            new_shape_a = shape[0]
        if (shape[1] % y_tiling) > 0:
            new_shape_b = y_tiling * (shape[1] // y_tiling) + y_tiling
        else:
            new_shape_b = shape[1]
        new_shape = new_shape_a, new_shape_b
        new_matrix = np.zeros(new_shape)
        new_matrix[: shape[0], : shape[1]] = matrix
        return new_matrix

    def __tiling(self, a, b, c):
        # Initial values
        CTA_id = 0
        SM_id = 0
        cluster_id = 0
        a = self.__complete(a,MS,KS)
        b = self.__complete(b,KS,NS)
        c = self.__complete(c,MS,NS)
        c_shape = c.shape
        for row_c in range(c_shape[0] // MS):
            new_row_c_start = MS * row_c
            for column_c in range(c_shape[1] // NS):
                new_column_c_start = NS * column_c
                CTA_info = {
                    "Cluster": cluster_id,
                    "SM": SM_id,
                    "CTA": {
                        "id": CTA_id,
                        "x": new_column_c_start,
                        "y": new_row_c_start,
                    },
                }
                self.__create_dict(CTA_info)
                cluster_id += 1
                SM_id += 1

    def __create_dict(self, CTA_info):
        self.__scheduler_info.append(CTA_info)
