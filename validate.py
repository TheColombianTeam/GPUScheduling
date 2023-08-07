from PyOpenTCU import Tensor


import numpy as np
import os


from utils.args import args
from log.logger import logger


import Schedulers as schedulers_list


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


def complete(matrix):
    shape = matrix.shape
    if (shape[0] % MS) > 0:
        new_shape_a = MS * (shape[0] // MS) + MS
    else:
        new_shape_a = shape[0]
    if (shape[1] % NS) > 0:
        new_shape_b = NS * (shape[1] // NS) + NS
    else:
        new_shape_b = shape[1]
    new_shape = new_shape_a, new_shape_b
    new_matrix = np.zeros(new_shape)
    new_matrix[: shape[0], : shape[1]] = matrix
    return new_matrix


def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename + ".npy")
    return np.load(golden_path)


# This method represent the warp scheduler - NOT CHANGES
def scheduler_sm(a, b, c, tensor):
    c_shape_original = c.shape
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    for row_c in range(c_shape[0] // KS):
        new_row_c_start = KS * row_c
        new_row_c_end = KS * row_c + KS
        for column_c in range(c_shape[1] // KS):
            new_column_c_start = KS * column_c
            new_column_c_end = KS * column_c + KS
            new_row_a_start = new_row_c_start
            new_row_a_end = new_row_c_end
            new_column_b_start = new_column_c_start
            new_column_b_end = new_column_c_end
            c_tensor = np.zeros([KS, KS])
            for column_a in range(a_shape[1] // KS):
                new_column_a_start = KS * column_a
                new_column_a_end = KS * column_a + KS
                new_row_b_start = new_column_a_start
                new_row_b_end = new_column_a_end
                a_tensor = a[
                    new_row_a_start:new_row_a_end, new_column_a_start:new_column_a_end
                ]
                b_tensor = b[
                    new_row_b_start:new_row_b_end, new_column_b_start:new_column_b_end
                ]
                c_tensor = c[
                    new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end
                ]
                c_tensor = tensor.mul(a_tensor, b_tensor, c_tensor)
                c[
                    new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end
                ] = c_tensor
    return c[: c_shape_original[0], : c_shape_original[1]]


def gpu_kernel_execution(a, b, c, CTAs, tensor):
    a_shape = a.shape
    for CTA in CTAs:
        cluster = CTA["Cluster"]
        sm = CTA["SM"]
        CTA_id = CTA["CTA"]["id"]
        new_row_c_start = CTA["CTA"]["x"]
        new_row_c_end = new_row_c_start + MS

        new_column_c_start = CTA["CTA"]["y"]
        new_column_c_end = new_column_c_start + NS

        new_row_a_start = new_row_c_start
        new_row_a_end = new_row_c_end
        new_column_b_start = new_column_c_start
        new_column_b_end = new_column_c_end

        c_block = np.zeros([MS, NS])

        for column_a in range(a_shape[1] // KS):
            new_column_a_start = KS * column_a
            new_column_a_end = KS * column_a + KS
            new_row_b_start = new_column_a_start
            new_row_b_end = new_column_a_end
            a_block = a[
                new_row_a_start:new_row_a_end, new_column_a_start:new_column_a_end
            ]
            b_block = b[
                new_row_b_start:new_row_b_end, new_column_b_start:new_column_b_end
            ]
            c_block = c[
                new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end
            ]
            #logger.info(f"A {a_block.shape} B {b_block.shape} C {c_block.shape}")
            c_block = scheduler_sm(a_block, b_block, c_block, tensor)
            c[
                new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end
            ] = c_block
    return c

def tiling(a, b, c):
    a = complete(a)
    b = complete(b)
    c = complete(c)
    tensor = Tensor()
    scheduler = getattr(schedulers_list, args.scheduler)()
    CTAs = scheduler.scheduler_algorithm(a, b, c)
    d = gpu_kernel_execution(a, b, c, CTAs, tensor)
    return d


def validate(d_golden, d):
    equals = True
    for row, columns in enumerate(d_golden):
        for column, golden in enumerate(columns):
            if not golden == d[row][column]:
                logger.warning(
                    f"Error: [{row}, {column}] does not have the correct value"
                )
                logger.warning(f"Expected {golden} obtained {d[row][column]}")
                equals = False
    if equals:
        logger.info(f"The tiling designed for {args.scheduler} scheduler works correct")


def main():
    a = read_matrix("a")
    b = read_matrix("b")
    c = read_matrix("c")
    d_golden = read_matrix("d")
    d = tiling(a, b, c)
    validate(d_golden, d)


if __name__ == "__main__":
    main()
