from PyOpenTCU import Tensor


import numpy as np

from utils.args import args


MS = args.mxm.MS
NS = args.mxm.NS
KS = args.mxm.KS


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
