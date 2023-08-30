import numpy as np


from utils.args import args


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


def scheduler(a, b, c, tensor):
    c_shape_original = c.shape
    a = complete(a)
    b = complete(b)
    c = complete(c)
    a_shape = a.shape
    b_shape = b.shape
    c_shape = c.shape
    for row_c in range(c_shape[0] // MS):  # -
        new_row_c_start = MS * row_c  # ----- X
        new_row_c_end = new_row_c_start + MS
        for column_c in range(c_shape[1] // NS):  # -
            new_column_c_start = NS * column_c  # ----- Y
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
                c_block = scheduler_sm(a_block, b_block, c_block, tensor)
                c[
                    new_row_c_start:new_row_c_end, new_column_c_start:new_column_c_end
                ] = c_block
    return c[: c_shape_original[0], : c_shape_original[1]]
