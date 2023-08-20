import numpy as np
import os, csv, json


def complete(matrix, x_tiling, y_tiling):
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


def save_csv(scheduler_info, filename):
    file_path = os.path.join(os.getcwd(), "Schedulers", "scheduled", f"{filename}.csv")
    file_ptr = open(file_path, "w+")
    writer = csv.writer(file_ptr)
    Table_title = ["CTA_id", "Cluster", "SM", "x", "y"]
    writer.writerow(Table_title)
    for CTA in range(len(scheduler_info)):
        row = []
        row.append(str(CTA))
        row.append(str(scheduler_info[CTA]["Cluster"]))
        row.append(str(scheduler_info[CTA]["SM"]))
        row.append(str(scheduler_info[CTA]["CTA"]["x"]))
        row.append(str(scheduler_info[CTA]["CTA"]["y"]))
        writer.writerow(row)
    file_ptr.close()

def save_json(scheduler_info, filename):
    file_path = os.path.join(os.getcwd(), "Schedulers", "scheduled", f"{filename}.json")
    with open(file_path, "w+") as JSONfile:
            json.dump(scheduler_info,JSONfile)        
    JSONfile.close()
    
