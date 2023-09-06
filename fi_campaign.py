from FaultInjector import golden
from FaultInjector import fault_list
from FaultInjector import injector
from FaultInjector import validator

from utils.args import args
import numpy as np

import os, sys, time
import csv, json

import multiprocessing as mp
from log.logger import logger

workers = int(args.workers)


def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename + ".npy")
    return np.load(golden_path)


def read_json(scheduling_policy):
    path = os.path.join(
        os.getcwd(), "Schedulers", "scheduled", scheduling_policy + ".json"
    )
    with open(path, encoding="utf-8") as json_file:
        CTAs = json.load(json_file)
    return CTAs


def read_fault_list():
    fault_list = []
    path = os.path.join(os.getcwd(), "FaultInjector", "fault_list.csv")
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fault_list.append(row)
    fault_list.pop(0)  # deleating Table name
    return fault_list[:][:]


def save_results(shared_faults_Queue):
    path = os.path.join(os.getcwd(), "FaultInjector", "results.csv")
    file_ptr = open(path, "w+")
    writer = csv.writer(file_ptr)
    Table_Title = [
        "Scheduling Policy",
        "ClusterTarget",
        "SMTarget",
        "faultID",
        "X",
        "Y",
        "Golden",
        "GoldenHexa",
        "Faulty",
        " FaultyHexa",
    ]
    writer.writerow(Table_Title)
    file_ptr.close()
    not_Done = True

    with open(path, "a") as f:
        writer = csv.writer(f)
        while not_Done:
            message = shared_faults_Queue.get()
            if str(message) == "#done#":
                not_Done = False
            else:
                to_write = []
                sp, cl, sm, FID, x, y, gol, gol_hex, faul, faul_hex = message
                to_write.append(sp)
                to_write.append(cl)
                to_write.append(sm)
                to_write.append(FID)
                to_write.append(x)
                to_write.append(y)
                to_write.append(gol)
                to_write.append(gol_hex)
                to_write.append(faul)
                to_write.append(faul_hex)
                writer.writerow(to_write)
                f.flush()


if __name__ == "__main__":

    golden(120, 120, 120)
    fault_list()
    # now I'm preparing the env to support multiprocessing with threads accessing same file for ouput storing through a Queue
    if workers <= 0:
        logger.warning(
            " Error number of process set through number of workers must be bigger then 0"
        )
        sys.exit()

    a = read_matrix("a")
    b = read_matrix("b")
    c = read_matrix("c")
    d = read_matrix("d")

    CTAs_2LRR = read_json("two_level_round_robin")
    CTAs_GRR = read_json("global_round_robin")
    CTAs_GS = read_json("greddy")
    CTAs_DCTA = read_json("distributed_CTA")
    CTAs_DB = read_json("distributed_block")

    faults_list = read_fault_list()
    n_faults_to_inject = len(faults_list)
    injected_faults = 0

    manager = mp.Manager()
    shared_faults_Queue = manager.Queue()

    storing_results_pool = mp.Pool(1)
    storing_results_pool.apply_async(save_results, (shared_faults_Queue,))

    beg = time.time()
    injector_pool = mp.Pool(workers)
    injector_jobs = []
    while injected_faults < n_faults_to_inject:
        injector_jobs.append(
            injector_pool.apply_async(
                injector,
                (
                    faults_list[injected_faults],
                    a,
                    b,
                    c,
                    d,
                    CTAs_2LRR,
                    CTAs_GRR,
                    CTAs_GS,
                    CTAs_DCTA,
                    CTAs_DB,
                    shared_faults_Queue,
                ),
            )
        )
        injected_faults += 1

    for job in range(len(injector_jobs)):
        injector_jobs[job].get()  # Waiting that all fault injection have been performed

    shared_faults_Queue.put(
        "#done#"
    )  # exiting infinite loop in process writing file with data from Queue

    end = time.time()
    logger.warning(
        "ellapsed time for fault injections campaign: "
        + str((end - beg) / 60)
        + " min "
    )
    injector_pool.close()
    injector_pool.join()

    storing_results_pool.close()
    storing_results_pool.join()

    logger.warning("Fault injector campaign completed")

    validator(
        120, 120
    )  # this function needs output tensor dimentions inposed in golden for heat map bins
