import os
from utils.args import args

import csv
from log.logger import logger

exaustive_test_n_faults = args.faults.exaustive_test
n_fault_to_inject = args.faults.faults_to_inject


def fault_list_generation():
    faults_ID = []
    for id in range(int(n_fault_to_inject)):
        faults_ID.append(id)
    return faults_ID[:]


def save_fault_list(file_ptr, FIDs, faulty_SM, faulty_cluster):
    writer = csv.writer(file_ptr)
    Table_title = ["Scheduler Policy", "Cluster target", "SM target", "Fault ID"]
    writer.writerow(Table_title)
    for id in range(len(FIDs)):
        row = []
        row.append("TwoLevelRoundRobin")
        row.append(int(faulty_cluster))
        row.append(int(faulty_SM))
        row.append(FIDs[id])
        writer.writerow(row)
        row[0] = "GlobalRoundRobin"
        writer.writerow(row)
        row[0] = "Greedy"
        writer.writerow(row)
        row[0] = "DistributedCTA"
        writer.writerow(row)
        row[0] = "DistributedBlock"
        writer.writerow(row)


def fault_list(faulty_SM=0, faulty_cluster=0):
    FIDs = fault_list_generation()
    fault_list_file_path = os.path.join(os.getcwd(), "FaultInjector", "fault_list.csv")
    file_ptr = open(fault_list_file_path, "w+")
    save_fault_list(file_ptr, FIDs, faulty_SM, faulty_cluster)
    file_ptr.close()

    logger.waring("Fault list module completed")
