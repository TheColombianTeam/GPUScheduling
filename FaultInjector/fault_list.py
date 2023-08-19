import os
import numpy as np


import random
from utils.args import args 
from log.logger import logger

import sys
import csv

bias = 380

exaustive_test_n_faults = args.faults.exaustive_test
n_fault_to_inject = args.faults.faults_to_inject
faulty_cluster = args.gpu.faulty_cluster
faulty_SM = args.gpu.faulty_SM

def SFI(seed):
    SFI = []
    sg = np.random.SeedSequence(seed)
    for s in sg.spawn(n_fault_to_inject):
        rg = np.random.Generator(np.random.MT19937(s))
        SFI.append(int(rg.integers(low=bias,high=exaustive_test_n_faults,size= 1)))
    return SFI[:]

def validation_SFI(list):
    test_failed = False
    n_equal = 0
    already_injected = []
    for fault in range(len(list)):
        if(test_failed):
            break
        if(int(list[fault]) < 0 or int(list[fault]) > exaustive_test_n_faults):
            logger.warning(
                    f"Error: [fault :{list[fault]} is not a fault that can be injected in TCU "
                )
            test_failed = True
        found = False
        for f in range(len(already_injected)):
            if(int(list[fault]) == int(already_injected[f])):
                n_equal += 1
                found = True        
        
        if(not(found)):  
            already_injected.append(int(list[fault]))
        
        if( n_equal > bias):
            test_failed = True      
    return test_failed,already_injected[:],n_equal


def compensation(list, n_fault_to_compensate): 
    for f in range(n_fault_to_compensate):
        list.append(f)
    return list[:]

def save_fault_list(file_ptr,FIDs):
    writer = csv.writer(file_ptr)
    Table_title = ["Scheduler Policy", "Cluster target",'SM target','Fault ID']
    writer.writerow(Table_title)
    for id in range(len(FIDs)):
        row = []
        row.append("TwoLevelRoundRobin")
        row.append(int(faulty_cluster))
        row.append(int(faulty_SM))
        row.append(FIDs[id])
        writer.writerow(row)
        row[0] = 'GlobalRoundRobin'
        writer.writerow(row)
        row[0] = 'Greedy'
        writer.writerow(row)
        row[0] = 'DistributedCTA'
        writer.writerow(row)
        row[0] = 'DistributedBlock'
        writer.writerow(row)
    
    file_ptr.close()

def fault_list():
    #I'm randomly generating the FIDs but will inject same faults for each scheduling policy
    # Therefore is going to be possible to observe the effect of the same fault across different Schedulers
    failed_validation = True
    while(failed_validation):
        FIDs = SFI(random.randint(0, 10000))
        failed_validation,FIDs, n_fault_to_compensate = validation_SFI(FIDs)

    #it's very hard to generate 8k differnt numbers randomly --> we have to deterministically ensure that we are injecting different faults
    FIDs = compensation(FIDs,n_fault_to_compensate)
    fault_list_file_path = os.path.join(os.getcwd(), "FaultInjector", "fault_list.csv")
    file_ptr = open(fault_list_file_path, "w+")
    save_fault_list(file_ptr, FIDs)

    print("Fault list module completed") 
    