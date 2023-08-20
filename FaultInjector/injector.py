import os
import numpy as np

import json , csv
from sfpy import *

from Kernel import faulty_Kernel_execution
import sys

from multiprocessing import Process
from multiprocessing.managers import BaseManager

from utils.args import args
format_type = str(args.config.format)

CTAs_2LRR = None #they are global since they must be read in the worker function --> in order to reduce number of args to pass
CTAs_GRR = None
CTAs_GS = None
CTAs_DTCA = None
CTAs_DB = None

class CustomMenager(BaseManager):
    pass

def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename + ".npy")
    return np.load(golden_path)

def read_json(scheduling_policy):
    path = os.path.join(os.getcwd(),'Schedulers','scheduled', scheduling_policy + '.json')
    with open(path,  encoding='utf-8') as json_file:
        CTAs = json.load(json_file)

    return CTAs

def read_fault_list():
    fault_list = []
    path = os.path.join(os.getcwd(), 'FaultInjector', 'fault_list.csv')
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fault_list.append(row)
    fault_list.pop(0)#deleating Table name
    return fault_list[:][:]

def corrupted_etrances_extrapolation(d,d_golden):
    global format_type

    faulty_entrances = []
    faulty_values = []
    golden_falty_values = []
    faulty_hex = []
    golden_faulty_hex = []

    for row, columns in enumerate(d_golden):
        for column, golden in enumerate(columns):
            if abs( float(golden) - float(d[row][column])) > 0.0:
                faulty_entrances.append((row,column))
                faulty_values.append(d[row][column])
                golden_falty_values.append(golden)
                
                if(format_type == 'float16'):
                    golden_format = Float16(golden)
                    faulty_format = Float16(d[row][column])
                    bits_golden = golden_format.bits
                    bits_faulty = faulty_format.bits
                    golden_faulty_hex.append( hex(bits_golden) )
                    faulty_hex.append( hex(bits_faulty))
                else:
                    print("Not implemented data fromat, dying")
                    sys.exit()
                    
    return faulty_entrances[:],faulty_values[:],golden_falty_values[:],faulty_hex[:],golden_faulty_hex[:]


def fi(fault,a,b,c,d_golden,shared_fault_result):
    global CTAs_2LRR
    global CTAs_GRR
    global CTAs_GS
    global CTAs_DCTA
    global CTAs_DB

    #decoding fault
    if(str(fault[0]) == 'TwoLevelRoundRobin'):
        CTA = CTAs_2LRR
    elif(str(fault[0]) == 'GlobalRoundRobin'):
        CTA = CTAs_GRR
    elif(str(fault[0]) == 'Greedy'):
        CTA = CTAs_GS
    elif(str(fault[0]) == 'DistributedBlock'):
        CTA = CTAs_DB
    elif(str(fault[0]) == 'DistributedCTA'):
        CTA = CTAs_DCTA
    else:
        print('error while decoding fault got :' + str(fault[0]))
        sys.exit()
    
    d_faulty = faulty_Kernel_execution(a,b,c,d_golden,CTA,int(fault[1]), int(fault[2]), int(fault[3]))
    faulty_entrances,falty_values,golden_faulty_values,faulty_hex,golden_faulty_hex = corrupted_etrances_extrapolation(d_faulty,d_golden)

    #shared fault list is a set of tuples (scheduler,cluster_target,SM_target,faultID,x,y,golden, golden_hex,faulty, faulty_hex)
    for entrance in range(len(faulty_entrances)):
        x,y = faulty_entrances[entrance]
        shared_fault_result.add(    (  str(fault[0]) , int(fault[1]), int(fault[2]), int(fault[3]),
                                                    x , y, golden_faulty_values[entrance], golden_faulty_hex[entrance],
                                                        falty_values[entrance],faulty_hex[entrance] )        )

def save_results(faults_set):
    path = os.path.join(os.getcwd(), 'FaultInjector', 'results.csv')
    file_ptr = open(path, 'w+')
    writer = csv.writer(file_ptr)
    Table_Title = ['Scheduling Policy', 'ClusterTarget', 'SMTarget', 'faultID', 'X', 'Y', 'Golden', 'GoldenHexa', 'Faulty', ' FaultyHexa']
    writer.writerow(Table_Title)
    for row in range(len(faults_set)):
        to_write = []
        sp,cl,sm,FID,x,y,gol,gol_hex,faul,faul_hex = faults_set[row]
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
        
    file_ptr.close()
    

def injector(number_of_workers):
    global CTAs_2LRR
    global CTAs_GRR
    global CTAs_GS
    global CTAs_DCTA
    global CTAs_DB

    if(number_of_workers <= 0):
        print(" Error number of process set through number of workers must be bigger then 0")
        sys.exit()

    a = read_matrix('a')
    b = read_matrix('b')
    c = read_matrix('c')
    d = read_matrix('d')

    CTAs_2LRR = read_json('two_level_round_robin')
    CTAs_GRR = read_json('global_round_robin')
    CTAs_GS = read_json('greddy')
    CTAs_DCTA = read_json('distributed_CTA')
    CTAs_DB = read_json('distributed_block')

    fault_list = read_fault_list()
    n_faults_to_inject = len(fault_list) #10
    injected_faults = 0

    # register the counter with the custom manager
    CustomMenager.register('set', set)
    # create a new manager instance
    with CustomMenager() as manager :
        shared_fault_result = manager.set()

        while( injected_faults < n_faults_to_inject):
            if((n_faults_to_inject - injected_faults) > number_of_workers):
                processes = [Process(target=fi, args=(fault_list[injected_faults + w],a,b,c,d,shared_fault_result)) for w in range(number_of_workers)]
                # start processes
            else:
                processes = [Process(target=fi, args=(fault_list[injected_faults + w],a,b,c,d,shared_fault_result)) for w in range(int(n_faults_to_inject-injected_faults))]
            #starting processes
            for process in processes:
                process.start()
            # wait for processes to finish
            for process in processes:
                process.join()
            
            #HERE WE CAN ALSO INSERT A BREAKPOINT THAT STORES PARTIAL RESULTS EVERY 500 INJECTED FAULTS TO BE SURE TO NOT LOSE DATA DUE TO SYSTEM CRASH
            injected_faults += number_of_workers

        save_results(list(shared_fault_result._getvalue()))
    
    print(" Fault injector module completed")
