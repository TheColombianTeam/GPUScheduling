import os
import numpy as np

import json , csv
from sfpy import *

from Kernel import faulty_Kernel_execution
import sys

from utils.args import args
format_type = str(args.config.format)

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


def injector(fault,a,b,c,d_golden,CTAs_2LRR,CTAs_GRR,CTAs_GS,CTAs_DCTA,CTAs_DB,shared_faults_Queue):
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
        shared_faults_Queue.put(    (  str(fault[0]) , int(fault[1]), int(fault[2]), int(fault[3]),
                                                    x , y, golden_faulty_values[entrance], golden_faulty_hex[entrance],
                                                        falty_values[entrance],faulty_hex[entrance] )        )


