import torch
import numpy as np

from sfpy import *
from ErrorInjector import ErrorInjector

import os,csv, sys
import random

from log.logger import logger


def read_fault_list():
    fault_list = []
    path = os.path.join(os.getcwd(), "FaultInjector", "fault_list.csv")
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fault_list.append(row)
    fault_list.pop(0)  # deleating Table name
    return fault_list[:][:]

def AvgAbsError(Np_tensor_1, Np_tensor_2):
    AbsErrors = []
    for row ,coloms in enumerate(Np_tensor_1):
        for colom, data in enumerate(coloms):
            if abs(Np_tensor_1[row][colom] - Np_tensor_2[row][colom]) != 0.0: 
                AbsErrors.append(abs(Np_tensor_1[row][colom] - Np_tensor_2[row][colom]))
    try:
        return sum(AbsErrors)/len(AbsErrors)
    except ZeroDivisionError:
        return 0.0


if __name__ == "__main__":
    FaultList = read_fault_list()
    test_passes = 0
    for fault in range(8000):
        FID = FaultList[fault][3]
        scheduler = FaultList[fault][0]
        ErrorInjecto = ErrorInjector(FID,scheduler,'DummyGPU',imported=False,validation=True)#trying to validate error injector/ not imported in CustomDNNLayers
        
        common_dim = random.randint(120,250)
        a = torch.randn(random.randint(120,250), common_dim)
        b = torch.randn(common_dim,random.randint(120,250))
        
        golden = ErrorInjecto.Float16_MatrixMul(a,b) #GEMM_v2 --> will be called by Modellor once faults are groupped in models 
        Faulty, Avg_model_error, RealInjections = ErrorInjecto.FaultyMatrixMult(a,b) #realInjections is produced by calling TCU simulator/Faulty by apllying mask according to experiment data

        AvgError = AvgAbsError(golden.numpy(), RealInjections.numpy())
        
        if  Avg_model_error == Avg_model_error == 0.0: 
            logger.warning('TEST PASSED') 
            logger.warning('Avg error introduced by fault : ' + str(AvgError))
            logger.warning('Avg error model : ' + str(Avg_model_error))
            logger.warning('Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())))
            logger.warning('Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))
            test_passes += 1
        elif(0.1 < float(AvgError/Avg_model_error) < 10 ):# the avg abs error must have same order of magnitude
            logger.warning('TEST PASSED') 
            logger.warning('Avg error introduced by fault : ' + str(AvgError))
            logger.warning('Avg error model : ' + str(Avg_model_error))
            logger.warning('Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())))
            logger.warning('Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))

            test_passes += 1
        else:    
            logger.warning('TEST FAILED')
            logger.warning('Avg error introduced by fault : ' + str(AvgError))
            logger.warning('Avg error model : ' + str(Avg_model_error))
            logger.warning('Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())))
            logger.warning('Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))


    logger.warning('Passed ' + str(test_passes*100/8000) + ' (%) of tests for 8k faults')



