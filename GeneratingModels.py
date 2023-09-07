#THIS SCRIPT HAS NOT BEEN TESTED AND MIGHT HAVE BUGS --> WILL BE CORRECTED EVENTUALLY 
import torch
import numpy as np

from sfpy import *
from ErrorInjector import ErrorInjector
from ErrorInjector import Modellor

import os,csv, sys
import random, time

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

def FindModellor(scheduler,Modellor_2lrr,Modellor_grr,Modellor_gs,Modellor_dcta,Modellor_db):
    if scheduler == 'TwoLevelRoundRobin':
        return Modellor_2lrr
    elif scheduler == 'GlobalRoundRobin':
        return Modellor_grr
    elif scheduler == 'Greedy':
        return Modellor_gs
    elif scheduler == 'DistributedCTA':
        return Modellor_dcta
    elif scheduler == 'DistributedBlock':
        return Modellor_db
    else:
        print('Error in find modellor got :' + str(scheduler))
        sys.exit()


if __name__ == "__main__":
    n_attempts = 5 #these are the number of time masks will be tempted to be created before classifing faults as impossible to be modelled
    FaultList = read_fault_list()

    Modellor_2lrr = Modellor('TwoLevelRoundRobin',imported=False,update=True)
    Modellor_grr = Modellor('GlobalRoundRobin',imported=False,update=True)
    Modellor_greedy = Modellor('Greedy',imported=False,update=True)
    Modellor_dcta = Modellor('DistributedCTA',imported=False,update=True)
    Modellor_db = Modellor('DistributedBlock',imported=False,update=True)

    st = time.time()
    test_passes = 0
    for fault in range(len(FaultList)):
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
            
            Modellor_ = FindModellor(scheduler,Modellor_2lrr,Modellor_grr, Modellor_greedy,Modellor_dcta,Modellor_db)
            Masks = ErrorInjecto.MasksListToModellor() #list of dict( {(x,y) : 0x00ff})
            associated = Modellor_.AssignFaultToModel(Masks,Avg_model_error,
                                                      ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                                      ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()),
                                                      FID)
            if not(associated): #then i have to create a brand new model for this fault
                Modellor_.AddModel(FID,Masks,Avg_model_error,
                                   ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                   ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()))
            
            test_passes += 1

        elif(0.1 < float(AvgError/Avg_model_error) < 10 ):# the avg abs error must have same order of magnitude
            logger.warning('TEST PASSED') 
            logger.warning('Avg error introduced by fault : ' + str(AvgError))
            logger.warning('Avg error model : ' + str(Avg_model_error))
            logger.warning('Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())))
            logger.warning('Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())))
            logger.warning('Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))

            Modellor_ = FindModellor(scheduler,Modellor_2lrr,Modellor_grr, Modellor_greedy,Modellor_dcta,Modellor_db)
            Masks = ErrorInjecto.MasksListToModellor() #list of dict( {(x,y) : 0x00ff})
            associated = Modellor_.AssignFaultToModel(Masks,Avg_model_error,
                                                      ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                                      ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()),
                                                      FID)
            if not(associated): #then i have to create a brand new model for this fault
                Modellor_.AddModel(FID,Masks,Avg_model_error,
                                   ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                   ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()))

            test_passes += 1
        else:  
                logger.warning('TEST FAILED')
                logger.warning('Avg error introduced by fault : ' + str(AvgError))
                logger.warning('Avg error model : ' + str(Avg_model_error))
                logger.warning('Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())))
                logger.warning('Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())))
                logger.warning('Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())))
                logger.warning('Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))

                n = 1
                FasterErrInjct = ErrorInjector(FID,scheduler,'DummyGPU',imported=False,validation=False)#we already have TCU results --> I need only masks
                while ( n < n_attempts):
                    Faulty = FasterErrInjct.FaultyMatrixMult(a,b)
                    if (0.1 < float(AvgError/FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy())) < 10 or 
                                    AvgError == FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) == 0.0 or
                                         (str(AvgError) == 'inf' and FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) > 1000) or
                                                (str(AvgError) == 'nan' and FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) > 1000)):
                        
                        logger.warning('GENERATED MODEL FOR FAULT THAT FAILED TEST AFTHER ATTEMPTS:  '+str(n))
                        Modellor_ = FindModellor(scheduler,Modellor_2lrr,Modellor_grr, Modellor_greedy,Modellor_dcta,Modellor_db)
                        Masks = FasterErrInjct.MasksListToModellor() #list of dict( {(x,y) : 0x00ff})
                        associated = Modellor_.AssignFaultToModel(Masks,Avg_model_error,
                                                      ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                                      ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()),
                                                      FID)
                        if not(associated): #then i have to create a brand new model for this fault
                            Modellor_.AddModel(FID,Masks,Avg_model_error,
                                        ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()),
                                        ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy()))
                        test_passes += 1
                        break
                    
                    else:
                        n += 1
                
                if n == n_attempts:
                    logger.warning('CANT GENERATE A MODEL FOR FAULT ID :'+str(FID) +' , schedulers irrilevant')
                        
    end = time.time()
    print('100 faults time :  '+str((end-st)/60)+ 'min')
    
    Modellor_2lrr.StoreModels()
    Modellor_grr.StoreModels()
    Modellor_greedy.StoreModels()
    Modellor_dcta.StoreModels()
    Modellor_db.StoreModels()
    
    logger.warning('Passed ' + str(test_passes*100/15) + ' (%) of tests for 8k faults')



