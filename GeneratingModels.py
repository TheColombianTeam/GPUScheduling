#THIS SCRIPT HAS NOT BEEN TESTED AND MIGHT HAVE BUGS --> WILL BE CORRECTED EVENTUALLY 
import torch
import numpy as np

from ErrorInjector import ErrorInjector
from ErrorInjector import Modellor

import os,csv, json
import random, time

from log.logger import logger
from utils.args import args

import multiprocessing as mp
from sfpy import *


n_attempts = 8 #these are the number of time masks will be tempted to be created before classifing faults as impossible to be modelled
Lock = mp.Lock()

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

def EvaluateAndAssignModel(Shared_Models_Queue):
    Modellor_2lrr = Modellor('TwoLevelRoundRobin',imported=False,update=True)
    Modellor_grr = Modellor('GlobalRoundRobin',imported=False,update=True)
    Modellor_greedy = Modellor('Greedy',imported=False,update=True)
    Modellor_dcta = Modellor('DistributedCTA',imported=False,update=True)
    Modellor_db = Modellor('DistributedBlock',imported=False,update=True)

    not_Done = True
    while not_Done:

        message = Shared_Models_Queue.get()

        if str(message) == '#DONE#':  
            not_Done = False
            print('Estoring')
            Modellor_2lrr.StoreModels()
            Modellor_grr.StoreModels()
            Modellor_greedy.StoreModels()
            Modellor_dcta.StoreModels()
            Modellor_db.StoreModels()
        else:
            
            FID,scheduler,Masks,MaxAbsErr,MinAbsErr,AvgAbsErr = message
            Modellor_ = FindModellor(scheduler,Modellor_2lrr,Modellor_grr, Modellor_greedy,Modellor_dcta,Modellor_db)
            associated = Modellor_.AssignFaultToModel(Masks,AvgAbsErr,
                                                      MaxAbsErr,
                                                      MinAbsErr,
                                                      FID)
            if not(associated): #then i have to create a brand new model for this fault
                Modellor_.AddModel(FID,Masks,AvgAbsErr,
                                   MaxAbsErr,
                                   MinAbsErr)
            
def GenerateModel(FID,scheduler,UnModellableFaults,Shared_Models_Queue):
    n_attempts = 8
    global Lock

    common_dim = random.randint(120,250)
    a = torch.randn(random.randint(120,250), common_dim)
    b = torch.randn(common_dim,random.randint(120,250))
    ErrorInjecto = ErrorInjector(FID,scheduler,str(args.gpu.name),imported=False,validation=True)#trying to validate error injector/ not imported in CustomDNNLayers
    golden = ErrorInjecto.Float16_MatrixMul(a,b) #GEMM_v2 --> will be called by Modellor once faults are groupped in models 
    Faulty, Avg_model_error, RealInjections = ErrorInjecto.FaultyMatrixMult(a,b) #realInjections is produced by calling TCU simulator/Faulty by apllying mask according to experiment data
    AvgError = AvgAbsError(golden.numpy(), RealInjections.numpy())
    if  AvgError == Avg_model_error == 0.0 or 0.1 < float(AvgError/Avg_model_error) < 10 :
        logger.warning(
            'TEST PASSED \n Avg error introduced by fault : ' + str(AvgError) + 
            '\n Avg error model : ' + str(Avg_model_error) + 
            ' \n Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())) + 
            ' \n Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())) +
            ' \n Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy()))+
            ' \n Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))
        Masks = ErrorInjecto.MasksListToModellor() 
        MaxAbsErr = ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())
        MinAbsErr = ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())

        for CTA in range(len(Masks)):
            for coordinate in Masks[CTA]:
                Masks[CTA][coordinate] = hex(Masks[CTA][coordinate].bits)

        Shared_Models_Queue.put(
            (FID,
            scheduler,
            Masks,
            MaxAbsErr,
            MinAbsErr,
            Avg_model_error)
        )
    else: 
        logger.warning('TEST FAILED \n Avg error introduced by fault : ' + str(AvgError) + 
                ' \n Avg error model : ' + str(Avg_model_error) + 
                '\n Min Error model : ' + str(ErrorInjecto.MinAbsError(Faulty.numpy(),golden.numpy())) + 
                '\n Min Error introduced by fault : ' + str(ErrorInjecto.MinAbsError(RealInjections.numpy(),golden.numpy())) + 
                '\n Max Error model : ' + str(ErrorInjecto.MaxAbsError(Faulty.numpy(),golden.numpy())) +
                '\n Max Error introduced by fault : ' + str(ErrorInjecto.MaxAbsError(RealInjections.numpy(),golden.numpy())))
        n = 1
        FasterErrInjct = ErrorInjector(FID,scheduler,str(args.gpu.name),imported=False,validation=False)
        while ( n < n_attempts):
                Faulty = FasterErrInjct.FaultyMatrixMult(a,b)
                if (0.1 < float(AvgError/FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy())) < 10 or 
                                    AvgError == FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) == 0.0 or
                                         (str(AvgError) == 'inf' and FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) > 1000) or
                                                (str(AvgError) == 'nan' and FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy()) > 1000)):
                        

                    Masks = FasterErrInjct.MasksListToModellor() #list of dict( {(x,y) : 0x00ff})
                    MaxAbsErr = FasterErrInjct.MaxAbsError(Faulty.numpy(),golden.numpy())
                    MinAbsErr = FasterErrInjct.MinAbsError(Faulty.numpy(),golden.numpy())
                    Avg_model_error = FasterErrInjct.AvgAbsError(Faulty.numpy(),golden.numpy())

                    for CTA in range(len(Masks)):
                        for coordinate in Masks[CTA]:
                            Masks[CTA][coordinate] = hex(Masks[CTA][coordinate].bits) #Queue doesnt support Float16 datatype

                    Shared_Models_Queue.put(
                                        (FID,
                                        scheduler,
                                        Masks,
                                        MaxAbsErr,
                                        MinAbsErr,
                                        Avg_model_error)
                                        )
                        
                    break
                    
                else:
                    n += 1
                
                if n == n_attempts:
                    logger.warning('CANT GENERATE A MODEL FOR FAULT ID :'+str(FID) +' , scheduler : ' + str(scheduler))
                    Lock.acquire()
                    UnModellableFaults.append(
                        ( int(FID), scheduler)
                    )
                    Lock.release()

def storeUnModellableFaults(UnModellableFaults):
    path = os.path.join(os.getcwd(),'ErrorInjector', 'Models', 'UnModellableFaults.json')
    with open(path, "w+") as JSONfile:
        json.dump(UnModellableFaults, JSONfile)

if __name__ == "__main__":
    workers = int(args.workers)
    UnModellableFaults = []
    FaultList = read_fault_list()
    manager = mp.Manager()
    Shared_Models_Queue = manager.Queue()

    EvaluatingAndAssignPool = mp.Pool(1)
    EvaluatingAndAssignPool.apply_async( EvaluateAndAssignModel , (Shared_Models_Queue,))

    GenerateModelsPool = mp.Pool(workers) 
    jobs = []   
    st = time.time()
    for fault in range(132):#len(FaultList)
        FID = FaultList[fault][3]
        scheduler = FaultList[fault][0]
        jobs.append(
            GenerateModelsPool.apply_async(
                GenerateModel,
                (int(FID),
                 scheduler,
                 UnModellableFaults,
                 Shared_Models_Queue)
            )
        )

    
    for j in range(len(jobs)):
        jobs[j].get()
        
    Shared_Models_Queue.put(
        '#DONE#'
    )
                   
    end = time.time()
    print(' Error modelling time for faults :  '+str((end-st)/60)+ 'min')
    EvaluatingAndAssignPool.close()
    EvaluatingAndAssignPool.join()
    GenerateModelsPool.close()
    GenerateModelsPool.join()

    storeUnModellableFaults(UnModellableFaults)


            
    
    




