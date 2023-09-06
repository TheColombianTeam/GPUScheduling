import torch
import os, json

import sys
from ErrorInjector.MaskInjector import MaskInjector

import numpy as np
from sfpy import *

from Schedulers import *
from Kernel import complete, faulty_Kernel_execution

from utils.args import args
from log.logger import logger

faulty_Cluster = int(args.faults.faulty_Cluster) 
faulty_SM = int(args.faults.faulty_SM)

class ErrorInjector():
    def __init__(self,FaultID,Scheduler,GPU = 'combinedGPU',imported = True, validation = False): #Imported flag is telling the code if Scheduelrs env is introduced in bigger env
        self._imported = imported                                                                # if Validation is true, returns max abs error of ErrorInjector with respcet to execution using PyOpenTCU
        self._validation  = validation
        self._ErrorModels = self.read_FaultsErrorModel(GPU)
        self._MaskInjector = MaskInjector(FaultID,Scheduler, self._ErrorModels)
        self._Scheduler = self.AllocateScheduler(Scheduler)
        self._FID = int(FaultID)
        self._GPU = GPU 
        self._CTAs = dict()
        self._Masks = []



    def FaultyMatrixMult(self, a, b ) :
        self._Masks = []
        golden_torch = self.Float16_MatrixMul(a,b)
        Faulty_np = golden_torch.numpy().copy()
        self.AllocateCTAs(a.numpy(),b.numpy(),Faulty_np)

        for CTA in self._CTAs : 
            if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM :
                Masks = self._MaskInjector.GenerateMasks() 
                self._Masks.append(Masks) # Afther validation --> this list is provided to Modellor to either assign fault to an existed model or generate a new one
                for coordinate in Masks :  #Masks is a dictionary {str(x,y) : Mask}
                    _x , _y = eval(coordinate)
                    try :
                        
                        Faulty_np[CTA['CTA']['x'] + _x][CTA['CTA']['y'] + _y ] = self._MaskInjector.ApplyMask( 
                                        Float16(Faulty_np[CTA['CTA']['x'] + _x][CTA['CTA']['y'] + _y ]), Masks[coordinate])
                    except: #This is an excpetion raised since curropting an entrance introduced by zero padding that is removed later
                        #print('Excpetion')
                        pass
        
        if not(self._validation):
            return torch.from_numpy(Faulty_np)
        
        else : #this part is not ususally executed since it increases computation cost, used for validation of ErrorModel in TestErrorModel.py 
            acc_shape = Faulty_np.shape
            c = np.zeros((acc_shape[0], acc_shape[1]))
            Faulty_Tensor_np = faulty_Kernel_execution(self.Float16Conversion(a.numpy()),
                                                       self.Float16Conversion(b.numpy()),
                                                       c, golden_torch.numpy(),self._CTAs,faulty_Cluster,
                                                       faulty_SM,self._FID)
            
            # The execution though Tensor introduces an offset error of max 0.004 with respect to Float16_MatrixMul
            # --> i'm going to use only faulty entrances from Faulty_Tensor_np for evaluation of ErrorInjector
            RealInjector_outcome = golden_torch.numpy().copy()

            for CTA in self._CTAs : 
                if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM :
                    Model = self._MaskInjector.RetFaultModel()
                    for coordinate in Model["FaultyEntrancesCTA"] :  
                        _x , _y = eval(coordinate)
                        try : 
                            RealInjector_outcome[CTA['CTA']['x'] + _x][CTA['CTA']['y'] + _y ] = Faulty_Tensor_np[CTA['CTA']['x'] + _x][CTA['CTA']['y'] + _y ]                                
                        except: #This is an excpetion raised since curropting an entrance introduced by zero padding that is removed later
                            pass
                                        
            return torch.from_numpy(Faulty_np), self.AvgAbsError(Faulty_np, golden_torch.numpy()), torch.from_numpy(RealInjector_outcome)

    def Float16_MatrixMul(self,a,b):
        self._aNp = a.numpy()
        self._bNp = b.numpy()
        #Float16 conversion inputs
        self._aNp = self.Float16Conversion(self._aNp)
        self._bNp = self.Float16Conversion(self._bNp)
        #Perform metrix mul
        self._cNp = np.matmul(self._aNp, self._bNp)
        #Float16 conversion output
        self._cNp = self.Float16Conversion(self._cNp)

        return torch.from_numpy(self._cNp)

    def read_FaultsErrorModel(self,GPU_name):
        if self._imported :
            path = os.path.join(os.getcwd(),'Schedulers', 'results',GPU_name,'FaultsErrorModel.json')
        else:
            path = os.path.join(os.getcwd(),'results',GPU_name,'FaultsErrorModel.json')

        with open(path,  encoding='utf-8') as json_file:
            ErrorModels = json.load(json_file)
        return ErrorModels

    def AllocateCTAs(self,a,b,c):
        a = complete(a)
        b = complete(b)
        c = complete(c)
        self._CTAs = self._Scheduler.scheduler_algorithm(a, b, c)

    def read_CTAs(self,scheduler):
        if self._imported :
            if scheduler == "TwoLevelRoundRobin":
                path = os.path.join(
                    os.getcwd(),'Schedulers' ,"Schedulers", "scheduled", "two_level_round_robin.json"
                )
            elif scheduler == "GlobalRoundRobin":
                path = os.path.join(
                    os.getcwd(),'Schedulers', "Schedulers", "scheduled", "global_level_round_robin.json"
                )
            elif scheduler == "Greedy":
                path = os.path.join(os.getcwd(), 'Schedulers', "Schedulers", "scheduled", "greddy.json")
            elif scheduler == "DistributedCTA":
                path = os.path.join(
                    os.getcwd(),"Schedulers" ,"Schedulers", "scheduled", "distributed_CTA.json"
                )
            elif scheduler == "DistributedBlock":
                path = os.path.join(
                    os.getcwd(), "Schedulers","Schedulers", "scheduled", "distributed_block.json"
                )
            else:
                print(" wrong scheduler")
                sys.exit()
        
        else:
            if scheduler == "TwoLevelRoundRobin":
                path = os.path.join(
                    os.getcwd(),"Schedulers", "scheduled", "two_level_round_robin.json"
                )
            elif scheduler == "GlobalRoundRobin":
                path = os.path.join(
                    os.getcwd(), "Schedulers", "scheduled", "global_level_round_robin.json"
                )
            elif scheduler == "Greedy":
                path = os.path.join(os.getcwd(),"Schedulers", "scheduled", "greddy.json")
            elif scheduler == "DistributedCTA":
                path = os.path.join(
                    os.getcwd(),"Schedulers", "scheduled", "distributed_CTA.json"
                )
            elif scheduler == "DistributedBlock":
                path = os.path.join(
                    os.getcwd(),"Schedulers", "scheduled", "distributed_block.json"
                )
            else:
                print(" wrong scheduler")
                sys.exit()

        with open(path,  encoding='utf-8') as json_file:
            self._CTAs = json.load(json_file)
        
    def Float16Conversion(self,a) :
        ret_array = a.copy() 
        for row,coloms in enumerate(a):
            for colom, data  in enumerate(coloms):
                ret_array[row][colom] = Float16(a[row][colom]) 
        return ret_array

    def AllocateScheduler(self,Scheduler):
        if Scheduler == "TwoLevelRoundRobin":
            return TwoLevelRoundRobin()
        elif Scheduler == "GlobalRoundRobin":
            return GlobalRoundRobin()
        elif Scheduler == "Greedy":
            return Greedy()
        elif Scheduler == "DistributedCTA":
            return DistributedCTA()
        elif Scheduler == "DistributedBlock":
            return DistributedBlock()
        else:
            print(" wrong scheduler")
            sys.exit()

    def AvgAbsError(self,Np_array_1, Np_array_2):
        AvgAbsError = []
        for row, coloms in enumerate(Np_array_1):
            for colom, data in enumerate(coloms) : 
                if abs(Np_array_1[row][colom] - Np_array_2[row][colom]) > 0.0 :
                    AvgAbsError.append(abs(Np_array_1[row][colom] - Np_array_2[row][colom]))
        try:
            return sum(AvgAbsError)/len(AvgAbsError)
        except ZeroDivisionError:
            return 0.0    

    def MaxAbsError(self,Np_array_1, Np_array_2):  
        MaxError = 0.0
        for row, coloms in enumerate(Np_array_1):
            for colom, data in enumerate(coloms) : 
                if abs(Np_array_1[row][colom] - Np_array_2[row][colom]) > MaxError :
                    MaxError = abs(Np_array_1[row][colom] - Np_array_2[row][colom])
        return MaxError
    
    def MinAbsError(self,Np_array_1, Np_array_2):
        MinError = 100000
        for row, coloms in enumerate(Np_array_1):
            for colom, data in enumerate(coloms) : 
                if abs(Np_array_1[row][colom] - Np_array_2[row][colom]) < MinError  and abs(Np_array_1[row][colom] - Np_array_2[row][colom]) != 0 :
                    MinError = abs(Np_array_1[row][colom] - Np_array_2[row][colom])
        if MinError != 100000 :
            return MinError
        else:
            return 0.0

    def MasksListToModellor(self):
        return self._Masks








