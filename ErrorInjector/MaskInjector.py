import os, json, sys
from log.logger import logger

import numpy as np
from sfpy import *

from utils.args import args
import random

faults_to_inject = args.faults.faults_to_inject

class MaskInjector():
    def __init__(self, FID, Scheduler,FaultsErrorModels):
        self._FID = int(FID)
        self._Scheduler = Scheduler
        self._FaultModel = self.FindErrorModel(FaultsErrorModels)
        self._Masks = dict()


    def GenerateMasks(self):
        self._Masks = dict()
        for key in self._FaultModel["FaultyEntrancesCTA"] : 
            P,BitsAlwaysFlipping, AvgMantissaBitFlipExpoRight, AvgExpBitFlip, AvgMantissaBitFlipExpoWrong =  self._FaultModel["FaultyEntrancesCTA"][key] 
            #we are always going to curropt -->Ignoring Probability of spatial propagation, introduing bits that are always flipping
            try : 
                Mask = Float16(int(BitsAlwaysFlipping,16))
            except: #when there are no bits, BitsAlwaysFlipping == None -->int cast generates an exception
                 Mask = Float16(int('0x0',16))

            self._BitsAlreadyFlipped,self._positionsAlreadyInjected = self.NumberofBitsInMask(Mask.bits)
            self._Probabilies = self.ExtrapolateProbabilities()
            #If in the bits that are always flipping there are exp bit flips or according to 
            #Probability of Exponent Corruption a mask with exponent bit flip is required the proper algo is called
            Events = ['Expo Corrupted', 'ExpGolden']
            if (int(Mask) >= pow(2,9)):
                Mask = Float16(Mask)
                self.ExpCorruptedMask(Mask.bits,AvgExpBitFlip,AvgMantissaBitFlipExpoWrong)
            else:
                P_ExpCorruption = float(self._FaultModel['ProbabilityExpCorruption'])/100
                P_ExpGolden = 1 - P_ExpCorruption
                
                Event = random.choices(Events,[P_ExpCorruption, P_ExpGolden], k= 1)
                if Event == 'Expo Corrupted' and AvgExpBitFlip != 0 : #want to inject exp but cant't if in this location has never been injected
                    Mask = self.ExpCorruptedMask(Mask.bits,AvgExpBitFlip,AvgMantissaBitFlipExpoWrong)
                elif(AvgMantissaBitFlipExpoRight != 0) : #want to inject only Mantissa but I can't if in this location I have masks with also  exponent injected 
                    Mask = self.MantissaCorruptedMask(Mask.bits,AvgMantissaBitFlipExpoRight)
                elif(AvgExpBitFlip != 0 and AvgMantissaBitFlipExpoWrong != 0): #if also this condition is not met means that faults generate no effect
                    Mask = self.ExpCorruptedMask(Mask.bits,AvgExpBitFlip,AvgMantissaBitFlipExpoWrong)
                else:
                    Mask = Float16(0)

            self._Masks.update({key : Mask})           
        
        return self._Masks
    
    def FindErrorModel(self, FaultsErrorsModel):
        fault = 0
        while(fault < len(FaultsErrorsModel)):
            if FaultsErrorsModel[fault]["FaultID"] == self._FID and FaultsErrorsModel[fault]["Scheduler"] == self._Scheduler:
                return FaultsErrorsModel[fault]
            fault += 1
        
        logger.warning(
            'Mask Injector raised exception invalid FID or Scheduler, recieved : ('+str(self._FID) +' and '+ str(self._Scheduler)+' ) \n'
        )
        logger.warning(
            '  0  < FDI <  '+ str(faults_to_inject)+ '   , Handled schedulers : TwoLevelRoundRObin, GlobalRoundRobin, Greedy, DistributedBlock, DistributedCTA'
        )
        sys.exit()

    def ExpCorruptedMask(self, Mask, AvgExpBitFlip, AvgMantissaBitFlipExpoWrong):
        
        NumberBitsAlreadyInjectedInExp = 0
        NumberBitsAlreadyInjectedMantissa = 0
        
        bit = 10
        while(bit < 15):
            if self._positionsAlreadyInjected.count(bit) != 0:
                NumberBitsAlreadyInjectedInExp +=  1
            bit += 1

        bit = 0
        while(bit < 10):
            if self._positionsAlreadyInjected.count(bit) != 0 :
                NumberBitsAlreadyInjectedMantissa += 1
            bit += 1

        BitsToFlipMantissa = self.NumberBitsToFlip(AvgMantissaBitFlipExpoWrong) - NumberBitsAlreadyInjectedMantissa
        BitsToFlipExp = self.NumberBitsToFlip(AvgExpBitFlip) - NumberBitsAlreadyInjectedInExp

        Mask = self.CompleteMask(Mask,
                                 BitsToFlipExp,
                                 self._Probabilies,
                                 ExpInjection=True)
        
        return self.CompleteMask(Mask.bits,
                                 BitsToFlipMantissa,
                                 self._Probabilies,
                                 ExpInjection=False)

    def MantissaCorruptedMask(self,Mask, AvgMantissaBitFlipExpRight):
        BitsToFlipMantissa = self.NumberBitsToFlip(AvgMantissaBitFlipExpRight)
        
        return self.CompleteMask(Mask,
                                 BitsToFlipMantissa - self._BitsAlreadyFlipped,
                                 self._Probabilies,
                                 ExpInjection= False )
          
    def NumberBitsToFlip(self,AvgNumberToFlip):
        
        IntegerPart = int(AvgNumberToFlip)
        FractionalPart = AvgNumberToFlip - IntegerPart

        Actions = ['RoundUp', 'RoundDown']
        
        if(FractionalPart < 0.20):
            return self.RoundDown(AvgNumberToFlip)
        
        elif(FractionalPart < 0.4):
            Action = random.choices(Actions,[0.25,0.75], k = 1)
            if Action == 'RoundUp' :
                return self.RoundUp(AvgNumberToFlip)
            else:
                return self.RoundDown(AvgNumberToFlip)
        
        elif(FractionalPart < 0.6):
            Action = random.choices(Actions,[0.5,0.5], k = 1)
            if Action == 'RoundUp' :
                return self.RoundUp(AvgNumberToFlip)
            else:
                return self.RoundDown(AvgNumberToFlip)
        
        elif( FractionalPart < 0.8):
            Action = random.choices(Actions,[0.75,0.25], k = 1)
            if Action == 'RoundUp' :
                return self.RoundUp(AvgNumberToFlip)
            else:
                return self.RoundDown(AvgNumberToFlip)
        else:
            return self.RoundUp(AvgNumberToFlip) 

    def ExtrapolateProbabilities(self):
        probabilities = []
        for key in self._FaultModel["BitFlipProbabilities"] :
                probabilities.append(float(self._FaultModel["BitFlipProbabilities"][key]) / 100) #probabilities are stored in %
        return probabilities[:]
  
    def NumberofBitsInMask(self,mask):
        bit = 0
        positions = []
        number_of_1 = 0
        while(bit < 16):
            bit_position_int = Float16(int(pow(2,bit)))
            bit_position_mask = bit_position_int.bits
            if (mask & bit_position_mask != 0):
                number_of_1 += 1
                positions.append(bit)
            bit += 1
        return number_of_1, positions[:]

    def CompleteMask(self,InputMask, BitsToFlip, ProbabilitiesBitFlip, ExpInjection = False):
        if BitsToFlip <= 0 : #Mask already completed with bits always flipping
            return Float16(InputMask)
        else :
            BitsToAddToMask = []
            Bits = [15 , 14, 13 ,12, 11, 10, 9, 8, 7,6 ,5 ,4 , 3, 2, 1, 0] 

            if ExpInjection :
                ExpBits = [14 , 13 , 12 , 11 , 10]

                while(len(BitsToAddToMask) < BitsToFlip):
                    CandidateBit = random.choices(Bits,ProbabilitiesBitFlip, k = 1)[0]
                    if ( BitsToAddToMask.count(CandidateBit) == 0 and ExpBits.count(CandidateBit) != 0):#if is not already been injected and is in exp
                        BitsToAddToMask.append(CandidateBit)

            else:
                MantissaBits = [15, 9, 8, 7, 6, 5, 4, 3 ,2 ,1, 0]#counting sign bit as mantissa
                while(len(BitsToAddToMask) < BitsToFlip):
                    CandidateBit = random.choices(Bits, ProbabilitiesBitFlip, k = 1)[0]
                    if ( BitsToAddToMask.count(CandidateBit) == 0 and MantissaBits.count(CandidateBit) != 0):
                        BitsToAddToMask.append(CandidateBit)


            
            bit_counter = 0
            while(bit_counter < 16):
                if BitsToAddToMask.count(bit_counter) > 0 :
                    bit_position_int = Float16(int(pow(2,bit_counter)))
                    bit_position_mask = bit_position_int.bits
                    InputMask = InputMask | bit_position_mask
                bit_counter += 1
        return Float16(InputMask)
   
    def RoundDown(self,input):
        return int(input)
    
    def RoundUp(self,input):
        return int(input)+1
    
    def ApplyMask(self,input, mask):
        return Float16(input.bits ^ mask.bits)

    def RetFaultModel(self):
        return self._FaultModel