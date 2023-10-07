from ErrorInjector import Modellor
import os, json,sys,csv

from matplotlib import pyplot as plt
from utils.args import args
from sfpy import *
import math

def read_results(gpu):
    path = os.path.join(os.getcwd(),'results',gpu,'results.csv')
    result_list = []
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            result_list.append(row)
    result_list.pop(0)  # deleating Table name
    return result_list[:][:]

def Average_absolute_Error(FID, sp, results):
    Abs_errors = []
    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            Abs_errors.append(
                abs(float(Float16(results[row][8])) - float(Float16(results[row][6])))
            )
    try:
        return sum(Abs_errors) / len(Abs_errors)
    except ZeroDivisionError:
        return 0.0

def write_Json(Data, path):
    with open(path, "w+") as JSONfile:
        json.dump(Data, JSONfile)
    JSONfile.close()

def ReadAAE(FIDs,gpu,scheduler,results):
    results = []
    AAE = dict()
    results = read_results(gpu)
    
    for fid in range(len(FIDs)):
        current_AAE = Average_absolute_Error(FIDs[fid],scheduler,results)

        if    str(current_AAE) == 'nan' or str(current_AAE) == 'inf':
                AAE.update({FIDs[fid] : 10000})
        else:
                AAE.update({FIDs[fid] : current_AAE})

    return AAE

def AvgAAEIsolatedFaults(fids, AAE):
    AAEs = []
    for f in AAE :
        #print(f) 
        if fids.count(f) != 0:
            AAEs.append(AAE[f])
    if AAEs.count(float('inf')) != 0 or  AAEs.count(float('nan')) != 0 : 
        return 100000
    #return max(AAEs)
    try:
        return sum(AAEs)/len(AAEs)
    except ZeroDivisionError:
        0.0

def MaxAAEIsolatedFaults(fids, AAE):
    AAEs = []
    for f in AAE :
        #print(f) 
        if fids.count(f) != 0:
            AAEs.append(AAE[f])
    if AAEs.count(float('inf')) != 0 or  AAEs.count(float('nan')) != 0 : 
        return None
    return max(AAEs)
    
def PloatAAE(AAEModels, factor,gpu):
    ToSortErrors = []
    ToPlotDict = dict()

    for m in range(len(AAE_Models)):
        ToSortErrors.append(AAE_Models[m]['AAEModel'])
        ToPlotDict.update({ToSortErrors[-1] : m })

    ToSortErrors.sort()

    ModelsErr = []
    FaultErr = []

    for m in range(len(ToSortErrors)):
        ModelsErr.append(ToSortErrors[m])
        FaultErr.append(
            AAEModels[ToPlotDict[ToSortErrors[m]]][str(factor)]
        )
    #Plotting
    x = []
    for f in range(len(AAE_Models)):
        x.append(f)

    #calculating cross correlation
    corss_correlation = []
    for k in range(len(x)):
        corss_correlation.append(0.0)
        for i in range(len(x)) : 
            corss_correlation[k] += FaultErr[i]*ModelsErr[i-k]
    

    #claculate correlation coefficient
    sumModelsErr = sum(ModelsErr)
    sumFaultsErr = sum(FaultErr)

    SumofProd = 0
    for f  in range(len(x)):
        SumofProd += ModelsErr[f]* FaultErr[f]
    
    sumModelsErrSquared = 0
    for f in range(len(x)):
        sumModelsErrSquared += ModelsErr[f]*ModelsErr[f]
    
    sumFaultsErrSquared = 0
    for f in range(len(x)):
        sumFaultsErrSquared += FaultErr[f]*FaultErr[f]

    CorrelationCoeff = (len(x) * SumofProd - sumModelsErr*sumFaultsErr) / math.sqrt((len(x)* sumFaultsErrSquared - sumFaultsErr * sumFaultsErr) *
                                                                                    (len(x) * sumModelsErrSquared - sumModelsErr*sumModelsErr) )

    print('Correlation coeff '+ str(factor) + ' : ' + str(CorrelationCoeff))


    plt.yscale('log')
    plt.plot(x,ModelsErr, color='red', label='Models AAE')
    plt.plot(x,FaultErr, color='black', label='Max AAE Faults associated to model')
    plt.plot(x,corss_correlation,color= 'blue', label= 'Cross Correlation')
    plt.legend()
    plt.xlabel('Models sorted in increasing AAE')
    plt.savefig("./ErrorInjector/"+ str(gpu)+'/'+str(factor)+".png")
    plt.show()
    plt.close()

def RankedBasedCorrelation(AAEModels):

    #Models AAE Ranking
    ModelsRanking = []
    ModelsAAESorted = []
    for m in range(len(AAEModels)):
        ModelsAAESorted.append(AAEModels[m]['AAEModel'])
    ModelsAAESorted.sort()

    already_ranked = []
    for m in range(len(ModelsAAESorted)):
        modelID = 0
        while(AAEModels[modelID]['AAEModel'] != ModelsAAESorted[m] and already_ranked.count(modelID) == 0):
            modelID += 1
        already_ranked.append(modelID)
        ModelsRanking.append({ 'RankingPosition' : m , 'ModelID': AAEModels[modelID]['ModelID']})
    
    #Max AAE of faults associated to model Ranking 
    MaxFaultsRanking = []
    MaxFaultsAAESorted = []
    
    for m in range(len(AAEModels)):
        MaxFaultsAAESorted.append(AAEModels[m]['MaxAAEAssociatedFaults'])
    MaxFaultsAAESorted.sort()

    already_ranked = []
    for m in range(len(MaxFaultsAAESorted)):
        modelID = 0
        while(AAEModels[modelID]['MaxAAEAssociatedFaults'] != MaxFaultsAAESorted[m] and already_ranked.count(modelID) == 0):
            modelID += 1
        already_ranked.append(modelID)
        MaxFaultsRanking.append({ 'RankingPosition' : m , 'ModelID': AAEModels[modelID]['ModelID']})
    
    #Avg AAE of faults associated to model Ranking
    AvgFaultsRanking = []
    AvgFaultsAAESorted = []
    for m in range(len(AAEModels)):
        AvgFaultsAAESorted.append(AAEModels[m]['AvgAAEAssociatedFaults'])
    AvgFaultsAAESorted.sort()

    already_ranked = []
    for m in range(len(AvgFaultsAAESorted)):
        modelID = 0
        while(AAEModels[modelID]['AvgAAEAssociatedFaults'] != AvgFaultsAAESorted[m] and already_ranked.count(modelID) == 0):
            modelID += 1
        already_ranked.append(modelID)
        AvgFaultsRanking.append({ 'RankingPosition' : m , 'ModelID': AAEModels[modelID]['ModelID']})


    #Calculating Ranked based correlation Model AAE against Avg AAE associated to faults
    RankingSquaredDifferenceSum = 0
    for model in range(len(ModelsRanking)):
        mid = ModelsRanking[model]['ModelID']
        RankingModel = ModelsRanking[model]['RankingPosition']

        m = 0
        while( AvgFaultsRanking[m]['ModelID'] != mid):
            m += 1
        RankingFault = AvgFaultsRanking[m]['RankingPosition']
        RankingSquaredDifferenceSum += (RankingModel - RankingFault)*(RankingModel - RankingFault)
    RankedBasedCorrelationModelAgainstAvgAAE = 1- 6*RankingSquaredDifferenceSum/(len(ModelsRanking) * (len(ModelsRanking)*len(ModelsRanking) - 1))
    print('RankedBasedCorrelationModelAgainstAvgAAE  : ' +str(RankedBasedCorrelationModelAgainstAvgAAE))

    #Calculating Ranked based correlation Model AAE against Max AAE associated to faults
    RankingSquaredDifferenceSum = 0
    for model in range(len(ModelsRanking)):
        mid = ModelsRanking[model]['ModelID']
        RankingModel = ModelsRanking[model]['RankingPosition']

        m = 0
        try: 
            while( MaxFaultsRanking[m]['ModelID'] != mid):
                m += 1
            RankingFault = MaxFaultsRanking[m]['RankingPosition']
            RankingSquaredDifferenceSum += (RankingModel - RankingFault)*(RankingModel - RankingFault)
        except IndexError:
            #print( MaxFaultsRanking[m]['ModelID'])
            print(mid)
            print(len(MaxFaultsRanking))
            sys.exit()

        
    RankedBasedCorrelationModelAgainstMaxAAE = 1- 6*RankingSquaredDifferenceSum/(len(ModelsRanking) * (len(ModelsRanking)*len(ModelsRanking) - 1))
    print('RankedBasedCorrelationModelAgainstMaxAAE : ' + str(RankedBasedCorrelationModelAgainstMaxAAE))

    


   
ToModelFaults = int(args.faults.faults_to_inject) -1
gpu = 'jetson_agx_32'
#FaultInjectionResultPath = os.path.join(os.getcwd(), 'results',gpu,'FaultsErrorModel.json')

Models = Modellor('TwoLevelRoundRobin',False,False,gpu)
n_models = Models.ModelIDs()
Modelled_faults = 0
ModelledFaultsIDs = []

for model in range(n_models):
    Modelled_faults += Models.NumberFaultsAssociatedModelID(model)
    ModelledFaultsIDs += Models.FaultsIDAssociatedModel(model)

    
print('Ammount of modelled faults' + gpu + 'is : ' + str(float(Modelled_faults*100/ToModelFaults)) + '%')

#Isolate UnmodellableFaults
UnModellableFaultsIDs = []
for m in range(ToModelFaults):
    if ModelledFaultsIDs.count(m) == 0 : 
        UnModellableFaultsIDs.append(m)

write_Json(UnModellableFaultsIDs,os.path.join(os.getcwd(), 'ErrorInjector',gpu,'UnModellableFaults.json'))
results = read_results(gpu)
AAE_Faults =  ReadAAE(ModelledFaultsIDs,gpu,'TwoLevelRoundRobin',results)
AAE_Models = []
for m in range(n_models):
    if MaxAAEIsolatedFaults(Models.FaultsIDAssociatedModel(m),AAE_Faults) != None : 
        AAE_Models.append({'ModelID': m ,'AAEModel': Models.AAEModel(m) , 
                       'MaxAAEAssociatedFaults' : MaxAAEIsolatedFaults(Models.FaultsIDAssociatedModel(m),AAE_Faults), 'AvgAAEAssociatedFaults' : 
                                                                       AvgAAEIsolatedFaults(Models.FaultsIDAssociatedModel(m),  AAE_Faults) })


    
PloatAAE(AAE_Models,'MaxAAEAssociatedFaults',gpu)
PloatAAE(AAE_Models,'AvgAAEAssociatedFaults', gpu)


RankedBasedCorrelation(AAE_Models)
sys.exit()
MRE_AAEModel_against_maxAAEFault = 0
Max_RE_AAEModel_against_maxAAEFault = 0
model_max = -1
for m in range(len(AAE_Models)):
    if AAE_Models[m]['MaxAAEAssociatedFaults'] != 0 and AAE_Models[m]['MaxAAEAssociatedFaults'] != 10000 : 
        mre = abs(AAE_Models[m]['MaxAAEAssociatedFaults'] - AAE_Models[m]['AAEModel'])*100/AAE_Models[m]['AAEModel']
        MRE_AAEModel_against_maxAAEFault += mre
        if mre > Max_RE_AAEModel_against_maxAAEFault:
            Max_RE_AAEModel_against_maxAAEFault = mre
            model_max = m


print('MRE_AAEModel_against_maxAAEFault : ' + str(MRE_AAEModel_against_maxAAEFault/n_models) + '  %')
print('Max MRE : ' + str(Max_RE_AAEModel_against_maxAAEFault) + '   %')
print(model_max)
print(AAE_Models[model_max]['MaxAAEAssociatedFaults'])
print(AAE_Models[model_max]['AAEModel'])









