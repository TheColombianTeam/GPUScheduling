from ErrorInjector import Modellor
import os, json,sys

from matplotlib import pyplot as plt
from utils.args import args

def write_Json(Data, path):
    with open(path, "w+") as JSONfile:
        json.dump(Data, JSONfile)
    JSONfile.close()

def ReadAAE(FIDs, path,scheduler):
    results = []
    AAE = dict()
    with open(path, encoding='utf-8') as file:
        results = json.load(file)


        for fid in range(len(FIDs)):
            r = 0
            while(int(results[r]["FaultID"]) != FIDs[fid] and results[r]["AverageAbsoluteError"] != scheduler ):
                r += 1

            if    str(results[r]["AverageAbsoluteError"]) == 'nan':
                AAE.update({FIDs[fid] : 1100})
            else:
                AAE.update({FIDs[fid] : abs(float(results[r]["AverageAbsoluteError"]))})

    return AAE

def MaxAAEIsolatedFaults(fids, AAE):
    AAEs = []
    for f in AAE :
        #print(f) 
        if fids.count(f) != 0:
            AAEs.append(AAE[f])
    
    return max(AAEs)

def PloatAAE(AAEModels):
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
            AAEModels[ToPlotDict[ToSortErrors[m]]]['MaxAAEAssociatedFaults' ]
        )
    #Plotting
    x = []
    for f in range(len(AAE_Models)):
        x.append(f)
    
    plt.yscale('log')
    plt.plot(x,ModelsErr, color='red', label='Models AAE')
    plt.plot(x,FaultErr, color='black',label='Max AAE Faults associated to model')
    plt.xlabel('Models sorted in increasing AAE')
    plt.savefig("./ErrorInjector/ModelsAAE.png")
    plt.show()
    plt.close()


    
ToModelFaults = int(args.faults.faults_to_inject) -1
gpu = 'jetson_agx_32'
FaultInjectionResultPath = os.path.join(os.getcwd(), 'results',gpu,'FaultsErrorModel.json')

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
AAE_Faults =  ReadAAE(ModelledFaultsIDs,FaultInjectionResultPath,'TwoLevelRoundRobin')
AAE_Models = []
for m in range(n_models):
    AAE_Models.append({'ModelID': m ,'AAEModel': Models.AAEModel(m) , 
                       'MaxAAEAssociatedFaults' : MaxAAEIsolatedFaults(Models.FaultsIDAssociatedModel(m),
                                                                       AAE_Faults)})
    

PloatAAE(AAE_Models)











