import os, time
from Schedulers import *

from matplotlib import pyplot as plt
from matplotlib import cm

import numpy as np
from utils.args import args
from log.logger import logger



def Plotting(x,y,scheduler):
    global dimentions,MS,NS
    try:
        os.chdir('Schedulers')
    except:
        pass

    path = os.path.join(os.getcwd(), 'DynamicBehaviour',scheduler + '.png')

    x_grid = np.arange(0, dimentions, MS)
    y_grid = np.arange(0, dimentions, NS)
    plt.hist2d(
        x,
        y,
        bins=(dimentions, dimentions),
        cmap=cm.gist_rainbow,
        range=[(0, dimentions), (0, dimentions)],
    )
    plt.title("Faulty CTA output tensor " + scheduler)
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="CTA occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)
    plt.savefig(path)
    plt.close()





dimentions = 120
faulty_SM = int(args.faults.faulty_SM)
faulty_Cluster = int(args.faults.faulty_Cluster)
MS = int(args.mxm.MS)
NS = int(args.mxm.NS)


logger.warning('Studing dynamic behaviout of Schedulers for gpu structure --> number of clusters : ' + str(args.gpu.cluster)
                + '  number of SM : ' + str(args.gpu.sm))


a = np.random.rand(dimentions,dimentions)
b = np.random.rand(dimentions,dimentions)
c = np.matmul(a,b)


TwoLevelRoundRobinScheduler = TwoLevelRoundRobin()
GlobalRoundRobinScheduler = GlobalRoundRobin()
GreedyScheduler = Greedy()
DistributedBlockScheduler = DistributedBlock()
DistributedCTAScheduler = DistributedCTA()

x_2lrr = []
y_2lrr = []

x_grr = []
y_grr = []

x_greedy = []
y_greedy = []

x_db = []
y_db = []

x_dcta = []
y_dcta = []

st = time.time()
for fault in range(100000):
    CTAs_2lrr = TwoLevelRoundRobinScheduler.scheduler_algorithm(a,b,c)
    CTAs_grr = GlobalRoundRobinScheduler.scheduler_algorithm(a,b,c)
    CTAs_greedy = GreedyScheduler.scheduler_algorithm(a,b,c)
    CTAs_db = DistributedBlockScheduler.scheduler_algorithm(a,b,c)
    CTAs_dcta = DistributedCTAScheduler.scheduler_algorithm(a,b,c)

    for CTA in CTAs_2lrr : 
        if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM : 
            for _x in range(MS):
                for _y in range(NS):
                    if (CTA['CTA']['x'] + _x) < dimentions and (CTA['CTA']['y'] + _y) < dimentions : 
                        x_2lrr.append(CTA['CTA']['x'] + _x) #fault propagation is confined in CTA dimentions
                        y_2lrr.append(CTA['CTA']['y'] + _y) 
    
    

    for CTA in CTAs_grr : 
        if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM : 
            for _x in range(MS):
                for _y in range(NS):
                    if (CTA['CTA']['x'] + _x) < dimentions and (CTA['CTA']['y'] + _y) < dimentions : 
                        x_grr.append(CTA['CTA']['x'] + _x) 
                        y_grr.append(CTA['CTA']['y'] + _y)
    

    for CTA in CTAs_greedy : 
        if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM : 
            for _x in range(MS):
                for _y in range(NS):
                    if (CTA['CTA']['x'] + _x) < dimentions and (CTA['CTA']['y'] + _y) < dimentions : 
                        x_greedy.append(CTA['CTA']['x'] + _x)
                        y_greedy.append(CTA['CTA']['y'] + _y)

    for CTA in CTAs_db : 
        if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM : 
            for _x in range(MS):
                for _y in range(NS):
                    if (CTA['CTA']['x'] + _x) < dimentions and (CTA['CTA']['y'] + _y) < dimentions : 
                        x_db.append(CTA['CTA']['x'] + _x)
                        y_db.append(CTA['CTA']['y'] + _y)


    for CTA in CTAs_dcta : 
        if CTA['Cluster'] == faulty_Cluster and CTA['SM'] == faulty_SM : 
            for _x in range(MS):
                for _y in range(NS):
                    if (CTA['CTA']['x'] + _x) < dimentions and (CTA['CTA']['y'] + _y) < dimentions : 
                        x_dcta.append(CTA['CTA']['x'] + _x)
                        y_dcta.append(CTA['CTA']['y'] + _y)


Plotting(x_2lrr,y_2lrr,'TwoLevelRoundRobin')
Plotting(x_grr, y_grr, 'GlobalRoundRobin')
Plotting(x_greedy,y_greedy,'Greedy')
Plotting(x_db,y_db,'DistributedBlock')
Plotting(x_dcta,y_dcta,'DistributedCTA')

end = time.time()
logger.warning('Execution time for 100k executions: ' + str((end-st)/60) + '  min')


