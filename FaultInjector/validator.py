import os
import csv
import seaborn as sbr

import matplotlib.pyplot as plt
from matplotlib import  cm

import numpy as np
from utils.args import args

MS = args.mxm.MS
NS = args.mxm.NS


def read_results(path):
    result_list = []
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            result_list.append(row)
    result_list.pop(0)#deleating Table name
    return result_list[:][:]

def heat_maps(results,x,y):
    #HEAT MAP GENERATION
    x_coordinates ={
        'TwoLevelRoundRobin' : [],
        'GlobalRoundRobin'   : [],
        'Greedy':              [],
        'DistributedCTA':      [],
        'DistributedBlock':    []
    }


    y_coordinates ={
        'TwoLevelRoundRobin' : [],
        'GlobalRoundRobin'   : [],
        'Greedy':              [],
        'DistributedCTA':      [],
        'DistributedBlock':    []
    }

    for fault_entrance in range(len(results)):
        x_coordinates[str(results[fault_entrance][0])].append(int(results[fault_entrance][4]))
        y_coordinates[str(results[fault_entrance][0])].append(int(results[fault_entrance][5]))

    #GENERATING PLOTS
    x_grid = np.arange(0,x, MS)
    y_grid = np.arange(0, y, NS)

    #2LRR
    plt.hist2d(x_coordinates['TwoLevelRoundRobin'],y_coordinates['TwoLevelRoundRobin'], bins=(x+1,y+1), 
                                                                    cmap=cm.gist_rainbow, range=[(0,x),(0,y)])
    plt.title("Faulty entrances output tensor Two level Round Robin Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    os.chdir('FaultInjector')
    try:
        os.mkdir('HeatMaps')
    except:
        print("Failed to create heatmaps repo, already exists, resuming execution")
    os.chdir('../')

    jpg_plots_DIR = os.path.join(os.getcwd(),'FaultInjector', "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, '2LRR.png')
    plt.savefig(_2LRR_jpg)
    plt.close()

    #GRR
    plt.hist2d(x_coordinates['GlobalRoundRobin'],y_coordinates['GlobalRoundRobin'], bins=(x+1,y+1), 
                                                                    cmap=cm.gist_rainbow, range=[(0,x),(0,y)])
    plt.title("Faulty entrances output tensor Global Round Robin Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(),'FaultInjector', "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR,'GRR.png')
    plt.savefig(_2LRR_jpg)
    plt.close()

    #greedy
    plt.hist2d(x_coordinates['Greedy'],y_coordinates['Greedy'], bins=(x+1,y+1), 
                                                                    cmap=cm.gist_rainbow, range=[(0,x),(0,y)])
    plt.title("Faulty entrances output tensor Greedy Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(),'FaultInjector', "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR,'Greedy.png')
    plt.savefig(_2LRR_jpg)
    plt.close()

    #DCTA
    plt.hist2d(x_coordinates['DistributedCTA'],y_coordinates['DistributedCTA'], bins=(x+1,y+1), 
                                                                    cmap=cm.gist_rainbow, range=[(0,x),(0,y)])
    plt.title("Faulty entrances output tensor Distributed CTA Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(),'FaultInjector', "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR,'DCTA.png')
    plt.savefig(_2LRR_jpg)
    plt.close()

    #DB
    plt.hist2d(x_coordinates['DistributedBlock'],y_coordinates['DistributedBlock'], bins=(x+1,y+1), 
                                                                    cmap=cm.gist_rainbow, range=[(0,x),(0,y)])
    plt.title("Faulty entrances output tensor Distributed Block Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(),'FaultInjector', "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR,'DB.png')
    plt.savefig(_2LRR_jpg)
    plt.close()
    





def validator(x, y):
    results_path = os.path.join(os.getcwd(), 'FaultInjector', 'results.csv')
    results = read_results(results_path)
    heat_maps(results,x,y)

    