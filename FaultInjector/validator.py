import os, sys
import csv, json

import seaborn as sbr
from sfpy import *

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
from utils.args import args
from log.logger import logger

MS = args.mxm.MS
NS = args.mxm.NS



def write_Json(Models, path):
    path = os.path.join(
        os.getcwd(), "FaultInjector", "ErrorModel", "FaultsErrorModel.json"
    )
    with open(path, "w+") as JSONfile:
        json.dump(Models, JSONfile)
    JSONfile.close()


def write_csv(Models, path):
    file_ptr = open(path, "w+")
    writer = csv.writer(file_ptr)
    Table_Title = [
        "FaultID",
        "Scheduling Policy",
        "MeanRelativeError(%)",
        "AverageAbsoluteError",
        "Discrepancy Average activations",
        "Discrepancy Standard Deviation activations",
        "Bit 0 Flip Probability",
        "Bit 1 Flip Probability",
        "Bit 2 Flip Probability",
        "Bit 3 Flip Probability",
        "Bit 4 Flip Probability",
        "Bit 5 Flip Probability",
        "Bit 6 Flip Probability",
        "Bit 7 Flip Probability",
        "Bit 8 Flip Probability",
        "Bit 9 Flip Probability",
        "Bit 10 Flip Probability",
        "Bit 11 Flip Probability",
        "Bit 12 Flip Probability",
        "Bit 13 Flip Probability",
        "Bit 14 Flip Probability",
        "Bit 15 Flip Probability",
        'Probability of Exponent Corruption',
        "Entrance",
        "Probability of entrance curroption(%)",
        "Mask",
        "Average Mantissa Bit Flip --> Golden Exponent",
        'Average Bit Flip Exponent ',
        "Average Mantissa Bit Flip --> Faulty Golden",
        "Ripetition...",
    ]

    writer.writerow(Table_Title)
    for fault in range(len(Models)):
        row = []
        row.append(Models[fault]["FaultID"])
        row.append(Models[fault]["Scheduler"])
        row.append(Models[fault]["MeanRelativeError(%)"])
        row.append(Models[fault]["AverageAbsoluteError"])
        row.append(Models[fault]["Discrepancy_Avg_against_golden_avg"])
        row.append(Models[fault]["Discrepancy_Std_against_golden_std"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit0)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit1)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit2)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit3)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit4)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit5)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit6)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit7)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit8)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit9)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit10)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit11)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit12)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit13)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit14)"])
        row.append(Models[fault]["BitFlipProbabilities"]["P(bit15)"])
        row.append(Models[fault]['ProbabilityExpCorruption'])

        for entrance in Models[fault]["FaultyEntrancesCTA"]:
            row.append(entrance)
            P, bit_always_flipping, AvgMantissaBitFlipExpRight, AvgBitFlipExp, AvgMantissaBitFlipExpWrong = Models[fault]["FaultyEntrancesCTA"][entrance]
            row.append(P)
            row.append(bit_always_flipping)
            row.append(AvgMantissaBitFlipExpRight)
            row.append(AvgBitFlipExp)
            row.append(AvgMantissaBitFlipExpWrong)
            

        writer.writerow(row)

    file_ptr.close()


def number_of_faulty_CTAs(scheduler, TargetCluster, TargetSM):
    if scheduler == "TwoLevelRoundRobin":
        path = os.path.join(
            os.getcwd(), "Schedulers", "scheduled", "two_level_round_robin.json"
        )
    elif scheduler == "GlobalRoundRobin":
        path = os.path.join(
            os.getcwd(), "Schedulers", "scheduled", "global_level_round_robin.json"
        )
    elif scheduler == "Greedy":
        path = os.path.join(os.getcwd(), "Schedulers", "scheduled", "greddy.json")
    elif scheduler == "DistributedCTA":
        path = os.path.join(
            os.getcwd(), "Schedulers", "scheduled", "distributed_CTA.json"
        )
    elif scheduler == "DistributedBlock":
        path = os.path.join(
            os.getcwd(), "Schedulers", "scheduled", "distributed_block.json"
        )
    else:
        print(" wrong scheduler")
        sys.exit()

    n_corrupted_CTA = 0
    with open(path, encoding="utf-8") as json_file:
        CTAs = json.load(json_file)
        for CTA in range(len(CTAs)):
            if CTAs[CTA]["Cluster"] == TargetCluster and CTAs[CTA]["SM"] == TargetSM:
                n_corrupted_CTA += 1
    return n_corrupted_CTA


def read_matrix(filename):
    path = os.getcwd()
    golden_path = os.path.join(path, "golden", filename + ".npy")
    return np.load(golden_path)


def read_fault_list():
    fault_list = []
    path = os.path.join(os.getcwd(), "FaultInjector", "fault_list.csv")
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            fault_list.append(row)
    fault_list.pop(0)  # deleating Table name
    return fault_list[:][:]


def read_results(path):
    result_list = []
    with open(path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            result_list.append(row)
    result_list.pop(0)  # deleating Table name
    return result_list[:][:]


def heat_maps(results, x, y):
    # HEAT MAP GENERATION
    x_coordinates = {
        "TwoLevelRoundRobin": [],
        "GlobalRoundRobin": [],
        "Greedy": [],
        "DistributedCTA": [],
        "DistributedBlock": [],
    }

    y_coordinates = {
        "TwoLevelRoundRobin": [],
        "GlobalRoundRobin": [],
        "Greedy": [],
        "DistributedCTA": [],
        "DistributedBlock": [],
    }

    for fault_entrance in range(len(results)):
        x_coordinates[str(results[fault_entrance][0])].append(
            int(results[fault_entrance][4])
        )
        y_coordinates[str(results[fault_entrance][0])].append(
            int(results[fault_entrance][5])
        )

    # GENERATING PLOTS
    x_grid = np.arange(0, x, MS)
    y_grid = np.arange(0, y, NS)

    # 2LRR
    plt.hist2d(
        x_coordinates["TwoLevelRoundRobin"],
        y_coordinates["TwoLevelRoundRobin"],
        bins=(x + 1, y + 1),
        cmap=cm.gist_rainbow,
        range=[(0, x), (0, y)],
    )
    plt.title("Faulty entrances output tensor Two level Round Robin Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    os.chdir("FaultInjector")
    try:
        os.mkdir("HeatMaps")
    except:
        logger.warning(
            "Failed to create heatmaps repo, already exists, resuming execution"
        )
    os.chdir("../")

    jpg_plots_DIR = os.path.join(os.getcwd(), "FaultInjector", "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, "2LRR.png")
    plt.savefig(_2LRR_jpg)
    plt.close()

    # GRR
    plt.hist2d(
        x_coordinates["GlobalRoundRobin"],
        y_coordinates["GlobalRoundRobin"],
        bins=(x + 1, y + 1),
        cmap=cm.gist_rainbow,
        range=[(0, x), (0, y)],
    )
    plt.title("Faulty entrances output tensor Global Round Robin Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(), "FaultInjector", "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, "GRR.png")
    plt.savefig(_2LRR_jpg)
    plt.close()

    # greedy
    plt.hist2d(
        x_coordinates["Greedy"],
        y_coordinates["Greedy"],
        bins=(x + 1, y + 1),
        cmap=cm.gist_rainbow,
        range=[(0, x), (0, y)],
    )
    plt.title("Faulty entrances output tensor Greedy Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(), "FaultInjector", "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, "Greedy.png")
    plt.savefig(_2LRR_jpg)
    plt.close()

    # DCTA
    plt.hist2d(
        x_coordinates["DistributedCTA"],
        y_coordinates["DistributedCTA"],
        bins=(x + 1, y + 1),
        cmap=cm.gist_rainbow,
        range=[(0, x), (0, y)],
    )
    plt.title("Faulty entrances output tensor Distributed CTA Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(), "FaultInjector", "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, "DCTA.png")
    plt.savefig(_2LRR_jpg)
    plt.close()

    # DB
    plt.hist2d(
        x_coordinates["DistributedBlock"],
        y_coordinates["DistributedBlock"],
        bins=(x + 1, y + 1),
        cmap=cm.gist_rainbow,
        range=[(0, x), (0, y)],
    )
    plt.title("Faulty entrances output tensor Distributed Block Scheduler")
    plt.xlabel("output tensor x coordinates")
    plt.ylabel("output tensot y coordintes")
    plt.colorbar(label="faulty entrance occurrences")

    plt.xticks(x_grid)
    plt.yticks(y_grid)
    plt.grid(color="black", linestyle="--", linewidth=0.5)

    jpg_plots_DIR = os.path.join(os.getcwd(), "FaultInjector", "HeatMaps")
    _2LRR_jpg = os.path.join(jpg_plots_DIR, "DB.png")
    plt.savefig(_2LRR_jpg)
    plt.close()


def MeanRelativeError(FID, sp, results):
    Relative_errors = []
    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            Relative_errors.append(
                abs(float(Float16(results[row][8])) - float(Float16(results[row][6])))
                * 100
                / float(Float16(results[row][6]))
            )
    try:
        return sum(Relative_errors) / len(Relative_errors)
    except ZeroDivisionError:
        return 0.0


def Average_absolute_Error(FID, sp, results):
    Abs_errors = []
    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            Abs_errors.append(
                float(Float16(results[row][8])) - float(Float16(results[row][6]))
            )
    try:
        return sum(Abs_errors) / len(Abs_errors)
    except ZeroDivisionError:
        return 0.0


def Discrepancies_due_to_fault_injection(FID, sp, results, d_golden):
    d_faulty = d_golden.copy()
    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            x = int(results[row][4])
            y = int(results[row][5])
            d_faulty[x][y] = float(Float16(results[row][8]))
    return np.std(d_faulty) - np.std(d_golden), np.mean(d_faulty) - np.mean(d_golden)


def Bit_flip_probability_Format_Float16(FID, sp, results):
    
    NumberEntrancesCorrupted = 0
    NumberEntrancesExpCorrupted = 0
    Bit_flip_occurrences = []
    for bit in range(16):
        Bit_flip_occurrences.append(0)

    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            golden_float16 = Float16(results[row][6])
            faulty_float16 = Float16(results[row][8])
            golden_bits = golden_float16.bits
            faulty_bits = faulty_float16.bits

            bit_flipped = golden_bits ^ faulty_bits
            bit = 0  # aka bit 0
            corrupted_exp = False
            while bit < 16:
                mask_float16 = Float16(pow(2, int(bit)))
                mask = mask_float16.bits
                if mask & bit_flipped != 0:
                    Bit_flip_occurrences[bit] += 1
                    if  9 < bit < 15 :
                        corrupted_exp = True
                bit += 1
            if corrupted_exp :
                NumberEntrancesExpCorrupted += 1
            NumberEntrancesCorrupted += 1

    all_bit_flips = sum(Bit_flip_occurrences)
    for bit in range(16):  # calculate P(%) of bit flip for each bit
        try:
            Bit_flip_occurrences[bit] = Bit_flip_occurrences[bit] * 100 / all_bit_flips

        except ZeroDivisionError as e:
            #logger.warning(
            #    " Error division by zero in bit flip probabilities, returning all zero probabilities"
            #)
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0.0

    return Bit_flip_occurrences[:], float(NumberEntrancesExpCorrupted*100/NumberEntrancesCorrupted)


def Faulty_Entrances_Probabilities_masks(FID, sp, results):
    Faulty_entrances = dict()

    for row in range(len(results)):
        if FID == int(results[row][3]) and sp == str(results[row][0]):
            x_ = int(
                int(results[row][4]) % MS
            )  # to find absolute position of fault propagation in CTA
            y_ = int(int(results[row][5]) % NS)


            try:
                P, bit_always_flipping, AvgMantissaBitFlipExpRight, AvgBitFlipExp, AvgMantissaBitFlipExpWrong, ExpCurroptionCounter = Faulty_entrances[
                    str((x_, y_))
                ]
                P += 1

                golden_float16 = Float16(results[row][6])
                faulty_float16 = Float16(results[row][8])
                golden_bits = golden_float16.bits
                faulty_bits = faulty_float16.bits
                bit_flipped = golden_bits ^ faulty_bits

                bit_always_flipping = (
                    bit_always_flipping & bit_flipped
                )  # in this way we only select bits thar are always flipping in that entrance in CTA

                bit = 15  # aka bit 0
                ExpCorrupted = False
                while bit >= 0:
                    mask_float16 = Float16(pow(2, int(bit)))
                    mask = mask_float16.bits
                    if mask & bit_flipped != 0:
                        if 9 < bit < 15 :
                            AvgBitFlipExp += 1
                            ExpCorrupted = True
                        elif ExpCorrupted: #Mantissa bit flip in a mask with corrupted exponent
                            AvgMantissaBitFlipExpWrong += 1
                        else : #Mantissa bit flip in a mask were exponent was rigth
                            AvgMantissaBitFlipExpRight += 1
                    bit -= 1

                if ExpCorrupted :
                    ExpCurroptionCounter += 1

                Faulty_entrances[str((x_, y_))] = (
                    P,
                    bit_always_flipping,
                    AvgMantissaBitFlipExpRight,
                    AvgBitFlipExp,
                    AvgMantissaBitFlipExpWrong,
                    ExpCurroptionCounter
                )

            except KeyError:
                golden_float16 = Float16(results[row][6])
                faulty_float16 = Float16(results[row][8])
                golden_bits = golden_float16.bits
                faulty_bits = faulty_float16.bits
                bit_flipped = golden_bits ^ faulty_bits

                bit = 15 # aka bit 0
                AvgMantissaBitFlipExpRight = 0
                AvgMantissaBitFlipExpWrong = 0
                AvgBitFlipExp = 0
                ExpCurroptionCounter = 0

                ExpCorrupted = False
                while bit >= 0:
                    mask_float16 = Float16(pow(2, int(bit)))
                    mask = mask_float16.bits
                    if mask & bit_flipped != 0:
                        if 9 < bit < 15 :
                            AvgBitFlipExp += 1
                            ExpCorrupted = True
                        elif ExpCorrupted : # Mantissa bit flip in a mask were exponent was corrupted
                            AvgMantissaBitFlipExpWrong += 1
                        else: #Mantissa bit flip in a mask were exponent was correct
                            AvgMantissaBitFlipExpRight += 1
                    bit -= 1
                
                if ExpCorrupted :
                    ExpCurroptionCounter += 1

                Faulty_entrances.update(
                    {str((x_, y_)): (1, bit_flipped, AvgMantissaBitFlipExpRight, AvgBitFlipExp, AvgMantissaBitFlipExpWrong,ExpCurroptionCounter)}
                )

    for entrance in Faulty_entrances:
        P, mask, AvgMantissaBitFlipExpRight, AvgBitFlipExp , AvgMantissaBitFlipExpWrong, ExpCurroptionCounter = Faulty_entrances[entrance]
        if P == 1:
            mask = None  # is useless to store bits that are always flipping if entrance has been curropted only once
            P = (
                P
                * 100
                / number_of_faulty_CTAs(
                    str(results[row][0]), int(results[0][1]), int(results[0][2])
                )
            )  # Target Cluster and SM are always the same


            Faulty_entrances[entrance] = (P, mask, AvgMantissaBitFlipExpRight, AvgBitFlipExp, AvgMantissaBitFlipExpWrong)# accomulators are directly average since P = 1
        else:
            TimesExpoRight = P - ExpCurroptionCounter
            try:
                AvgMantissaBitFlipExpRight = AvgMantissaBitFlipExpRight/TimesExpoRight
            except ZeroDivisionError:
                AvgMantissaBitFlipExpRight = 0.0
            
            try:
                AvgBitFlipExp = AvgBitFlipExp/ExpCurroptionCounter
                AvgMantissaBitFlipExpWrong = AvgMantissaBitFlipExpWrong/ExpCurroptionCounter
            except ZeroDivisionError:
                AvgBitFlipExp = 0.0
                AvgMantissaBitFlipExpWrong = 0.0

            P = (
                P
                * 100
                / number_of_faulty_CTAs(
                    str(results[row][0]), int(results[0][1]), int(results[0][2])
                )
            )#calculating probability of spatial propagation in faukty CTA

            Faulty_entrances[entrance] = (P, hex(mask), AvgMantissaBitFlipExpRight, AvgBitFlipExp, AvgMantissaBitFlipExpWrong)

    return Faulty_entrances


def fault_info_extrapolation(fault_id, sp, results):
    fault_error_model = {
        "FaultID": fault_id,
        "Scheduler": sp,
        "MeanRelativeError(%)": None,
        "AverageAbsoluteError": None,  # faulty - golden
        "Discrepancy_Avg_against_golden_avg": None,
        "Discrepancy_Std_against_golden_std": None,  # these 2 parameters might be used for fault detection
        "BitFlipProbabilities": {
            "P(bit15)": 0.0,
            "P(bit14)": 0.0,
            "P(bit13)": 0.0,
            "P(bit12)": 0.0,
            "P(bit11)": 0.0,
            "P(bit10)": 0.0,
            "P(bit9)": 0.0,
            "P(bit8)": 0.0,
            "P(bit7)": 0.0,
            "P(bit6)": 0.0,
            "P(bit5)": 0.0,
            "P(bit4)": 0.0,
            "P(bit3)": 0.0,
            "P(bit2)": 0.0,
            "P(bit1)": 0.0,
            "P(bit0)": 0.0,
        },
        'ProbabilityExpCorruption': 0.0,
        "FaultyEntrancesCTA": dict()  # this is a dictionary with key (x,y) : (Probability of curroption of that entrance, Mask to apply, Average number of bit flip Mantissa Exp Right, AvgBitFlipExponent, AvgBitFlipMantissaIfExponentCorrupted)
            # X , Y are inegered between Ms, Ns --> if CTA is associated to faulty HW which are the entrances of to curropt
            # Mask to apply is usually set to 'None' and probabilities of bit flip are explited to curropt data in
            # injector model but if  particular bit flips are always occuring a mask is provided
            # To generate the mask i will also need:
            #Average number of bit flip in Mantissa when exponent right --> mask size if according Probability ExpCorruption a mask with no bit flip in exponent is produced  
            # --> If according to 'ProbabilityExpCorruption' a mask with bit flips in exponent is needed:
            # Avg Bit Flip in Exponent will be generated according to bit flip probabilites of exponent
            # Avg Bit Fip Mantissa if exponent is wrong will be used to generate mask in mantissa 
    }
    fault_error_model["MeanRelativeError(%)"] = MeanRelativeError(
        fault_id, sp, results
    )
    fault_error_model["AverageAbsoluteError"] = Average_absolute_Error(
        fault_id, sp, results
    )

    d = read_matrix("d")
    (
        fault_error_model["Discrepancy_Std_against_golden_std"],
        fault_error_model["Discrepancy_Avg_against_golden_avg"],
    ) = Discrepancies_due_to_fault_injection(fault_id, sp, results, d)

    bit_flip_probabilities, fault_error_model["ProbabilityExpCorruption"] = Bit_flip_probability_Format_Float16(
        fault_id, sp, results
    )
    for bit in range(len(bit_flip_probabilities)):
        fault_error_model["BitFlipProbabilities"][
            "P(bit" + str(bit) + ")"
        ] = bit_flip_probabilities[bit]

    fault_error_model["FaultyEntrancesCTA"] = Faulty_Entrances_Probabilities_masks(
        fault_id, sp, results
    )
    return fault_error_model



def validator(x, y):
    results_path = os.path.join(os.getcwd(), "FaultInjector", "results.csv")
    results = read_results(results_path)
    heat_maps(results, x, y)

    # Now I extrapolate and store all the informations required for error modelling of fault injections
    os.chdir("FaultInjector")
    try:
        os.mkdir("ErrorModel")
    except:
        logger.warning(
            "Failed to create ErrorModel repo, already exists, resuming execution"
        )
    os.chdir("../")

    faults = read_fault_list()
    Faults_Error_Model_List = []
    for f in range(len(faults)):
        Faults_Error_Model_List.append(
            fault_info_extrapolation(int(faults[f][3]), str(faults[f][0]), results)
        )

    path = os.path.join(
        os.getcwd(), "FaultInjector", "ErrorModel", "FaultsErrorModel.json"
    )
    write_Json(Faults_Error_Model_List, path)
    path = os.path.join(
        os.getcwd(), "FaultInjector", "ErrorModel", "FaultsErrorModel.csv"
    )
    write_csv(Faults_Error_Model_List, path)

    logger.warning("Validator module completed")
