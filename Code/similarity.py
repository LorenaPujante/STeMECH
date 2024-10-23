
####################################
# LLAMADA A FUNCIONES DE SIMILITUD #
####################################

import time

import pandas as pd
import numpy as np

from config import *
from trajectoriesMatcher import *

from trajectoryDist_DTW import dtw
from trajectoryDist_DTW_SpTemp import dtw_SpTemp

from trajectoryDist_tsJoin import tsJoin
from trajectoryDist_tsJoin_2 import tsJoin_2

from trajectoryDist_LCSS import lcss
#from trajectoryDist_LCSS_DP import lcss_dp
from trajectoryDist_LCSS_WTW import lcss_wtw


from clustering import *
from similarityCalculator import *
from writer_reader_similarities import *



#################
# MAIN FUNCTION #
#################

def calculateTrajectoriesSimilarity(dicTrajectories, dicPatTMStep, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa, margin, similarityFunctions, maxNSteps):
    
    if required_parameters['timeInFile']:
        fileName = required_parameters['nameFolder_Time'] + '/time.txt'
        file = open(fileName, "w")
    else:
        file = None

    arrayDataframes = []
    arrayDataframes_01 = []
    
    # Dataframes for saving results
    df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2 = createDataframesForTrajSim(dicTrajectories, similarityFunctions)
    df_dtw_01, df_dtw_st_01, df_lcss_01, df_lcss_2_01, df_tsJoin_01, df_tsJoin_2_01 = createDataframesForTrajSim(dicTrajectories, similarityFunctions)

    dicPat1_Pat2_TrajectoryShort = {}
    print("\nPAIRS OF TRAJECTORIES:")
    for pat1 in dicTrajectories.keys():
        for pat2 in dicTrajectories.keys():    

            if pat1 < pat2:
                key = "{}_{}".format(pat1, pat2)

                # Se igualan ambas trayectorias
                traj1 = dicTrajectories[pat1]
                traj2 = dicTrajectories[pat2]
                if traj1 is None  or  traj2 is None:
                    dicPat1_Pat2_TrajectoryShort[key] = None
                else:
                    newTraj1, newTraj2 = setSameLengthToTrajectories(dicTrajectories, dicPatTMStep, pat1, pat2)
                    print("\n\nTRAJECTORY PATIENTS: {} - {}".format(pat1, pat2))
                    if newTraj1 is None or newTraj2 is None:
                        print(" - None trajectory")
                        dicPat1_Pat2_TrajectoryShort[key] = None
                    else:
                        print(" - Traj 1:")
                        for step in newTraj1:
                            print("   - {}\t- {}_{}".format(step[0], step[1].bed, step[1].hu))
                        print(" - Traj 2:")
                        for step in newTraj2:
                            print("   - {}\t- {}_{}".format(step[0], step[1].bed, step[1].hu))
                        print()
                        
                        # Si alguna trayectoria tiene longitud=0, la similitud será mínima -> 0
                            # En realidad, no haría falta poner el valor a 0 porque el dataframe ya se crea con todo 0s
                        if len(newTraj1)==0  or  len(newTraj2)==0:
                            dicPat1_Pat2_TrajectoryShort[key] = None
                        
                        else:
                            pair = (newTraj1, newTraj2)
                            dicPat1_Pat2_TrajectoryShort[key] = pair

    print("\nMATCH TRAJECTORIES: DONE\n")                

    
    # Cada Funcion de similitud se trata por separado
    for simMethod in similarityFunctions:
        # Calculate similarities
        print("\n* {}".format(getSimilarityMethodName_Short(simMethod)))
        mainFunction_calculateSimilarities(simMethod,similarityFunctions, sp_matrix01,hu_matrix,maxDiffSteps,  beta,alfa,margin, file, dicPat1_Pat2_TrajectoryShort, df_dtw,df_dtw_st, df_lcss,df_lcss_2, df_tsJoin,df_tsJoin_2)
        print("\nSIMILARITIES FOR {}: DONE".format(getSimilarityMethodName_Short(simMethod)))

        # Normalize data to [0, 1]
        print("\nNORMALIZATING OF SIMILARITIES FOR {}".format(getSimilarityMethodName_Short(simMethod)))
        df, df_01 = mainFunction_normalize(simMethod, dicTrajectories,sp_matrix01,maxNSteps, df_dtw,df_dtw_st, df_lcss,df_lcss_2, df_tsJoin,df_tsJoin_2,   df_dtw_01,df_dtw_st_01, df_lcss_01,df_lcss_2_01, df_tsJoin_01,df_tsJoin_2_01)
        print("NORMALIZATION OF SIMILARITIES FOR {}: DONE".format(getSimilarityMethodName_Short(simMethod)))

        # Write on files
        print("\nWRITING RESULTS FOR {}: DONE".format(getSimilarityMethodName_Short(simMethod)))
        mainFunction_write(simMethod, df, False)    # SIMILARITIES
        mainFunction_write(simMethod, df_01, True)  # SIMILARITIES NORMALIZED
        print("WRITTEN RESULTS FOR {}: DONE".format(getSimilarityMethodName_Short(simMethod)))  


    if required_parameters['timeInFile']:
        file.close()
    
    arrayDataframes.append(df_dtw)
    arrayDataframes.append(df_dtw_st)
    arrayDataframes.append(df_lcss)
    arrayDataframes.append(df_lcss_2)
    arrayDataframes.append(df_tsJoin)
    arrayDataframes.append(df_tsJoin_2)

    arrayDataframes_01.append(df_dtw_01)
    arrayDataframes_01.append(df_dtw_st_01)
    arrayDataframes_01.append(df_lcss_01)
    arrayDataframes_01.append(df_lcss_2_01)
    arrayDataframes_01.append(df_tsJoin_01)
    arrayDataframes_01.append(df_tsJoin_2_01)

    return arrayDataframes, arrayDataframes_01



    #-----------------------#
    # FUNCIONES PRINCIPALES #
    #-----------------------#

def mainFunction_calculateSimilarities(simMethod,similarityFunctions, sp_matrix01,hu_matrix,maxDiffSteps,  beta,alfa,margin, file, dicPat1_Pat2_TrajectoryShort, df_dtw,df_dtw_st, df_lcss,df_lcss_2, df_tsJoin,df_tsJoin_2):
    if required_parameters['timeInFile']:
        file.write("* {}\n".format(getSimilarityMethodName_Short(simMethod)))
    for key, pair in dicPat1_Pat2_TrajectoryShort.items():
        pats = key.split('_')
        pat1 = pats[0]
        pat2 = pats[1]
        print("\n   - Pats: {}".format(key))
        if pair is None:
            setSim0(pat1, pat2,  df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2, similarityFunctions)
            print("\tSim_1 / Sim_2: 0")
            if required_parameters['timeInFile']:
                file.write("{}_{}: 0.0 secs\t- 0.0 mins (None Trajectory)\n".format(pat1, pat2))
        else:
            newTraj1 = pair[0]
            newTraj2 = pair[1]
            calculateSimilarities(pat1, pat2,  newTraj1, newTraj2,  sp_matrix01, hu_matrix,  maxDiffSteps,  beta, alfa,  margin,  df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2, simMethod, file)


def mainFunction_normalize(simMethod, dicTrajectories,sp_matrix01,maxNSteps, df_dtw,df_dtw_st, df_lcss,df_lcss_2, df_tsJoin,df_tsJoin_2,   df_dtw_01,df_dtw_st_01, df_lcss_01,df_lcss_2_01, df_tsJoin_01,df_tsJoin_2_01):
    if simMethod == 'dtw' and df_dtw_01 is not None:
        df = df_dtw
    elif simMethod == 'dtw_st' and df_dtw_st_01 is not None:
        df = df_dtw_st
    elif simMethod == 'lcss' and df_lcss_01 is not None:
        df = df_lcss
    elif simMethod == 'lcss_2' and df_lcss_2_01 is not None:
        df = df_lcss_2
    elif simMethod == 'tsJoin' and df_tsJoin_01 is not None:
        df = df_tsJoin
    elif simMethod == 'tsJoin_2' and df_tsJoin_2_01 is not None:
        df = df_tsJoin_2
    
    maxSimTraj = getMaxSimilarityToNormalice(simMethod, sp_matrix01, maxNSteps)
    listPats = list(dicTrajectories.keys())
    df_01 = createDataframePatXPatZeros(dicTrajectories)
    for pat1 in listPats:
        for pat2 in listPats:
            value_01 = normaliceTo01_value(df[pat1][pat2], 0, maxSimTraj) 
            df_01[pat1][pat2] = value_01        
    if simMethod == 'dtw' and df_dtw_01 is not None:
        df_dtw_01 = df_01
    elif simMethod == 'dtw_st' and df_dtw_st_01 is not None:
        df_dtw_st_01 = df_01
    elif simMethod == 'lcss' and df_lcss_01 is not None:
        df_lcss_01 = df_01
    elif simMethod == 'lcss_2' and df_lcss_2_01 is not None:
        df_lcss_2_01 = df_01
    elif simMethod == 'tsJoin' and df_tsJoin_01 is not None:
        df_tsJoin_01 = df_01
    elif simMethod == 'tsJoin_2' and df_tsJoin_2_01 is not None:
        df_tsJoin_2_01 = df_01

    return df, df_01


def mainFunction_write(simMethod, df, is01):
        # Preparar datos para el clustering
    data, listPats = dataframeToArray(df)
        # Guardar datos en un fichero csv
    writeSimilarityMatrixInCSV(data, listPats, simMethod, is01, 0)



    #----------------------#
    # FUNCIONES AUXILIARES #
    #----------------------#

def createDataframesForTrajSim(dicTrajectories, similarityFunctions):
    df_dtw = None
    df_dtw_st = None
    df_lcss = None
    df_lcss_2 = None
    df_tsJoin = None
    df_tsJoin_2 = None

    for f in similarityFunctions:
        if f == 'dtw':
            df_dtw = createDataframePatXPatZeros(dicTrajectories)
        elif f == 'dtw_st':
            df_dtw_st = createDataframePatXPatZeros(dicTrajectories)
        elif f == 'lcss':
            df_lcss = createDataframePatXPatZeros(dicTrajectories)
        elif f == 'lcss_2':
            df_lcss_2 = createDataframePatXPatZeros(dicTrajectories)
        elif f == 'tsJoin':
            df_tsJoin = createDataframePatXPatZeros(dicTrajectories)
        elif f == 'tsJoin_2':
            df_tsJoin_2 = createDataframePatXPatZeros(dicTrajectories)

    return df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2

def createDataframePatXPatZeros(dicTrajectories):
    listPats = list(dicTrajectories.keys())
    nPats = len(listPats)
    dfSimilarities = pd.DataFrame(np.zeros((nPats, nPats)), columns=listPats, index=listPats, dtype=float)

    return dfSimilarities

def createDataframePatXPatZeros_givenListPats(listPats):
    nPats = len(listPats)
    dfSimilarities = pd.DataFrame(np.zeros((nPats, nPats)), columns=listPats, index=listPats, dtype=float)

    return dfSimilarities


def setSim0(pat1, pat2,  df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2, similarityFunctions):

    for f in similarityFunctions:
        if f == 'dtw' and df_dtw is not None:
            df_dtw[pat1][pat2] = 0.0
            df_dtw[pat2][pat1] = 0.0
        elif f == 'dtw_st' and df_dtw_st is not None:
            df_dtw_st[pat1][pat2] = 0.0
            df_dtw_st[pat2][pat1] = 0.0
        elif f == 'lcss' and df_lcss is not None:
            df_lcss[pat1][pat2] = 0.0
            df_lcss[pat2][pat1] = 0.0
        elif f == 'lcss_2' and df_lcss_2 is not None:
            df_lcss_2[pat1][pat2] = 0.0
            df_lcss_2[pat2][pat1] = 0.0
        elif f == 'tsJoin' and df_tsJoin is not None:
            df_tsJoin[pat1][pat2] = 0.0
            df_tsJoin[pat2][pat1] = 0.0
        elif f == 'tsJoin_2' and df_tsJoin_2 is not None:
            df_tsJoin_2[pat1][pat2] = 0.0
            df_tsJoin_2[pat2][pat1] = 0.0


def calculateSimilarities(pat1, pat2,  stepsP1, stepsP2,  sp_matrix01, hu_matrix,  maxDiffSteps,  beta, alfa,  margin,  df_dtw, df_dtw_st, df_lcss, df_lcss_2, df_tsJoin, df_tsJoin_2, simFunction, file):

    #for f in required_parameters['similarityFunctions']:
    if simFunction == 'dtw' and df_dtw is not None:
        maxStep = getMaxStep(stepsP1, stepsP2)
        # SIMILARITY 1
        matrixes['matrixDTW'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_1, diff_secs, diff_mins = getSimilarityDTW(stepsP1, stepsP2, sp_matrix01, hu_matrix)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2S
        matrixes['matrixDTW'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_2, diff_secs, diff_mins = getSimilarityDTW(stepsP2, stepsP1, sp_matrix01, hu_matrix)
        print("\tSim_2: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))
        

        df_dtw[pat1][pat2] = sim_1
        df_dtw[pat2][pat1] = sim_2
    
    elif simFunction == 'dtw_st' and df_dtw_st is not None:
        maxStep = getMaxStep(stepsP1, stepsP2)
        # SIMILARITY 1
        matrixes['matrixDTW_SP'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_1, diff_secs, diff_mins = getSimilarityDTW_SpTemp(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2
        matrixes['matrixDTW_SP'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_2, diff_secs, diff_mins = getSimilarityDTW_SpTemp(stepsP2, stepsP1, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_2: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))

        df_dtw_st[pat1][pat2] = sim_1
        df_dtw_st[pat2][pat1] = sim_2
    
    elif simFunction == 'lcss' and df_lcss is not None:
        maxStep = getMaxStep(stepsP1, stepsP2)
        # SIMILARITY 1
        matrixes['matrixLCSS'] = [[None for x in range(maxStep)] for y in range(maxStep)] #[[None] * maxStep] * maxStep
        sim_1, diff_secs, diff_mins = getSimilarityLCSS(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2
        matrixes['matrixLCSS'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_2, diff_secs, diff_mins = getSimilarityLCSS(stepsP2, stepsP1, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_2: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))
        
        df_lcss[pat1][pat2] = sim_1[1]
        df_lcss[pat2][pat1] = sim_2[1]
    
    elif simFunction == 'lcss_2' and df_lcss_2 is not None:
        maxStep = getMaxStep(stepsP1, stepsP2)
        # SIMILARITY 1
        matrixes['matrixLCSS_2'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_1, diff_secs, diff_mins = getSimilarityLCSS_WTW(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa, stepsP1, stepsP2, margin)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2
        matrixes['matrixLCSS_2'] = [[None for x in range(maxStep)] for y in range(maxStep)]
        sim_2, diff_secs, diff_mins = getSimilarityLCSS_WTW(stepsP2, stepsP1, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa, stepsP2, stepsP1, margin)
        print("\tSim_2: {} \n\t\tme: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))
        
        df_lcss_2[pat1][pat2] = sim_1[1]
        df_lcss_2[pat2][pat1] = sim_2[1]
    
    elif simFunction == 'tsJoin' and df_tsJoin is not None:
        # SIMILARITY 1
        sim_1, diff_secs, diff_mins = getSimilarityTSJoin(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2
        sim_2, diff_secs, diff_mins = getSimilarityTSJoin(stepsP2, stepsP1, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_2: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))
        
        df_tsJoin[pat1][pat2] = sim_1
        df_tsJoin[pat2][pat1] = sim_2
    
    elif simFunction == 'tsJoin_2' and df_tsJoin_2 is not None:
        # SIMILARITY 1
        sim_1, diff_secs, diff_mins = getSimilarityTSJoin_2(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_1: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_1, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat1, pat2, diff_secs, diff_mins))
        # SIMILARITY 2
        sim_2, diff_secs, diff_mins = getSimilarityTSJoin_2(stepsP2, stepsP1, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
        print("\tSim_2: {} \n\t\tTime: {:.6f} secs\t- {:.6f} mins".format(sim_2, diff_secs, diff_mins))
        if required_parameters['timeInFile']:
            file.write("{}_{}: {:.6f} secs\t- {:.6f} mins\n".format(pat2, pat1, diff_secs, diff_mins))

        df_tsJoin_2[pat1][pat2] = sim_1
        df_tsJoin_2[pat2][pat1] = sim_2
        


def getSimilarityDTW(stepsP1, stepsP2, sp_matrix01, hu_matrix):
    inicio = time.time()
    similarity = dtw(stepsP1, stepsP2, sp_matrix01, hu_matrix)
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min


def getSimilarityDTW_SpTemp(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa):
    inicio = time.time()
    similarity = dtw_SpTemp(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min


def getSimilarityLCSS(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa):
    inicio = time.time()
    similarity = lcss(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min

def getSimilarityLCSS_WTW(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, margin):
    inicio = time.time()
    similarity = lcss_wtw(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, [], [], margin)    #lcss_3_margin  #lcss_3_margin_all
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min

def getSimilarityTSJoin(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa):
    inicio = time.time()
    similarity = tsJoin(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min

def getSimilarityTSJoin_2(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa):
    inicio = time.time()
    similarity = tsJoin_2(stepsP1, stepsP2, sp_matrix01, hu_matrix, maxDiffSteps, beta, alfa)
    fin = time.time()
    diff = fin-inicio
    diff_min = diff/60
    
    return similarity, diff, diff_min



#----------------------------#
# OTRAS FUNCIONES AUXILIARES #
#----------------------------#

def getMaxStep(stepsP1, stepsP2):
    maxStep1 = stepsP1[len(stepsP1)-1][0]
    maxStep2 = stepsP2[len(stepsP2)-1][0]
    if maxStep1 >= maxStep2:
        maxStep = maxStep1+1
    else:
        maxStep = maxStep2+1

    return maxStep

#################################
# NORMALIZATION OF SIMILARITIES #
#################################

def normaliceTo01_arrayDataFrames(arrayDataframes, dicTrajectories, sp_matrix01, maxNSteps):
    arrayDataframes_01 = []
    listPats = list(dicTrajectories.keys())
    j = 0
    for i in range(len(arrayDataframes)):
        df = arrayDataframes[i]
        if df is not None:
            simMethod = required_parameters['similarityFunctions'][j]
            maxSimTraj = getMaxSimilarityToNormalice(simMethod, sp_matrix01, maxNSteps)
            j += 1

            # V' = ((V-minA)/(maxA-minA)) * (maxB-minB) + minB
            # minB = 0    ;   maxB = 1
            # minA = 0    ;   maxB = maxSimTraj
            
            df_01 = createDataframePatXPatZeros(dicTrajectories)
            for pat1 in listPats:
                for pat2 in listPats:
                    value_01 = normaliceTo01_value(df[pat1][pat2], 0, maxSimTraj) 
                    df_01[pat1][pat2] = value_01

            arrayDataframes_01.append(df_01)

        else:
            arrayDataframes_01.append(None)

    return arrayDataframes_01


def normaliceTo01_value(value, oldMin, oldMax):
    a = value-oldMin
    b = oldMax-oldMin
    return a/b

def getMaxSimilarityToNormalice(simMethod, sp_matrix01, maxNSteps):
    if simMethod not in ['tsJoin', 'tsJoin_2']:
        sp_matrix01_list = sp_matrix01.values.tolist()
        sp_matrix01_list = [x for xs in sp_matrix01_list for x in xs]
        sp_matrix01_list = list(set(sp_matrix01_list))
        sp_matrix01_list.sort()
        
        minSpDist = sp_matrix01_list[1]
        minSpDist *= 0.5    # Same HU
        
        maxSpSim = getSpatialSimilarity_distKnown(minSpDist)
        maxTmpSim = getTemporalSimilarity_distKnown(0, required_parameters['beta'])
        
        maxSim = getSimilarity_simsKnown(maxSpSim, maxTmpSim, required_parameters['alfa'])
        
        maxSimTraj = maxSim * maxNSteps
        
        return maxSimTraj
    return 2
