from similarityCalculator import *
from config import *


##############################################
# LCSS PERO EN PROGRAMACION DINAMICA         #
# ------------------------------------------ #
# FUNCIONA IGUAL QUE EL RECURSIVO OPTIMIZADO #
##############################################

def lcss_dp(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    
    minSpatialSim = getSpatialSimilarity_distKnown(39.5/102.5)         # DEFINICION DE BROTE   # Distancia hasta Area -> Zona Lógica           # 39.5 = Distancia Bed-LZ-Bed   # 102.5 = Valor máximo en la Matriz de Locations
    maxTempDist = getTemporalDistance_diffKnown(required_parameters['maxStepsBackwardLCSS'], maxDiffSteps)
    minTemporalSim = getTemporalSimilarity_distKnown(maxTempDist, beta)
                                
    maxStep = stepsP1[len(stepsP1)-1][0]
    minStep = stepsP1[0][0]
    numRowsColumns = maxStep-minStep+1 +1
    c = [[0 for x in range(numRowsColumns)] for y in range(numRowsColumns)]
    cc = [[0 for x in range(numRowsColumns)] for y in range(numRowsColumns)]
    b = [[0 for x in range(numRowsColumns)] for y in range(numRowsColumns)]

    for i in range(1, numRowsColumns):
        for j in range(1, numRowsColumns):
            bedHU_1 = stepsP1[i-1][1]
            bedHU_2 = stepsP2[j-1][1]
            step1 = stepsP1[i-1][0]
            step2 = stepsP1[j-1][0]

            spatialSim = getSpatialSimilarity(bedHU_1, bedHU_2, sp_matrix, hu_matrix)
            tempSim = getTemporalSimilarity(step1, step2, maxDiffSteps, beta)
            if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                test = c[i-1][j-1]
                c[i][j] = test+1
                cc[i][j] = cc[i-1][j-1] + similarity
                b[i][j] = "D"
            else:
                if c[i-1][j] > c[i][j-1]:
                    c[i][j] = c[i-1][j]
                    cc[i][j] = cc[i-1][j]
                    b[i][j] = "U"
                elif c[i-1][j] < c[i][j-1]:
                    c[i][j] = c[i][j-1]
                    cc[i][j] = cc[i][j-1]
                    b[i][j] = "L"
                else:
                    if cc[i-1][j] >= cc[i][j-1]:
                        c[i][j] = c[i-1][j]
                        cc[i][j] = cc[i-1][j]
                        b[i][j] = "U"
                    else:
                        c[i][j] = c[i][j-1]
                        cc[i][j] = cc[i][j-1]
                        b[i][j] = "L"
            
    return c[len(c)-1][len(c)-1], cc[len(cc)-1][len(cc)-1] 

