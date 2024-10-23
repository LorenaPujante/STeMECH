
from similarityCalculator import *



def tsJoin(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    
    # SPATIAL DISTANCE
    spSim1 = getSpatialSimilarity_tsJoin(stepsP1, stepsP2, sp_matrix, hu_matrix)
    spSim2 = getSpatialSimilarity_tsJoin(stepsP2, stepsP1, sp_matrix, hu_matrix)
    spSimilarity = spSim1 + spSim2

    # TEMPORAL DISTANCE
    tempSim1 = getTemporalSimilarity_tsJoin(stepsP1, stepsP2, maxDiffSteps, beta)
    tempSim2 = getTemporalSimilarity_tsJoin(stepsP2, stepsP1, maxDiffSteps, beta)
    tmpSimilarity = tempSim1 + tempSim2

    similarity = getSimilarity_simsKnown(spSimilarity, tmpSimilarity, alfa)

    return similarity



######################
# SIMILITUD ESPACIAL #
######################

def getSpatialSimilarity_tsJoin(stepsP1, stepsP2, sp_matrix, hu_matrix):

    totalSimilarity = 0
    dicBedHU_1_tested = {}   # Para guardar la maxima similitud de un par (Bed, HU)

    for step1 in stepsP1:
        step1_bedHU = step1[1]
        
        key = "{}_{}".format(step1_bedHU.bed, step1_bedHU.hu)
        if key in dicBedHU_1_tested:
            maxSimilarity = dicBedHU_1_tested[key]   # Se obtiene directamente la similitud maxima
        
        else:
            
            dicBedHU_2_tested = {}   # Para guardar la similitud espacial del step1 con los steps de la trayectoria 2
            
            maxSimilarity = 0

            for step2 in stepsP2:
                step2_bedHU = step2[1]
                
                key2 = "{}_{}".format(step1_bedHU.bed, step1_bedHU.hu)
                # Ya se ha comprobado el par (BedHU_1, BedHU_2)
                if key2 in dicBedHU_2_tested:   
                    spSim = dicBedHU_2_tested[key2]
                # No se ha comprobado, por lo que hay que calcular su similitud
                else:
                    spSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                    dicBedHU_2_tested[key2] = spSim

                # Comprobar si la similitud es mayor a la máxima encontrada
                if spSim > maxSimilarity:
                    maxSimilarity = spSim

            dicBedHU_1_tested[key] = maxSimilarity   # Se añade la similitud maxima del par 1    

        
        totalSimilarity += maxSimilarity

    totalSimilarity /= len(stepsP1) # Se diviede entre el numero de steps de la trayectoria 1
    
    return totalSimilarity


def getSpatialSimilarity_tsJoin_basic(stepsP1, stepsP2, sp_matrix, hu_matrix):

    totalSimilarity = 0

    for step1 in stepsP1:
        step1_bedHU = step1[1]
        maxSimilarity = 0

        for step2 in stepsP2:
            step2_bedHU = step2[1]
            spSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
            if spSim > maxSimilarity:
                maxSimilarity = spSim
                
        totalSimilarity += maxSimilarity

    totalSimilarity /= len(stepsP1) # Se diviede entre el numero de steps de la trayectoria 1

    return totalSimilarity



######################
# SIMILITUD TEMPORAL #
######################

def getTemporalSimilarity_tsJoin(stepsP1, stepsP2, maxDiffSteps, beta):

    totalSimilarity = 0

    for step1 in stepsP1:
        step1_nStep = step1[0]
        maxSimilarity = 0

        for step2 in stepsP2:
            step2_nStep = step2[0]
            tmpSim = getTemporalSimilarity(step1_nStep, step2_nStep, maxDiffSteps, beta)
            if tmpSim > maxSimilarity:
                maxSimilarity = tmpSim
                
        totalSimilarity += maxSimilarity

    totalSimilarity /= len(stepsP1) # Se diviede entre el numero de steps de la trayectoria 1

    return totalSimilarity


