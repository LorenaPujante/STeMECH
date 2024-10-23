import math


######################
# DIMENSION ESPACIAL #
######################

def getSpatialSimilarity(stepBedHU1, stepBedHU2, sp_matrix, hu_matrix):
    
    spDistance = getSpatialDistance(stepBedHU1, stepBedHU2, sp_matrix, hu_matrix)
    spSimilarity = 1 - spDistance
    
    return spSimilarity


def getSpatialDistance(ev1, ev2, sp_matrix, hu_matrix):
    
    bed1 = ev1.bed
    bed2 = ev2.bed
    hu1 = ev1.hu
    hu2 = ev2.hu

    # Cálcular la distancia
    spDistance = sp_matrix[bed1][bed2]
    if hu1 == hu2:
        spDistance *= 0.5
    elif hu_matrix[hu1][hu2] == 2:
        spDistance *= 0.7

    return spDistance


def getSpatialSimilarity_distKnown(spDistance):
    spSimilarity = 1 - spDistance
    return spSimilarity



######################
# DIMENSION TEMPORAL #
######################

def getTemporalSimilarity(step1, step2, maxDiffSteps, beta):
    
    tmpDistance = getTemporalDistance(step1, step2, maxDiffSteps)
    
    # Pasarla la distancia a similitud
    newBeta = math.log(beta)    # Por defecto, la base es el número e   -> Logaritmo neperiano
    tmpSimilarity = newBeta * tmpDistance
    tmpSimilarity = math.exp(tmpSimilarity)

    return tmpSimilarity


def getTemporalDistance(step1, step2, maxDiffSteps):
    
    diff = abs(step1-step2)
    tmpDistance = diff/maxDiffSteps

    return tmpDistance


def getTemporalDistance_diffKnown(diffSteps, maxDiffSteps):
    tmpDistance = diffSteps/maxDiffSteps
    return tmpDistance


def getTemporalSimilarity_distKnown(tmpDistance, beta):
    newBeta = math.log(beta)
    tmpSimilarity = newBeta * tmpDistance
    tmpSimilarity = math.exp(tmpSimilarity)

    return tmpSimilarity


########################
# UNION DE DIMENSIONES #
########################

def getSimilarity(step1, step2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    
    step1_bedHU = step1[1]
    step1_step = step1[0]
    step2_bedHu = step2[1]
    step2_step = step2[0]

    spSimilarity = getSpatialSimilarity(step1_bedHU, step2_bedHu, sp_matrix, hu_matrix)
    tmpSimilarity = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)

    similarity = alfa*spSimilarity + (1-alfa)*tmpSimilarity

    return similarity 


def getSimilarity_simsKnown(spSimilarity, tmpSimilarity, alfa):
    similarity = alfa*spSimilarity + (1-alfa)*tmpSimilarity
    return similarity 