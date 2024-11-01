from similarityCalculator import *
from config import *

def tsJoin_2(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    maxDiffSteps += 1
    
    sim1 = getSpatioTemporalDistance_tsJoin(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
    sim2 = getSpatioTemporalDistance_tsJoin(stepsP2, stepsP1, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
    similarity = sim1 + sim2

    return similarity



def getSpatioTemporalDistance_tsJoin(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    totalSimilarity = 0
    for step1 in stepsP1:
        maxSimilarity = 0
        
        for step2 in stepsP2:
            sim = getSimilarity(step1, step2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)

            if sim > maxSimilarity:
                maxSimilarity = sim

        totalSimilarity += maxSimilarity

    if required_parameters['maxDiffStepsSTLC']:
        totalSimilarity /= maxDiffSteps
    else:
        totalSimilarity /= len(stepsP1)

    return totalSimilarity
