from similarityCalculator import *
from config import *

def dtw_SpTemp(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):

    # Base cases    
    if len(stepsP1)==0 and len(stepsP2)==0:
        return [0,0]    
    if len(stepsP1)==0 or len(stepsP2)==0:
        return [999999,0]
    
    

    # Recursivo
    #  sim(Head(P1), Head(P2)) + max {
    #                                   DTW(P1, Rest(P2)),          # Option 1
    #                                   DTW(Rest(P1), P2),          # Option 2
    #                                   DTW(Rest(P1), Rest(P2))     # Option 3
    #                                 }
    headP1 = stepsP1[0]
    headP2 = stepsP2[0]
    restP1 = stepsP1[1:]
    restP2 = stepsP2[1:]
    ev1 = headP1[1]
    ev2 = headP2[1]
    step1 = headP1[0]
    step2 = headP2[0]
    
    # Optimizacion
    if matrixes_opt['matrixDTW_SP'][step1][step2] is not None:
        return matrixes_opt['matrixDTW_SP'][step1][step2]


    sim_HeadP1_HeadP2 = getSimilarity(headP1, headP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
    dist_HeadP1_HeadP2_sp = getSpatialDistance(ev1, ev2, sp_matrix, hu_matrix)
    dist_HeadP1_HeadP2_tmp = getTemporalDistance(step1, step2, maxDiffSteps)*(1-beta)
    dist_HeadP1_HeadP2 =  (alfa*dist_HeadP1_HeadP2_sp) + ((1-alfa)*dist_HeadP1_HeadP2_tmp)

    res_P1_RestP2 = dtw_SpTemp(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)     # Option 1
    res_RestP1_P2 = dtw_SpTemp(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)     # Option 2
    res_RestP1_RestP2 = dtw_SpTemp(restP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)  # Option 3
    
    listResults = [res_P1_RestP2, res_RestP1_P2, res_RestP1_RestP2]
    minDist = 9999999999
    maxSim = 0
    for r in listResults:
        if r[0]<minDist:
            minDist = r[0]
            maxSim = r[1]
        elif r[0]==minDist:
            if r[1] > maxSim:
                maxSim = r[1]
    distance = dist_HeadP1_HeadP2 + minDist
    similarity = sim_HeadP1_HeadP2 + maxSim
    
    # Optimizacion
    matrixes_opt['matrixDTW_SP'][step1][step2] = [distance, similarity]
    
    return [distance, similarity]
