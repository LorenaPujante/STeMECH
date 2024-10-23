
from similarityCalculator import *
from config import *

def dtw(stepsP1, stepsP2, sp_matrix, hu_matrix):
    
    # Caso base     # Se reduce a uno solo
    if len(stepsP1)==0 or len(stepsP2)==0:
        return 0    # Nada de similitud
    
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
    if matrixes['matrixDTW'][step1][step2] is not None:
        return matrixes['matrixDTW'][step1][step2]
    
    sim_HeadP1_HeadP2 = getSpatialSimilarity(ev1, ev2, sp_matrix, hu_matrix)
    
    sim_P1_RestP2 = dtw(stepsP1, restP2, sp_matrix, hu_matrix)      # Option 1
    sim_RestP1_P2 = dtw(restP1, stepsP2, sp_matrix, hu_matrix)      # Option 2
    sim_RestP1_RestP2 = dtw(restP1, restP2, sp_matrix, hu_matrix)   # Option 3
    listSimilarities = [sim_P1_RestP2, sim_RestP1_P2, sim_RestP1_RestP2]
    maxSimilarity = max(listSimilarities)

    similarity = sim_HeadP1_HeadP2 + maxSimilarity
    # Optimizacion
    matrixes['matrixDTW'][step1][step2] = similarity
    
    return similarity