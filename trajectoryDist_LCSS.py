from similarityCalculator import *
from config import *



def lcss(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa):
    
    # Caso Base
    if len(stepsP1)==0 or len(stepsP2)==0:
        return 0, 0
    
    minSpatialSim = getSpatialSimilarity_distKnown(39.5/102.5)         # DEFINICION DE BROTE   # Distancia hasta Area -> Zona Lógica           # 39.5 = Distancia Bed-LZ-Bed   # 102.5 = Valor máximo en la Matriz de Locations
    maxTempDist = getTemporalDistance_diffKnown(required_parameters['maxStepsBackwardLCSS'], maxDiffSteps)
    minTemporalSim = getTemporalSimilarity_distKnown(maxTempDist, beta)
                                
    headP1 = stepsP1[0]    
    headP2 = stepsP2[0]    
    restP1 = stepsP1[1:]    
    restP2 = stepsP2[1:]
    
    step1_bedHU = headP1[1]
    step1_step = headP1[0]
    step2_bedHU = headP2[1]
    step2_step = headP2[0]

    # Optimizacion
    if matrixes['matrixLCSS'][step1_step][step2_step] is not None:
        #print("{}_{}: Not None".format(step1_step, step2_step))
        result = matrixes['matrixLCSS'][step1_step][step2_step]
        return result[0], result[1]
    #print("{}_{}: None".format(step1_step, step2_step))

    # Caso Recursivo 1
    # if    spatial_dist(Head(P1), Head(P2)) >= min_Spatial_Similarity
    #       AND
    #       temporal_dist(Head(P1), Head(P2)) >= min_Temporal_Similarity
    # then
    #       lcss(Rest(P1), Rest(P2)) + 1
    spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
    tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
    if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
        #print("{}_{}: Recursivo 1".format(step1_step, step2_step))
        similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
        nMatches_rec, similarity_rec = lcss(restP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
        similarity += similarity_rec
        nMatches = nMatches_rec + 1
        # Optimizacion
        result = (nMatches, similarity)
        matrixes['matrixLCSS'][step1_step][step2_step] = result
        #print("{}_{}: ({}, {})".format(step1_step, step2_step, nMatches, similarity))
        return nMatches, similarity
    

    # Caso Recursivo 2
    #  max {
    #       LCSS(Rest(P1), P2),     # Option 2
    #       LCSS(P1, Rest(P2))      # Option 3
    #      }
    
    # OPTIMIZACION: Si la similitud temporal ya es menor de la mínima, no hace falta comprobar las opciones en las que los steps se siguen alejando
        # Es decir: Si vas por el par de steps [0,3] y 'option 1' sería [0,4], donde ya sabes que al ser un step muy alejado no hay probabilidad de contagio, pues nos saltamos la opción y pasamos directamente a 'option 2', que sería [1,3]
    if tempSim < minTemporalSim:
        if headP1[0] > headP2[0]:
            #print("{}_{}: Recursivo 2 opt SR".format(step1_step, step2_step))
            nMatches, similarity = lcss(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
        else:    # if headP1[0] < headP2[0]:
            #print("{}_{}: Recursivo 2 opt RS".format(step1_step, step2_step))
            nMatches, similarity = lcss(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)
    
    else:
        #print("{}_{}: Recursivo 2 RS".format(step1_step, step2_step))
        nMatches_opt1, sim_RestP1_P2 = lcss(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)   # Option 1
        #print("{}_{}: Recursivo 2 SR".format(step1_step, step2_step))
        nMatches_opt2, sim_P1_RestP2 = lcss(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa)   # Option 2
        if nMatches_opt1 > nMatches_opt2:
            nMatches = nMatches_opt1
            similarity = sim_RestP1_P2
        elif nMatches_opt2 > nMatches_opt1:
            nMatches = nMatches_opt2
            similarity = sim_P1_RestP2
        else:
            if sim_RestP1_P2 > sim_P1_RestP2:
                nMatches = nMatches_opt1
                similarity = sim_RestP1_P2
            else:
                nMatches = nMatches_opt2
                similarity = sim_P1_RestP2


    # Optimizacion
    result = (nMatches,similarity)
    matrixes['matrixLCSS'][step1_step][step2_step] = result
    #print("{}_{}: ({}, {})".format(step1_step, step2_step, nMatches, similarity))
    return nMatches, similarity




