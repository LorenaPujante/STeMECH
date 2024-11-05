
from similarityCalculator import *
from config import *

####################################
# SOLO MIRA HACIA ADELANTE Y ATRÁS #
####################################
# SE PRUEBAN TODAS LAS OPCIONES    # 
####################################


# En un step el margen solo se puede utilizar para "mirar hacia atras", nada de comprobar los siguientes steps
def lcss_wtw(stepsP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1, stepsUsed2, margin):
    # Caso Base
    if len(stepsP1)==0 or len(stepsP2)==0:
        return 0, 0
    

    minSpatialSim = getSpatialSimilarity_distKnown(required_parameters['maxSpDist'])
    maxTempDist = getTemporalDistance_diffKnown(required_parameters['maxStepsBackwardLCSS'], maxDiffSteps)
    minTemporalSim = getTemporalSimilarity_distKnown(maxTempDist, beta)
    
    headP1 = stepsP1[0]    
    headP2 = stepsP2[0]    
    restP1 = stepsP1[1:]    
    restP2 = stepsP2[1:]

    # Optimizacion
    step1_bedHU = headP1[1]
    step1_step = headP1[0]
    step2_bedHU = headP2[1]
    step2_step = headP2[0]
    if matrixes_opt['matrixLCSS_2'][step1_step][step2_step] is not None:
        listaSoluciones = matrixes_opt['matrixLCSS_2'][step1_step][step2_step]
        
        stepsUsed1_step = []
        stepsUsed2_step = []
        for s in stepsUsed1:
            stepsUsed1_step.append(s[0])
        for s in stepsUsed2:
            stepsUsed2_step.append(s[0])
        
        found = False
        z = 0
        while z<len(listaSoluciones) and not found:
            solucion = listaSoluciones[z]
            if len(stepsUsed1)==0  and  len(solucion[2])==0:
                return solucion[0], solucion[1]
            if len(stepsUsed1)==0  and  len(solucion[2])!=0:
                z += 1
            elif len(stepsUsed1)!=0  and  len(solucion[2])==0:
                z += 1
            else:
                isIn1 = all(x in solucion[2] for x in stepsUsed1_step)
                isIn2 = all(x in solucion[3] for x in stepsUsed2_step)
                if isIn1 and isIn2:
                    return solucion[0], solucion[1]
                z += 1 

        
        

    # Caso Recursivo 1
    # if    spatial_dist(Head(P1), Head(P2)) >= min_Spatial_Similarity
    #       AND
    #       temporal_dist(Head(P1), Head(P2)) >= min_Temporal_Similarity
    # then
    #       lcss(Rest(P1), Rest(P2)) + 1
    
    totalRes = 1 + 2*margin + 2*margin      # N_Resultados = Head(P1)_Head(P2) + 
                                            #                + [ Head(P1)_Margin_1(P2) + Head(P1)_Margin_2(P2) + ... + Head(P1)_Margin_N(P2) ] + [ Margin_1(P1)_Head(P2) + Margin_2(P1)_Head(P2) + ... + Margin_N(P1)_Head(P2) ]    # (Hacia atras)
                                            #                + [ Head(P1)_Margin_1(P2) + Head(P1)_Margin_2(P2) + ... + Head(P1)_Margin_N(P2) ] + [ Margin_1(P1)_Head(P2) + Margin_2(P1)_Head(P2) + ... + Margin_N(P1)_Head(P2) ]    # (Hacia delante)
                                            # = 1 + margin + margin + margin + margin = 1 + 2*margin + 2*margin 
    similaritiesAndPair = []
    for i in range(totalRes):
        
        # Head(P1)_Head(P2)
        if i == 0:
            step1 = headP1
            step2 = headP2
            if step1 not in stepsUsed1  and  step2 not in stepsUsed2:

                step1_bedHU = step1[1]
                step1_step = step1[0]
                step2_bedHU = step2[1]
                step2_step = step2[0]

                spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
                if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                    similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                    simPair = [similarity, step1, step2]
                    similaritiesAndPair.append(simPair)
                    simPair = [similarity, step1, step2]
                    similaritiesAndPair.append(simPair)


        
        # Se usa el margen hacia atras
        elif i < 1+2*margin: 
            
            # Head(P1) - Margin_prev(P2)
            if i <= margin:
                step1 = headP1
                if step1 not in stepsUsed1:
                    
                    indPrev = i
                    step2_step = headP2[0]
                    step2_prev = step2_step-indPrev
                    j = 0
                    step2 = None
                    while j<len(stepsCompletosP2) and step2 is None:
                        if stepsCompletosP2[j][0] == step2_prev:
                            step2 = stepsCompletosP2[j]
                        else:
                            j += 1
                    
                    if step2 is not None  and  step2 not in stepsUsed2:
                        step2_bedHU = step2[1]
                        step2_step = step2[0]
                        step1_bedHU = headP1[1]
                        step1_step = headP1[0]
                        
                        spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                        tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
                        if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                            similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                            simPair = [similarity, step1, step2]
                            similaritiesAndPair.append(simPair)


            # Margin_prev(P1) - Head(P2)
            else:
                step2 = headP2
                if step2 not in stepsUsed2:
                    indPrev = i - margin
                    step1_step = headP1[0]
                    step1_prev = step1_step-indPrev
                    j = 0
                    step1 = None
                    while j<len(stepsCompletosP1) and step1 is None:
                        if stepsCompletosP1[j][0] == step1_prev:
                            step1 = stepsCompletosP1[j]
                        else:
                            j+= 1
                    
                    if step1 is not None  and  step1 not in stepsUsed1:
                        step1_bedHU = step1[1]
                        step1_step = step1[0]
                        step2_bedHU = headP2[1]
                        step2_step = headP2[0]

                        spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                        tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
                        if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                            similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                            simPair = [similarity, step1, step2]
                            similaritiesAndPair.append(simPair)

        # Se usa el margen hacia delante
        else:   
            # Head(P1) - Margin_post(P2)
            if i <= margin*3:
                step1 = headP1
                if step1 not in stepsUsed1:
                    indPost = i - 2*margin
                    step2_step = headP2[0]
                    step2_post = step2_step+indPost
                    j = 0
                    step2 = None
                    while j<len(stepsCompletosP2) and step2 is None:
                        if stepsCompletosP2[j][0] == step2_post:
                            step2 = stepsCompletosP2[j]
                        else:
                            j += 1
                    
                    if step2 is not None  and  step2 not in stepsUsed2:
                        step2_bedHU = step2[1]
                        step2_step = step2[0]    
                        step1_bedHU = headP1[1]
                        step1_step = headP1[0]
                        
                        spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                        tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
                        if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                            similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                            simPair = [similarity, step1, step2]
                            similaritiesAndPair.append(simPair)
            # Margin_post(P1) - Head(P2)
            else:
                step2 = headP2
                    
                if step2 not in stepsUsed2:
                    indPost = i - 3*margin
                    step1_step = headP1[0]
                    step1_post = step1_step+indPost
                    j = 0
                    step1 = None
                    while j<len(stepsCompletosP1) and step1 is None:
                        if stepsCompletosP1[j][0] == step1_post:
                            step1 = stepsCompletosP1[j]
                        else:
                            j+= 1
                    
                    if step1 is not None  and  step1 not in stepsUsed1:
                        step1_bedHU = step1[1]
                        step1_step = step1[0]
                        step2_bedHU = headP2[1]
                        step2_step = headP2[0]

                        spatialSim = getSpatialSimilarity(step1_bedHU, step2_bedHU, sp_matrix, hu_matrix)
                        tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
                        if spatialSim >= minSpatialSim  and  tempSim >= minTemporalSim:
                            similarity = getSimilarity_simsKnown(spatialSim, tempSim, alfa)
                            simPair = [similarity, step1, step2]
                            similaritiesAndPair.append(simPair)                            

    # Comprobar si alguna de las opciones ha dado resultado
    if len(similaritiesAndPair) != 0:
    
        # Comprobar todos los resultados
        results = []
        for sPair in similaritiesAndPair:

            sim = sPair[0]
            sim_ev1 = sPair[1]
            sim_ev2 = sPair[2]
            
            stepsUsed1_clone = stepsUsed1.copy()
            stepsUsed1_clone.append(sim_ev1)
            stepsUsed2_clone = stepsUsed2.copy()
            stepsUsed2_clone.append(sim_ev2)


            # Habría que mirar en cuál de las dos trayectorias se ha elegido el punto que tocaba y en cuál (si es el caso) se ha elegido uno anterior/posterior
            if headP1[0] == sim_ev1:
                if headP2[0] == sim_ev2:
                    # Se selecciona el step head de ambas trayectorias  -> Se sigue con el siguiente punto en ambas
                    nMatches_rec, similarity_rec = lcss_wtw(restP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1_clone, stepsUsed2_clone, margin)
                else:
                    # Se selecciona el step head de T1, y otro de T2    -> Se sigue con el siguiente de T1 y con el head de T2
                    nMatches_rec, similarity_rec = lcss_wtw(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1_clone, stepsUsed2_clone, margin)
            else:
                # Se selecciona el step head de T2, y otro de T1    -> Se sigue con el siguiente de T2 y con el head de T1
                nMatches_rec, similarity_rec = lcss_wtw(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1_clone, stepsUsed2_clone, margin)

            similarity = sim + similarity_rec
            nMatches = nMatches_rec + 1
            steps1_r1 = []
            steps2_r1 = []
            for s in stepsUsed1_clone:
                steps1_r1.append(s[0])
            for s in stepsUsed2_clone:
                steps2_r1.append(s[0])
    
            result = [nMatches, similarity, steps1_r1, steps2_r1]
            results.append(result)
    
        maxMatches = -1
        maxSim = -1
        steps1Sol_r1 = []
        steps2Sol_r1 = []
        for res in results:
            sim = res[1]
            matches = res[0]
            if matches > maxMatches:
                maxMatches = matches
                maxSim = sim
                steps1Sol_r1 = res[2]
                steps2Sol_r1 = res[3]
            elif matches == maxMatches:
                if sim > maxSim:
                    maxMatches = matches
                    maxSim = sim
                    steps1Sol_r1 = res[2]
                    steps2Sol_r1 = res[3]
        
        nMatches = maxMatches
        similarity = maxSim

        # Optimizacion
        if matrixes_opt['matrixLCSS_2'][step1_step][step2_step] is None:
            matrixes_opt['matrixLCSS_2'][step1_step][step2_step] = []
        solucion = [nMatches,similarity,steps1Sol_r1,steps2Sol_r1]
        matrixes_opt['matrixLCSS_2'][step1_step][step2_step].append(solucion)

        return nMatches, similarity

    
    # Caso Recursivo 2:
    #  max {
    #       LCSS(Rest(P1), P2),     # Option 2
    #       LCSS(P1, Rest(P2))      # Option 3
    #      }

    # OPTIMIZACION: Si la similitud temporal ya es menor de la mínima, no hace falta comprobar las opciones en las que los steps se siguen alejando
    # Es decir: Si vas por el par de steps [0,3] y 'option 1' sería [0,4], donde ya sabes que al ser un step muy alejado no hay probabilidad de contagio, pues nos saltamos la opción y pasamos directamente a 'option 2', que sería [1,3]
    step1_step = headP1[0]
    step2_step = headP2[0]
    tempSim = getTemporalSimilarity(step1_step, step2_step, maxDiffSteps, beta)
    if tempSim < minTemporalSim:
        if headP1[0] > headP2[0]:
            nMatches, similarity = lcss_wtw(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1, stepsUsed2, margin)
        else:    # if headP1[0] < headP2[0]:
            nMatches, similarity = lcss_wtw(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1, stepsUsed2, margin)
    
    else:
        nMatches_opt1, sim_RestP1_P2 = lcss_wtw(restP1, stepsP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1, stepsUsed2, margin)   # Option 1
        nMatches_opt2, sim_P1_RestP2 = lcss_wtw(stepsP1, restP2, sp_matrix, hu_matrix, maxDiffSteps, beta, alfa, stepsCompletosP1, stepsCompletosP2, stepsUsed1, stepsUsed2, margin)   # Option 2
        
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
    steps1_r2 = []
    steps2_r2 = []
    for s in stepsUsed1:
        steps1_r2.append(s[0])
    for s in stepsUsed2:
        steps2_r2.append(s[0])
    if matrixes_opt['matrixLCSS_2'][step1_step][step2_step] is None:
        matrixes_opt['matrixLCSS_2'][step1_step][step2_step] = []
    solucion = [nMatches,similarity,steps1_r2,steps2_r2]
    matrixes_opt['matrixLCSS_2'][step1_step][step2_step].append(solucion)

    return nMatches, similarity
