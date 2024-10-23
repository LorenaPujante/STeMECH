
########
# MAIN #
########

def matchTrajectoriesLength(traj1, traj2):
    newTraj1 = traj1.copy()
    newTraj2 = traj2.copy()

    # Ambas trayectorias tienen los mismos steps
    if traj1[0][0] == traj2[0][0]  and  len(traj1) == len(traj2):
        return newTraj1, newTraj2
    
    # Una trayectoria empieza antes de que termine la otra
    if traj1[len(traj1)-1][0] < traj2[0][0]  or  traj2[len(traj2)-1][0] < traj1[0][0]:
        return None, None

    # Una de las trayectorias empieza antes
    if traj1[0][0] != traj2[0][0]:
        if traj1[0][0] < traj2[0][0]:
            newTraj1 = removeFirstElementsFromTrajectory(newTraj1, newTraj2)
        else:
            newTraj2 = removeFirstElementsFromTrajectory(newTraj2, newTraj1)

    # Una de las trayectorias termina después
    if traj1[len(traj1)-1][0] != traj2[len(traj2)-1][0]:
        if traj1[len(traj1)-1][0] < traj2[len(traj2)-1][0]:
            newTraj2 = removeLastElementsFromTrajectory(newTraj2, newTraj1)
        else:
            newTraj1 = removeLastElementsFromTrajectory(newTraj1, newTraj2)

    return newTraj1, newTraj2 


def setSameLengthToTrajectories(dicTrajectories, dicPatTMStep, pat1, pat2):

    # Se igualan ambas trayectorias
    traj1 = dicTrajectories[pat1]
    traj2 = dicTrajectories[pat2]
    newTraj1, newTraj2 = matchTrajectoriesLength(traj1, traj2)
    if newTraj1 is None or newTraj2 is None:
        return None, None

    # Además de igualar las trayectorias, también hay que cortarlas de largo donde se da el último TestMicro    ->  Se recuerda que se quiere saber por qué 2 pacientes se han contagiado. Una vez los dos están infecciosos, ya no hace falta seguir investigando 
    tmPat1 = dicPatTMStep[pat1]
    tmPat2 = dicPatTMStep[pat2]
    if tmPat1 < tmPat2:
        tmStep = tmPat2
    else:
        tmStep = tmPat1
    
    lenTraj1 = len(newTraj1)
    lenTraj2 = len(newTraj2)
    if newTraj1[lenTraj1-1][0] > tmStep:
        staySteps = tmStep-newTraj1[0][0]+1
        newTraj1 = newTraj1[:staySteps]
    if newTraj2[lenTraj2-1][0] > tmStep:
        staySteps = tmStep-newTraj2[0][0]+1
        newTraj2 = newTraj2[:staySteps]

    return newTraj1, newTraj2


######################################################################
# FUNCIONES AUXILIARES PARA IGUALAR LA DISTANCIA DE LAS TRAYECTORIAS #
######################################################################

def removeFirstElementsFromTrajectory(traj1, traj2):
    diff = traj2[0][0] - traj1[0][0]
    traj1 = traj1[diff:]
    
    return traj1
    
def removeLastElementsFromTrajectory(traj1, traj2):
    lenTraj1 = len(traj1)
    lenTraj2 = len(traj2)
    diff = traj1[lenTraj1-1][0] - traj2[lenTraj2-1][0]
    traj1 = traj1[:-diff]

    return traj1



def getMaxTrajectoryLength(dicPatEventsStep):
    maxEvents = 0
    for events in dicPatEventsStep.values():
        if len(events) > maxEvents:
            maxEvents = len(events)

    return maxEvents
