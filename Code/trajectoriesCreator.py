import datetime as dt
from datetime import datetime, timedelta
import math

from config import *
from queriesTrajectories import *



##########
# CLASES #
##########

class Event:
    def __init__(self, id, start, end, bed, hu, typeEvnt, idPat):
        self.id = id
        self.start = start
        self.end = end
        self.bed = bed
        self.hu = hu
        self.typeEvnt = typeEvnt
        self.idPat = idPat

class BedHU:
    def __init__(self, bed, hu):
        self.bed = bed
        self.hu = hu
        


########################################################
# FUNCIONES PRINCIPALES  PARA OBTENER LAS TRAYECTORIAS #
########################################################

def getPatients(dateStart, dateEnd, idLoc, idMicroorg, driver):
    q = Q2()
    query = q.getQuery(dateStart, dateEnd, idLoc, idMicroorg)
    results = driver.executeQuery(query)
    pats = []
    for res in results['results']['bindings']:
        idPat = res['p_id']['value']
        pats.append(idPat)
    
    patsString = "("
    for i in range(len(pats)-1):
        idPat = pats[i]
        patsString += "{}, ".format(idPat)
    patsString += "{})".format(pats[len(pats)-1])

    return pats, patsString


def setSearchInterval(dateStart, dateEnd, idMicroorg, patsString, driver):
    
    # Get Last TestMicro of the Patients (the datetime)
    lastTestDate = getLastTestMicroDatetime(dateStart, dateEnd, idMicroorg, patsString, driver)
    
    # Get the First Admission of the Patients (the datetime) 
    firstEpDate = getFirstAdmissionDatetime(patsString, driver)
    
    # Get Minimum dateStart for trajectories
    dateStart_date = datetime.strptime(dateStart, "%Y-%m-%dT%H:%M:%S")
    minDateStart = dateStart_date - dt.timedelta(days=required_parameters['maxDaysTrajForward'])
    
    # Set dates for search trajectories
    dateEnd_trajectories = lastTestDate
    if firstEpDate < minDateStart:
        dateStart_trajectories = minDateStart
    else:
        dateStart_trajectories = firstEpDate
    
    # dateStart (and dateEnd) must match the start (or end) time of a step
    dateStart_trajectories = setTimeFirstStep(dateStart_trajectories)    
    dateEnd_trajectories = setTimeLastStep(dateEnd_trajectories)
    
    return dateStart_trajectories, dateEnd_trajectories


def getTrajectories(dateStart_trajectories, dateEnd_trajectories, patsString, driver):
    q = QueryGetEventsFromPatsQ2()
    query = q.getQuery(dateStart_trajectories, dateEnd_trajectories, patsString)
    results = driver.executeQuery(query)
    dicTrajectories = getEventsFromTrajectories(results, dateStart_trajectories, dateEnd_trajectories)

    return dicTrajectories


def getPatientsLastTM(dateStart_trajectories, dateEnd_trajectories, idMicroorg, patsString, driver):
    
    dicPatTestMicro = {}
    q = QueryGetLastTest()
    query = q.getQuery_byPatient(dateEnd_trajectories, idMicroorg, patsString)
    results = driver.executeQuery(query)
    for res in results['results']['bindings']:
        patId = res['p_id']['value']
        tm_date = res['lastTMDate']['value']
        tm_date = datetime.strptime(tm_date, "%Y-%m-%dT%H:%M:%S")
        dicPatTestMicro[patId] = tm_date

    # Get step of the TestMicro
    dicPatTMStep = {}
    for pat, tm in dicPatTestMicro.items():
        step = getStep(tm, dateStart_trajectories)
        dicPatTMStep[pat] = step

    return dicPatTestMicro, dicPatTMStep



    #----------------------#
    # FUNCIONES AUXILIARES #
    #----------------------#

def getLastTestMicroDatetime(dateStart, dateEnd, idMicroorg, patsString, driver): 
    q = QueryGetLastTest()
    query = q.getQuery(dateStart, dateEnd, idMicroorg, patsString)
    results = driver.executeQuery(query)
    res = results['results']['bindings'][0]
    lastTestDate = res['lastTMDate']['value']
    lastTestDate = datetime.strptime(lastTestDate, "%Y-%m-%dT%H:%M:%S")
    
    return lastTestDate

def getFirstAdmissionDatetime(patsString, driver):
    q = QueryGetFirstEpisode()
    query = q.getQuery(patsString)
    results = driver.executeQuery(query)
    res = results['results']['bindings'][0]
    firstEpDate = res['firstEpDate']['value']
    firstEpDate = datetime.strptime(firstEpDate, "%Y-%m-%dT%H:%M:%S")

    return firstEpDate


def setTimeFirstStep(dateStart_trajectories: datetime):
    t1 = dt.time(8, 0, 0)
    t2 = dt.time(16, 0, 0)
    time_dateStart = dateStart_trajectories.time()
    date_dateStart = dateStart_trajectories.date()
    if time_dateStart < t1:
        t = dt.time(0, 0, 0)
        dtCombined = datetime.combine(date_dateStart, t)
    elif time_dateStart < t2: 
        t = dt.time(8, 0, 0)
        dtCombined = datetime.combine(date_dateStart, t)
    else:
        t = dt.time(16, 0, 0)
        dtCombined = datetime.combine(date_dateStart, t)
    
    return dtCombined

def setTimeLastStep(dateEnd_trajectories: datetime):
    t1 = dt.time(8, 0, 0)
    t2 = dt.time(16, 0, 0)
    time_dateEnd = dateEnd_trajectories.time()
    date_dateEnd = dateEnd_trajectories.date()
    if time_dateEnd >= t2:
        t = dt.time(23, 59, 59)
        dtCombined = datetime.combine(date_dateEnd, t)
    elif time_dateEnd >= t1:
        t = dt.time(15, 59, 59)
        dtCombined = datetime.combine(date_dateEnd, t)
    else:
        t = dt.time(7, 59, 59)
        dtCombined = datetime.combine(date_dateEnd, t)
    dateEnd_trajectories = dtCombined

    return dtCombined



    
######################################
# CREAR DICCIONARIOS PACIENTE-EVENTO #
######################################
 
def getEventsFromTrajectories(trajectories, dateStart, dateEnd):
    dicPatEvents = getDicPatEvents(trajectories, dateStart, dateEnd)
    dicTrajectories = getDicTrajectoriesWithSteps(dicPatEvents, dateStart)

    return dicTrajectories

def getDicPatEvents(results, dateStart, dateEnd):
    
    dicPatsEvents = {}
    
    idPatAnt = results['results']['bindings'][0]['p_id']['value']
    patEvents = []
    for res in results['results']['bindings']:
        idPat = res['p_id']['value']

        evUri = res['ev']['value']
        id, typeEvent = getEventIdAndType(evUri)
        bed = res['bed_id']['value']
        hu = res['hu_id']['value']

        # Solo queremos comparar las trayectorias en el periodo definido
        start = getDatetimeFromString(res['ev_start']['value'])
        if start < dateStart:
            start = dateStart
        end = getDatetimeFromString(res['ev_end']['value'])   
        if end > dateEnd:
            end = dateEnd 
        
        event = Event(id, start, end, bed, hu, typeEvent, idPat)

        if idPat == idPatAnt:
            patEvents.append(event)
        else:
            dicPatsEvents[idPatAnt] = patEvents

            patEvents = []
            patEvents.append(event)
            idPatAnt = idPat

    dicPatsEvents[idPat] = patEvents
    
    return dicPatsEvents

def getEventIdAndType(evUri):
    uriSplit = evUri.split("/")
    id = uriSplit[-1]
    
    typeEvent = uriSplit[-2]
    uriSplit2 = typeEvent.split("#")
    typeEvent = uriSplit2[-1]

    return id, typeEvent

def getDatetimeFromString(stringDate):
    dateRes = stringDate.replace("T", " ")
    dateRes = datetime.strptime(dateRes, '%Y-%m-%d %H:%M:%S')
    return dateRes


def getDicTrajectoriesWithSteps(dicPatEvents, dateStart):
    
    dicTrajectories = {}
    for p in dicPatEvents.keys():
        dicTrajectories[p] = []

        events = dicPatEvents[p]
        for ev in events:
            bedHu = BedHU(ev.bed, ev.hu)
            startStep, endStep = getStepsFromEvent(ev, dateStart)
            for i in range(startStep, endStep+1):
                pairStep = (i, bedHu)
                dicTrajectories[p].append(pairStep)
    
    return dicTrajectories


#############################################
# OPERACIONES TEMPORALES Y CALCULO DE STEPS #
#############################################

def getTotalSteps(dateStart, dateEnd):
    diff = dateEnd - dateStart
    diffDays = diff.total_seconds()/86400      # 60*60*24
    nSteps = math.ceil(diffDays*3)     # Cada step son 8 horas -> Cada día tiene 3 steps
    
    return nSteps    


def getStep(dateStep, dateStart):
    diff = dateStep-dateStart
    diffDays = diff.total_seconds()/86400      # 60*60*24
    step = math.floor(diffDays*3) #math.ceil(diffDays*3)
    
    return step


def getStepsFromEvent(ev, firstDate):
    startStep = getStep(ev.start, firstDate)
    endStep = getStep(ev.end, firstDate)
    
    return startStep, endStep



