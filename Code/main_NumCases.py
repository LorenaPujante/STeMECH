

from datetime import datetime
import datetime as dt

from driverGraphdb import DriverGraphDB
from queries_numTests import *
from config import *




##########
# PARAMS #
##########

repository = required_parameters['repository']
maxDaysTrajForward = 7
beta = 0.5
alfa = 0.5



##################
# MAIN NUM CASES #
##################

def main_numCases():
    # DRIVER
    driver = DriverGraphDB()
    driver.setRepository(repository)

    maxDate = getLastDate(driver)
    minDate = getFirstDate(driver)
    print("minDate: {}".format(minDate))
    print("maxDate: {}".format(maxDate))
    
    
    # Tests cada 7 días
    getTestMicroBy_7Days(driver, minDate, maxDate)
    print()
    getTestMicroBy_7Days_byFloor(driver, minDate, maxDate)
    print()
    getTestMicroBy_7Days_byFloor_lax(driver, minDate, maxDate)






############################################################################
# FUNCIONES PARA OBTENER PRIMERA Y ULTIMA FECHA Y HORA DE LA BASE DE DATOS #
############################################################################

def getLastDatetime(driver):
    q = QueryDates()
    query = q.getLastDate()
    results = driver.executeQuery(query)
    res = results['results']['bindings'][0]
    maxDatetime = res['maxDate']['value']
    maxDatetime = datetime.strptime(maxDatetime, "%Y-%m-%dT%H:%M:%S")

    return maxDatetime

def getLastDate(driver):
    maxDatetime = getLastDatetime(driver)
    maxDate = maxDatetime.replace(hour=0, minute=0, second=0) 

    return maxDate


def getFirstDatetime(driver):
    q = QueryDates()
    query = q.getFirstDate()
    results = driver.executeQuery(query)
    res = results['results']['bindings'][0]
    minDatetime = res['minDate']['value']
    minDatetime = datetime.strptime(minDatetime, "%Y-%m-%dT%H:%M:%S")

    return minDatetime

def getFirstDate(driver):
    minDatetime = getFirstDatetime(driver)
    minDate = minDatetime.replace(hour=0, minute=0, second=0) 

    return minDate



###########################################################
# FUNCIONES AUXILIARES PARA ELEGIR PARAMETROS DE BUSQUEDA #
###########################################################

def getTestMicroBy_7Days(driver, minDate, maxDate):
    
    q = QueryNumTestMicro()
    dateStart = minDate

    print("\nNumber of Positive TestMicro by 7 days:")

    continuar = True
    while continuar:
        dateEnd = dateStart + dt.timedelta(days=7)
        dateEnd = dateEnd - dt.timedelta(seconds=1)
        dateStart_string = dateStart.strftime("%Y-%m-%dT%H:%M:%S")
        dateEnd_string = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
        query = q.getQuery_7days(dateStart_string, dateEnd_string)

        results = driver.executeQuery(query)
        for res in results['results']['bindings']:
            idMicroorg = res['m_id']['value']
            numTM = res['count']['value']
            print(" ({})\tStart: {}\t- End: {}\t->\t{} cases".format(idMicroorg, dateStart.strftime("%d/%m"), dateEnd.strftime("%d/%m"), numTM))

        # NextDate
        dateStart = dateStart + dt.timedelta(days=1)
        if dateStart > maxDate:
            continuar = False

def getTestMicroBy_7Days_byFloor(driver, minDate, maxDate):
    
    q = QueryNumTestMicro()
    dateStart = minDate

    print("\nNumber of Positive TestMicro by 7 days by Floor:")

    continuar = True
    while continuar:
        dateEnd = dateStart + dt.timedelta(days=7)
        dateEnd = dateEnd - dt.timedelta(seconds=1)
        dateStart_string = dateStart.strftime("%Y-%m-%dT%H:%M:%S")
        dateEnd_string = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
        query = q.getQuery_7days_byFloor(dateStart_string, dateEnd_string)

        results = driver.executeQuery(query)
        for res in results['results']['bindings']:
            idMicroorg = res['m_id']['value']
            numTM = res['count']['value']
            floor = res['floor_desc']['value']
            floor_id = res['floor_id']['value']
            print(" ({})\tStart: {}\t- End: {}\t- Floor: {} ({})\t->\t{} cases".format(idMicroorg, dateStart.strftime("%d/%m"), dateEnd.strftime("%d/%m"), floor, floor_id, numTM))
        print(" ---------------------------------------------------------")

        # NextDate
        dateStart = dateStart + dt.timedelta(days=1)
        if dateStart > maxDate:
            continuar = False

def getTestMicroBy_7Days_byFloor_lax(driver, minDate, maxDate):
    
    q = QueryNumTestMicro()
    dateStart = minDate

    print("\nNumber of Positive TestMicro by 7 days by Floor (lax):")

    continuar = True
    while continuar:
        dateEnd = dateStart + dt.timedelta(days=7)
        dateEnd = dateEnd - dt.timedelta(seconds=1)
        dateStart_string = dateStart.strftime("%Y-%m-%dT%H:%M:%S")
        dateEnd_string = dateEnd.strftime("%Y-%m-%dT%H:%M:%S")
        query = q.getQuery_7days_byFloor_lax(dateStart_string, dateEnd_string)

        results = driver.executeQuery(query)
        for res in results['results']['bindings']:
            idMicroorg = res['m_id']['value']
            numTM = res['count']['value']
            floor = res['floor_desc']['value']
            floor_id = res['floor_id']['value']
            print(" ({})\tStart: {}\t- End: {}\t- Floor: {} ({})\t->\t{} cases".format(idMicroorg, dateStart.strftime("%d/%m"), dateEnd.strftime("%d/%m"), floor, floor_id, numTM))
        print(" ---------------------------------------------------------")

        # NextDate
        dateStart = dateStart + dt.timedelta(days=1)
        if dateStart > maxDate:
            continuar = False



if __name__ == "__main__":
    main_numCases()
