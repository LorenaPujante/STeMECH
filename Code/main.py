

from driverGraphdb import DriverGraphDB

from config import *
from creatorGraph import * 
from creatorMatrix import *
from trajectoriesCreator import *
from trajectoriesMatcher import *
from similarity import *
from similarityCalculator import *
from writer_reader_similarities import *
from clustering import *

import time



########
# MAIN #
########

def main():

    # DRIVER #
    driver = DriverGraphDB()
    driver.setRepository(required_parameters['repository'])

    
    # GRAFOS Y MATRICES DE LOCATIONS Y HUs #
    if required_parameters['zero']:
        sp_matrix, sp_matrix01, hu_matrix = startFromZero(driver)
    else:
        sp_matrix, sp_matrix01, hu_matrix = startFromFiles()



    # TRAJECTORIES #

    # Get patients
    pats, patsString = getPatients(required_parameters['dateStart'], required_parameters['dateEnd'], required_parameters['idLoc'], required_parameters['idMicroorg'], driver)
    print("\npatsString: {}".format(patsString))

    # Set interval time to get trajectories:
        # - Start datetime: First Admission or a calculated date
        # - End datetime: Last TestMicro
    dateStart_trajectories, dateEnd_trajectories = setSearchInterval(required_parameters['dateStart'], required_parameters['dateEnd'], required_parameters['idMicroorg'], patsString, driver)
    print("+ dateStart_trajectories: {}".format(dateStart_trajectories))
    print("+ dateEnd_trajectories: {}".format(dateEnd_trajectories))

    # Get max num steps
    maxNSteps = getTotalSteps(dateStart_trajectories, dateEnd_trajectories)
    print("maxNSteps: {}".format(maxNSteps))

    # Get the Patients' trajectories
    dicTrajectories = getTrajectories(dateStart_trajectories, dateEnd_trajectories, patsString, driver)
    print("\nPATIENTS: ({})\n\t{}".format(len(list(dicTrajectories.keys())), list(dicTrajectories.keys())))
    
    # Get the Patients's last TestMicro
    dicPatTestMicro, dicPatTMStep = getPatientsLastTM(dateStart_trajectories, dateEnd_trajectories, required_parameters['idMicroorg'], patsString, driver)


    # SIMILARITIES #
    maxDiffSteps = maxNSteps-1      # Si hay, por ejemplo, 50 steps ->  La máxima diferencia será: 50-1 = 49    # Este parametro es para la normalizacion de la Distancia Temporal
    arrayDataframes, arrayDataframes_01 = calculateTrajectoriesSimilarity(dicTrajectories, dicPatTMStep, sp_matrix01, hu_matrix, maxDiffSteps, required_parameters['beta'], required_parameters['alfa'], required_parameters['margin'], required_parameters['similarityFunctions'], maxNSteps)
    print("\n\nEVERYTHING IS DONE!")

    # It is not needed since it is already done in calculateTrajectoriesSimilarity()
    # NORMALIZACION DE LAS SIMILITUDES A [0,1]
    '''print("\nNORMALIZANDO SIMILITUDES A [0,1]")
    arrayDataframes_01 = normaliceTo01_arrayDataFrames(arrayDataframes, dicTrajectories, sp_matrix01, maxNSteps)
            

    # GUARDAR EN FICHEROS
    print("\nGUARDANDO DATOS EN FICHEROS")
    writeDataframes(arrayDataframes, False, 1)
    writeDataframes(arrayDataframes_01, True, 1)'''
    
    

        
        
        

#########################################################################################################################################################################################################################################################################################################


#################################
# CREACION DE GRAFOS Y MATRICES #
#################################

def startFromZero(driver):
    
    # Grafo con Locations
    G1, beds = getGraphLocations(driver)
    print("Grafo Locations: CREATED\t - {} Nodes".format(len(G1.nodes)))
    # Matrix con Shortest Path entre Locations
    nameFolder_Matrix = required_parameters['nameFolder_Matrix']
    sp_matrix = getShortestPathLocs_Matrix(G1, beds, nameFolder_Matrix + '/sp_matrix.csv', nameFolder_Matrix + '/sp_header_index.csv')
    print("Shortest path matrix: DONE")
    # Matriz normalizada al rango [0,1]
    sp_matrix01 = getSPMatrix_01(sp_matrix, nameFolder_Matrix + '/sp_matrix_01.csv', nameFolder_Matrix + '/sp_header_index_01.csv')    


    # Grafo con HUs y Services 
    G2 = getGraphLogicalLayout(driver)
    print("Grafo Services: CREATED\t - {} Nodes".format(len(G2.nodes)))
    # Matrix 0/1 si HU/Services conectados
    hu_matrix = get0_1Matrix_HUs(G2, nameFolder_Matrix + '/hu_matrix.csv', nameFolder_Matrix + '/hu_header_index.csv')
    print("0/1 Services matrix: DONE")

    return sp_matrix, sp_matrix01, hu_matrix


def startFromFiles():
    nameFolder_Matrix = required_parameters['nameFolder_Matrix']
    sp_matrix = CSVToMatrix(nameFolder_Matrix + '/sp_matrix.csv', nameFolder_Matrix + '/sp_header_index.csv')
    print("Shortest path matrix: DONE")
    sp_matrix01 = CSVToMatrix(nameFolder_Matrix + '/sp_matrix_01.csv', nameFolder_Matrix + '/sp_header_index_01.csv')
    print("Shortest path matrix: DONE")
    hu_matrix = CSVToMatrix(nameFolder_Matrix + '/hu_matrix.csv', nameFolder_Matrix + '/hu_header_index.csv')
    print("0/1 Services matrix: DONE")

    return sp_matrix, sp_matrix01, hu_matrix




#################
# WRITE IN FILE #
#################

def writeDataframes(arrayDataframes, n01, folder):
    j = 0
    for i in range(len(arrayDataframes)):
        df = arrayDataframes[i]
        if df is not None:
            simMethod = required_parameters['similarityFunctions'][j]
            j += 1


            # Preparar datos para el clustering
            data, listPats = dataframeToArray(df)

            # Guardar datos en un fichero csv
            writeSimilarityMatrixInCSV(data, listPats, simMethod, n01, folder)


##############################################################################################################################################################################################################################################################################################



if __name__ == "__main__":
    main_start = time.time()
    
    main()
    
    main_end = time.time()
    main_diff_seconds = main_end-main_start
    main_diff_minutes = main_diff_seconds/60
    print("\n\nTotal Time: {:.6f} secs\t- {:.6f} mins".format(main_diff_seconds, main_diff_minutes))
    
