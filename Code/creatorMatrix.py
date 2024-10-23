import networkx as nx
import pandas as pd
import numpy as np
import csv


#########################
# FUNCIONES PRINCIPALES #
#########################

# Crear matriz con el coste del shortest path entre las Locs
def getShortestPathLocs_Matrix(G, beds, nameFile_matrix, nameFile_headers):
    
    listNodes = G.nodes
    #df = pd.DataFrame(np.zeros((len(listNodes), len(listNodes))), columns=listNodes, index=listNodes)
    df = pd.DataFrame(np.zeros((len(beds), len(beds))), columns=beds, index=beds)
    for i in range(len(df)):
        row = df.index.values[i]
        for j in range(i,len(df.columns)):
            col = df.columns[j]
            if row != col:  # Si son la misma Loc -> El coste seguirá siendo 0.0
                sp_cost = getShortestPathCost(G, row, col)
                df[row][col] = sp_cost
                df[col][row] = sp_cost
        
    print("SHORTEST PATH DONE")
    matrixToCSV(df, nameFile_matrix, nameFile_headers)    
    
    return df


# Crear matriz en la que se indica si 2 HUs están conectadas por el Servicio
def get0_1Matrix_HUs(G, nameFile_matrix, nameFile_headers):
    listNodes = []
    # Solo se añaden HUs
    for (p, d) in G.nodes(data=True):
        if 'HospitalizationUnit' in d['uri']:
            listNodes.append(p)
    
    df = pd.DataFrame(np.zeros((len(listNodes), len(listNodes))), columns=listNodes, index=listNodes, dtype=int)
    for i in range(len(df)):
        row = df.index.values[i]
        for j in range(i,len(df.columns)):
            col = df.columns[j]
            
            sp_cost = 0         # Si NO son la misma HU ni del mismo Servicio -> Coste = 0
            if row == col:  
                sp_cost = 1     # Si son la misma HU -> Coste = 1
            else:               # Si NO son la misma HU pero forman parte del mismo Servicio -> Coste = 2
                if nx.has_path(G, row, col):
                    sp_cost = 2
                
            df[row][col] = sp_cost
            df[col][row] = sp_cost

    print("HUs MATRIX DONE")

    matrixToCSV(df, nameFile_matrix, nameFile_headers) 

    return df



########################
# FUNCIONES AUXILIARES #
########################

# Crear Ficheros CSV a partir de una Matriz
def matrixToCSV(df, nameFile_matrix, nameFile_headers):
    # Fichero CSV con la matriz sin el nombre de las columnas/filas 
    df.to_csv(nameFile_matrix, index=False, header=False)           

    # Fichero CSV con los nombres de las columnas/filas
    listHeader = df.columns.values.tolist()
    with open(nameFile_headers,"w") as f: 
        wr = csv.writer(f)
        wr.writerow(listHeader)


# Crear una Matriz a partir de Ficheros CSV
def CSVToMatrix(nameFile_matrix, nameFile_headers):
    # Lectura de la matriz sin el nombre de las columnas/filas
    df = pd.read_csv(nameFile_matrix, header=None)
    
    # Lectura del fichero con el nombre de las columnas/filas
    file = open(nameFile_headers, "r")
    header = list(csv.reader(file, delimiter=","))  
    header = header[0]  
    
    df.columns = header         # Nombre de las columnas
    df = df.set_axis(header)    # Nombre de las filas

    return df
    

# SHORTEST PATH #
def getShortestPathCost(G, source, target):
    try:
        sp_cost = nx.shortest_path_length(G, source=source, target=target, weight='cost')
    except:
        sp_cost = 200

    return sp_cost




####################################################
# NORMALIZACION [0,1] DE LOS COSTES DE LAS ARISTAS #
####################################################

def getSPMatrix_01(sp_matrix, nameFile_matrix, nameFile_headers):
    
    sp_matrix01 = sp_matrix.copy()

    maxValue = getMaxValue_SPMatrix(sp_matrix01)
    for i in range(len(sp_matrix01)):
        row = sp_matrix01.index.values[i]
        for j in range(i,len(sp_matrix01.columns)):
            col = sp_matrix01.columns[j]
            
            normValue = sp_matrix01[row][col]/maxValue
            sp_matrix01[row][col] = normValue
            sp_matrix01[col][row] = normValue
                
    print("NORMALIZING SHORTEST PATH MATRIX DONE")
    
    matrixToCSV(sp_matrix01, nameFile_matrix, nameFile_headers)
    
    return sp_matrix01  

def getMaxValue_SPMatrix(sp_matrix):
    maxValue = 0
    for i in range(len(sp_matrix)):
        row = sp_matrix.index.values[i]
        for j in range(i,len(sp_matrix.columns)):
            col = sp_matrix.columns[j]
            
            if sp_matrix[row][col] > maxValue:
                maxValue = sp_matrix[row][col]

    return maxValue
