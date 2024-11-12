
from config import *
from createDataframe import *


############################
# PREPARACION DE LOS DATOS #
# LECTURA / ESCRITURA      #
############################

def writeSimilarityMatrixInCSV(data, listPats, simMethod, n01):
    nameFolder_SimArrays = required_parameters['nameFolder_SimArrays']
    fileName = nameFolder_SimArrays + '/' + getSimilarityMethodName_Short(simMethod)
    if n01:
        fileName += '_01'
    fileName += '.csv'
    f = open(fileName, "w")
    
    # Write labels
    line = "patStart,"
    for pat in listPats:
        line += "{},".format(pat)
    line = line[:-1]
    line += '\n'
    f.write(line)
    
    # Write similarities
    for z in range(len(listPats)):
        line = '{},'.format(listPats[z])
        dataPat = data[z]
        for dp in dataPat:
            line += '{},'.format(dp)
        line = line[:-1]
        line += '\n'
        f.write(line)
    
    f.close()


def readCSVToSimilarityMatrix(simMethod):
    nameFolder_SimArrays = required_parameters['nameFolder_SimArrays']
    fileName = nameFolder_SimArrays + '/' + getSimilarityMethodName_Short(simMethod)
    if required_parameters['normalice01']:
        fileName += '_01'
    fileName += '.csv'
    
    data = []
    pats = []
    with open(fileName) as file:
        for lineString in file:
            
            lineString = str(lineString)
            line = lineString.split(',')

            if line[0] == 'patStart':
                numPats = len(line)-1
                for i in range(numPats):
                    j = i+1
                    pat = line[j]
                    if j==numPats:
                        pat = pat[:-1]  # Para quitar el salto de linea
                    pats.append(pat)
            else:
                dataPat = []
                for i in range(numPats):
                    j = i+1
                    sim = float(line[j])
                    dataPat.append(sim)
                data.append(dataPat)
            
    file.close()
    
    return data, pats


def readCSVToSimilarityMatrix_df(simMethod):

    nameFolder_SimArrays = required_parameters['nameFolder_SimArrays']
    fileName = nameFolder_SimArrays + '/' + getSimilarityMethodName_Short(simMethod)
    if required_parameters['normalice01']:
        fileName += '_01'
    fileName += '.csv'


    listPats = []
    indPat1 = 0
    df = None
    with open(fileName) as file:
        for lineString in file:
            
            lineString = str(lineString)
            line = lineString.split(',')

            if line[0] == 'patStart':
                numPats = len(line)-1
                for i in range(numPats):
                    j = i+1
                    pat = line[j]
                    if j==numPats:
                        pat = pat[:-1]  # Para quitar el salto de linea
                    listPats.append(pat)

                df = createDataframePatXPatZeros_givenListPats(listPats)
            
            else:
                # Read data
                dataPat = []
                for i in range(numPats):
                    j = i+1
                    sim = float(line[j])
                    dataPat.append(sim)

                # pasar datos a dataframe
                indPat2 = 0
                pat1 = listPats[indPat1]
                for sim in dataPat:
                    pat2 = listPats[indPat2]
                    df[pat1][pat2] = sim
                    indPat2 += 1
                indPat1 += 1

    return df
            


###########################
# ESCRITURA DE RESULTADOS #
###########################

def writeResultsClustering_allKs(allValidationScores, minClusters, maxClusters):
    nameFolder = required_parameters['nameFolder_Outputs']
    fileName = nameFolder + '/' + "clustering_validationScores_Ks.txt"

    file = open(fileName, "w")

    for similarityLabel, validationScores in allValidationScores.items():
        line = "* {}\n".format(similarityLabel)
        file.write(line)
        for values in validationScores:
            line = "   + {}\n".format(values[0])
            file.write(line)
            numTestClusters = maxClusters-minClusters + 1
            for i in range(numTestClusters):
                k = i+minClusters
                line = "      - {}: {}\n".format(k, values[1][i])
                file.write(line) 
            file.write('\n')
        file.write('\n')

    file.close()


def writeResultsClustering_allKs_sil(allPairsSilhouette, minClusters, maxClusters):
    nameFolder = required_parameters['nameFolder_Outputs']
    fileName = nameFolder + '/' + "clustering_SilhouetteScores_Ks.txt"

    file = open(fileName, "w")

    for similarityLabel, silScores in allPairsSilhouette.items():
        line = "* {}\n".format(similarityLabel)
        file.write(line)
        numTestClusters = maxClusters-minClusters + 1
        for i in range(numTestClusters):
            k = i+minClusters
            line = "   - {}: {}\n".format(k, silScores[1][i])
            file.write(line) 
        file.write('\n')

    file.close()


def writeResultsClustering_specificK(similarityMethod, k, dictClusters_pairs, cluster_centroids):
    nameFolder = required_parameters['nameFolder_Outputs']
    fileName = nameFolder + '/' + "clustering_{}_K{}.txt".format(getSimilarityMethodName_Short(similarityMethod), k)

    file = open(fileName, "w")

    clusterLabels = list(dictClusters_pairs.keys())
    clusterLabels.sort()
    line = '* {}:\n'.format(getSimilarityMethodName_Short(similarityMethod))
    file.write(line)
    for clusterLabel in clusterLabels:
        patients = dictClusters_pairs[clusterLabel]
        line = "   + Cluster {}: ({} pats)\n".format(clusterLabel+1, len(patients))
        file.write(line)
        for pat in patients:
            line = "      - {}: {}\n".format(pat[0], pat[1])
            file.write(line)
        line = "         centroid: {}\n\n".format(cluster_centroids[clusterLabel].tolist()) 
        file.write(line)
    
    file.close()
