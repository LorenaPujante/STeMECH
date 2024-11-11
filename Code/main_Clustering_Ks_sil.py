
from config import *
from writer_reader_similarities import *
from clustering_sil import *
from plot import *


def main():

    # READ DATA
    arrayDataframes = []
    for simMethod in required_parameters['similarityFunctions']:
        df = readCSVToSimilarityMatrix_df(simMethod)
        arrayDataframes.append(df)

    # CLUSTERING    
    minClusters = 2
    maxClusters = getMaxNumClustersToTry(arrayDataframes, minClusters)
    print("min: {}  -  max: {}".format(minClusters, maxClusters))
        

    allPairsSilhouette = check_ValidationScores_ForRangeKClusters_allSimilarities(arrayDataframes, required_parameters['similarityFunctions'], minClusters, maxClusters, required_parameters_clustering['numRows'])
    writeResultsClustering_allKs_sil(allPairsSilhouette, minClusters, maxClusters)




#----------------------#
# FUNCIONES AUXILIARES #
#----------------------#

def getMaxNumClustersToTry(data, minClusters):
    i = 0
    found = False
    numPats = 0
    while i<len(data) and not found:
        d = data[i]
        if d is not None:
            found = True
            numPats = len(d.index)
        else:
            i += 1

    if required_parameters_clustering['maxClustersPats']:  # Se probaran hasta tantos clusters como pacientes-1
        maxClusters = numPats-1
    else:
        maxClusters = math.ceil(numPats/2)  # Se probaran hasta una k = a la mitad de pacientes

    if maxClusters < minClusters:
        maxClusters = minClusters

    return maxClusters





############################################################################################################################        

if __name__ == "__main__":
    main() 