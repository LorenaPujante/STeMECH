
import numpy as np

from config import *
from writer_reader_similarities import *
from clustering_sil import *
from plot import *


def main():
    
    sil_scores = []

    for ind in range(len(required_parameters['similarityFunctions'])):
        simMethod = required_parameters['similarityFunctions'][ind]
        k = required_parameters_clustering['Ks'][ind]
    
        # Read data from files
        data, pats = readCSVToSimilarityMatrix(simMethod)
        data_np = np.array(data)
        
        # Diccionario en el que cada paciente tiene sus similitudes
        dictPatSims = {}
        for i in range(len(pats)):
            pat = pats[i]
            sims = data[i]
            dictPatSims[pat] = sims
        
        # Clustering
        dictClusters_pairs, cluster_centroids, silhouette = clusteringKMeans(data_np, dictPatSims, k, simMethod)
        
        # Mostrar resultados Clustering
        writeResultsClustering_specificK(simMethod, k, dictClusters_pairs, cluster_centroids)

        # Para mostrar las Validation scores de cada Similarity method
        sil_scores.append(silhouette)
        
        print(silhouette)
        nameFolder = required_parameters['nameFolder_Outputs']
        fileName = nameFolder + '/' + "clustering_{}_K{}.txt".format(getSimilarityMethodName_Short(simMethod), k)
        file = open(fileName, "a")
        line = "\n · Silhouette: {}".format(silhouette)
        file.write(line)
        file.close()

    pairSilhouette = (getNameValidationScore('sil'), sil_scores)

    visualizationValidationScores_bis_sil(pairSilhouette, required_parameters_clustering['numRows'])






if __name__ == "__main__":
    main() 
    