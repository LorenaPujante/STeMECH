
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from sklearn.datasets import make_blobs

from config import *
from validationScores import *
from plot import *



########
# MAIN #
########


def check_ValidationScores_ForRangeKClusters_allSimilarities(arrayDataframes, similarityFunctions, minClusters, maxClusters, numRows):

    allPairsSilhouette = {}
    j = 0
    for i in range(len(arrayDataframes)):
        df = arrayDataframes[i]
        if df is not None:
            simMethod = similarityFunctions[j]
            j += 1

            # Preparar datos para el clustering
            data, listPats = dataframeToArray(df)
            data_np = np.array(data)

            pairSilhouette = check_ValidationScores_ForRangeKClusters(data_np, minClusters, maxClusters, numRows, simMethod)
            allPairsSilhouette[simMethod] = pairSilhouette

    print(allPairsSilhouette)
    return allPairsSilhouette



def check_ValidationScores_ForRangeKClusters(data, minClusters, maxClusters, numRows, similarityMethod):
    
    # Get validation scores for each K
    pairSilhouette = searchForNClusters_bis(data,  minClusters, maxClusters)
    
    # Image
    visualizationValidationScores_sil(pairSilhouette, minClusters, maxClusters, similarityMethod)

    return pairSilhouette


# Get validation scores for each K
def searchForNClusters_bis(data, minClusters, maxClusters):

    sil_scores = []
    
    for nCluster in range(minClusters, maxClusters+1):
        
        numClusters = nCluster
        kmeans = KMeans(init="k-means++", n_clusters=numClusters)  # Por defecto, n_init=1 
        cluster_labels = kmeans.fit_predict(data)
        
        # Calcular silhouette
        silhouette = getSilhouetteScore(data, cluster_labels)
        sil_scores.append(silhouette)
        

    pairSilhouette = (getNameValidationScore('sil'), sil_scores)
    
    return pairSilhouette 


    
#----------------------#
# FUNCIONES AUXILIARES #
#----------------------#

def dataframeToArray(df):
    listPats = list(df.index)
    data = []
    for pat1 in listPats:
        dataPat1 = []
        for pat2 in listPats:
            dataPat1.append(df[pat1][pat2])
        data.append(dataPat1)
    
    return data, listPats

def getSeed(n, seeds):
    maxInd = 31     # Random state range: [0, 2**32 - 1]

    ind_seed = int(n/3)
    seed = pow(2, ind_seed)
    
    if seed in seeds  or  ind_seed>maxInd:
        seed = seeds[-1]+1 
    seeds.append(seed)
    
    return seed


def getDistanceMatrix(data):
    eu_data = euclidean_distances(data)
    return eu_data


def getPointsInClusters(data, cluster_labels):
    dictClusters = {}
    for i in range(len(data)):
        cluster_label = cluster_labels[i]
        if cluster_label in dictClusters:
            dictClusters[cluster_label].append(data[i])
        else:
            dictClusters[cluster_label] = []
            dictClusters[cluster_label].append(data[i])

    return dictClusters


def getNumElementsClusters(cluster_labels, n_clusters):
    size_clusters = []
    cluster_labels_list = cluster_labels.tolist()
    for i in range(n_clusters):
        size_clusters.append(cluster_labels_list.count(i))

    return size_clusters

def getNumElementsClusters_withDict(dictClusters):
    size_clusters = [None] * len(dictClusters)
    for label, cluster in dictClusters.items():
        size_clusters[label] = len(cluster)

    return size_clusters


def getMeanFromFeatures(data):
    dictFeatures = {}
    for row in data:
        for i in range(len(row)):
            if i not in dictFeatures:
                dictFeatures[i] = []
            dictFeatures[i].append(row[i])

    dictMeans = {}
    for feature, values in dictFeatures.items():
        sumValues = sum(values)
        dictMeans[feature] = sumValues/len(values)

    return dictMeans


###################################################################################################################################################################


##########
# MAIN 2 #
##########

# Funcion para que le pases los datos y la k y te devuelva cada cluster con sus elementos
# También hace el plot de los "puntos reducidos" del cluster 
def clusteringKMeans(data, dictPatSims, numClusters, similarityMethod):
    kmeans = KMeans(init="k-means++", n_clusters=numClusters)  # Por defecto, n_init=1 
    cluster_labels = kmeans.fit_predict(data)
    cluster_centroids = kmeans.cluster_centers_
    dictClusters = getPointsInClusters(data, cluster_labels)

    dictClusters_pairs = getDictClusters_MatchingPatients(dictPatSims, dictClusters)     # En este diccionario cada punto del cluster viene acompañado de la etiqueta de su paciente en forma pair(pat, array_de_similitudes)
    #print(dictClusters_pairs)

    # Visualization
    visualizationOfClusterWithReducedData(data, cluster_centroids, numClusters, dictClusters_pairs, similarityMethod)

    # Obtener propiedades de los clusters
    cluster_centroids = kmeans.cluster_centers_
    dictClusters = getPointsInClusters(data, cluster_labels)
    
    # Calcular silhouette
    silhouette = getSilhouetteScore(data, cluster_labels)
    

    return dictClusters_pairs, cluster_centroids, silhouette


#----------------------#
# FUNCIONES AUXILIARES #
#----------------------#

def getDictClusters_MatchingPatients(dictPatSims, dictClusters):
    dictClusters_pairs = {}     # En este diccionario cada punto del cluster viene acompañado de la etiqueta de su paciente en forma pair(pat, array_de_similitudes)
    listPats = list(dictPatSims.keys())
    listPatSims = list(dictPatSims.values())
    for label, points in dictClusters.items():
        arrayPairs = []
        for p in points:
            i = 0
            found = False
            while i<len(listPats) and not found:
                sims = listPatSims[i]
                j = 0
                iguales = True
                while j<len(p) and iguales:
                    sim_round1 = round(p[j], 8)
                    sim_round2 = round(sims[j], 8)
                    if sim_round1 != sim_round2:
                        iguales = False
                    j += 1
                if iguales:
                    found = True
                else:
                    i += 1
            if found:
                pair = (listPats[i], listPatSims[i])
                arrayPairs.append(pair)
        dictClusters_pairs[label] = arrayPairs

    return dictClusters_pairs


