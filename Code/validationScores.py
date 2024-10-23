
import math
from collections import Counter

from scipy.spatial import distance
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


##########################
# COHESION / COMPACTNESS #
##########################

# https://medium.com/aprendizaje-no-supervisado/clustering-cee49ad0061f
# Sum of Squared Within (SSW)
    # Medida interna especialmente usada para evaluar la COHESIÓN
    # Cohesión: Cuán cerca están los puntos de un mismo cluster
def getSSW(data, cluster_labels, cluster_centroids):
    
    ssw = 0
    for i in range(len(data)):
        cluster_label = cluster_labels[i]
        cluster_centroid = cluster_centroids[cluster_label]
        euclidean_dst = distance.euclidean(data[i], cluster_centroid)
        dst = pow(euclidean_dst, 2)
        
        ssw += dst

    return ssw


# Ball & Hall index
def getBH_Index(ssw, numClusters):
    bh_index = ssw/numClusters

    return bh_index



##############
# SEPARATION #
##############

# https://medium.com/aprendizaje-no-supervisado/clustering-cee49ad0061f
# Sum of Squared Between (SSB)
    # Medida de SEPARACIÓN utilizada para evaluar la distancia interclúster
    # Separación: Cuán lejos están los puntos de clusters diferentes
def getSSB(data, dictClusters, size_clusters, cluster_centroids):
    
    # Get media del dataset
    mean_dataset = data.mean(0)
    
    # Calculo
    ssb = 0
    for label in dictClusters.keys():
        cluster_centroid = cluster_centroids[label]
        euclidean_dst = distance.euclidean(cluster_centroid, mean_dataset)
        dst = pow(euclidean_dst, 2)
        dst *= size_clusters[label]

        ssb += dst
    
    return ssb


def getDaviesBouldinScore(data, cluster_labels):
    db_score = davies_bouldin_score(data, cluster_labels)
    return db_score



########
# BOTH #
########

def getCalinskiHarabaszScore(data, cluster_labels):
    ch_score = calinski_harabasz_score(data, cluster_labels)
    return ch_score


# Dunn index
def getDunn_index(dictClusters):

    maxDistInCluster = -1
    for cluster in dictClusters.values():
        for i in range(len(cluster)):
            data1 = cluster[i]
            for j in range(i+1, len(cluster)):
                data2 = cluster[j]
                euclidian_dist = distance.euclidean(data1, data2)
                if euclidian_dist > maxDistInCluster:
                    maxDistInCluster = euclidian_dist


    minDistBtwClusters = 10000000000
    for label1, cluster1 in dictClusters.items():
        for label2, cluster2 in dictClusters.items():
            if label2 > label1:
                for data1 in cluster1:
                    for data2 in cluster2:
                        euclidian_dist = distance.euclidean(data1, data2)
                        if euclidian_dist < minDistBtwClusters:
                            minDistBtwClusters = euclidian_dist


    d_index = minDistBtwClusters / maxDistInCluster
    return d_index


# Xie-Beni score
def getXB_score(dictClusters, cluster_centroids, numClusters):
    
    compactness = 0
    u_fuzzy = 1
    for label, data in dictClusters.items():
        cluster_centroid = cluster_centroids[label]
        for d in data:
            euclidean_dst = distance.euclidean(cluster_centroid, d)
            dst = pow(euclidean_dst, 2)
            dst *= pow(u_fuzzy, 2)

            compactness += dst
    
    minDst = 10000000000
    for i in range(numClusters):
        centroid_1 = cluster_centroids[i]
        for j in range(i+1, numClusters):
            centroid_2 = cluster_centroids[j]
            euclidean_dst = distance.euclidean(centroid_1, centroid_2)
            dst = pow(euclidean_dst,2)
            if dst < minDst:
                minDst = dst
    separation = numClusters*minDst

    xb_score = compactness/separation
    return xb_score


# Hartigan index
def getHartigan_Index(ssw, ssb):
    aux = ssw/ssb
    h_index = math.log(aux, 10)
    return h_index


# Xu Coefficient
def getXu_coefficient(ssw, numFeatures, numData, numClusters):
    aux = numFeatures*pow(numData, 2)
    a = ssw/aux
    a = math.sqrt(a)
    a = math.log(a, 10)
    a *= numFeatures
    
    b = math.log(numClusters, 10)
    
    xu_index = a+b
    return xu_index


# Hurbert statistic
def getHuberts_statistic(data, dictClusters, cluster_centroids, distMatrix, numData):

    m = numData * (numData-1) / 2

    distSum = 0
    for label1 in dictClusters.keys():
        for label2 in dictClusters.keys():
            if label2 > label1:
                centroid1 = cluster_centroids[label1]
                centroid2 = cluster_centroids[label2]
                distCentrds = distance.euclidean(centroid1, centroid2)
                values1 = dictClusters[label1]
                values2 = dictClusters[label2]
                for value1 in values1:
                    i = getIndexFromValueInData(data, value1)
                    for value2 in values2:
                        j = getIndexFromValueInData(data, value2)
                        prox = distMatrix[i][j]   
                        diff = distCentrds * prox
                        

                distSum += diff

    h_statistic = distSum/m
    return h_statistic


def getSilhouetteScore(data, cluster_labels):
    silhouette = silhouette_score(data, cluster_labels)
    return silhouette


# Root-Mean-Square Standard Desviation (RMSSTD)
def getRMSSTD(dictClusters, dictMeans, numFeatures):
    
    sum1 = 0
    for cluster in dictClusters.values():
        for i in range(numFeatures):
            for point in cluster:
                valueFeature = point[i]
                meanFeature = dictMeans[i] 
                diff = pow(valueFeature-meanFeature, 2)

                sum1 += diff

    sum2 = 0
    for cluster in dictClusters.values():
        for i in range(numFeatures):
            aux = len(cluster)-1
            sum2 += aux

    rmsstd = sum1/sum2
    #rmsstd = abs(rmsstd)
    rmsstd = math.sqrt(rmsstd)
    return rmsstd


def getSD_index(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData):
    avgScattering = getAvgScatteringForClusters(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData)
    totalSeparation = getTotalSeparationBtwClusters(dictClusters, cluster_centroids)

    sd = numClusters*avgScattering + totalSeparation
    return sd, avgScattering, totalSeparation


#----------------------#
# FUNCIONES AUXILIARES #
#----------------------#

def getVarianceCluster(cluster, centroid, numFeatures):
    avgVariance = 0
    for i in range(numFeatures):
        sum = 0
        for row in cluster:
            diff = row[i]-centroid[i]
            diff = pow(diff, 2)
            sum += diff
        
        variance = sum/len(cluster)
        avgVariance += variance

    avgVariance /= numFeatures
    return variance

def getVarianceData(data, numFeatures, numData):    
    avgVariance = 0
    for i in range(numFeatures):

        meanFeature = 0
        for row in data:
            meanFeature += row[i]
        meanFeature /= numData

        sum = 0
        for row in data:
            diff = row[i]-meanFeature
            diff = pow(diff, 2)
            sum += diff
        
        variance = sum/numData
        avgVariance += variance

    avgVariance /= numFeatures
    return avgVariance

def getAvgScatteringForClusters(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData):
    
    sum = 0
    for i in range(numClusters):
        varianceCluster = getVarianceCluster(dictClusters[i], cluster_centroids[i], numFeatures)
        varianceData = getVarianceData(data, numFeatures, numData)
        div = varianceCluster/varianceData
        sum += div

    avgScattering = sum/numClusters
    return avgScattering

def getSeparationSingleLinkage(cluster_centroids):
    minSeparation = 1000000000
    for i in range(len(cluster_centroids)):
        for j in range(i+1, len(cluster_centroids)):
            diff = distance.euclidean(cluster_centroids[i], cluster_centroids[j])
            if diff < minSeparation:
                minSeparation = diff
    
    return minSeparation

def getSeparationCompleteLinkage(cluster_centroids):
    maxSeparation = -1
    for i in range(len(cluster_centroids)):
        for j in range(i+1, len(cluster_centroids)):
            diff = distance.euclidean(cluster_centroids[i], cluster_centroids[j])
            if diff > maxSeparation:
                maxSeparation = diff
    
    return maxSeparation

def getTotalSeparationBtwClusters(dictClusters, cluster_centroids):
    minSeparation = getSeparationSingleLinkage(cluster_centroids)
    maxSeparation = getSeparationCompleteLinkage(cluster_centroids)
    aux = maxSeparation/minSeparation
    
    sum = 0
    for label1 in dictClusters.keys():
        for label2 in dictClusters.keys():
            if label2>label1:
                diff = distance.euclidean(cluster_centroids[label1], cluster_centroids[label2])
                diff = pow(diff, -1)
                sum += diff

    totalSeparation = aux * sum

    return totalSeparation



def getIndexFromValueInData(data, value):
    i = 0
    found = False
    while i<len(data) and not found:
        v = data[i]
        if Counter(v) == Counter(value):
            found = True
        else:
            i += 1

    return i




'''
# R-Squared (RS)
    # DEVUELVE 0 PORQUE LAS SUM_1A Y SUM_1B SUMAN EL MISMO VALOR  ->  SEGURAMENTE TIENE QUE VER EL "number of data values", QUE AQUÍ SE ESTÁN USANDO VALORES CONTINUOS, EN LUGAR DE ENTEROS 
def getRS(dictClusters, data, dictMeans, numFeatures):
    sum1 = 0
    sum1a = 0
    sum1b = 0
    for i in range(numFeatures):
        for row in data:
            valueFeature = row[i]
            meanFeature = dictMeans[i]
            diff = pow(valueFeature-meanFeature, 2)            
            sum1a += diff
    
    for cluster in dictClusters.values():
        for i in range(numFeatures):
            for point in cluster:
                valueFeature = point[i]
                meanFeature = dictMeans[i] 
                diff = pow(valueFeature-meanFeature, 2)
                sum1b += diff
    print(sum1a)
    print(sum1b)
    sum1 = sum1a - sum1b

    sum2 = 0
    for i in range(numFeatures):
        for row in data:
            valueFeature = row[i]
            meanFeature = dictMeans[i]
            diff = pow(valueFeature-meanFeature, 2)            
            sum2 += diff

    rs = sum1 / sum2
    return rs
'''
