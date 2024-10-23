
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


def check_ValidationScores_ForRangeKClusters_allSimilarities(arrayDataframes, similarityFunctions, minClusters, maxClusters, numRows, markPoints):

    allValidationScores = {}
    j = 0
    for i in range(len(arrayDataframes)):
        df = arrayDataframes[i]
        if df is not None:
            simMethod = similarityFunctions[j]
            j += 1

            '''print(df)
            print(type(df))
            print("values")
            print(df.values)
            print("labels")
            print(len(df.index))
            print(df.index)
            print("title")
            print(simMethod)'''


            # Preparar datos para el clustering
            data, listPats = dataframeToArray(df)
            data_np = np.array(data)


            validationScores = check_ValidationScores_ForRangeKClusters(data_np, minClusters, maxClusters, numRows, markPoints, simMethod)
            allValidationScores[simMethod] = validationScores

    return allValidationScores



def check_ValidationScores_ForRangeKClusters(data, minClusters, maxClusters, numRows, markPoints, similarityMethod):
    
    if data is None:
        data, y = make_blobs(
            n_samples=20,
            n_features=4,
            centers=4,  # Para que haya 4 Blobs (Gotas - Clusters)
            cluster_std=1,
            center_box=(1.0, 20.0),
            shuffle=True,
            random_state=1,
        )

    '''print(data)
    print(type(data))
    print("Num Points: {}".format(len(data)))
    print("Num Features: {}".format(len(data[0])))
    print(data.shape)
    print(type(data[0]))'''

    # Get validation scores for each K
    validationScores = searchForNClusters_bis(data, minClusters, maxClusters)
    
    # Image
    visualizationValidationScores(validationScores, minClusters, maxClusters, numRows, markPoints, similarityMethod)

    return validationScores


# Get validation scores for each K
def searchForNClusters_bis(data, minClusters, maxClusters):

    sil_scores = []
    ch_scores = []
    db_scores = []
    ssw_scores = []
    ssb_scores = []
    bh_scores = []
    hartigan_scores = []
    xu_scores = []
    dunn_scores = []
    xb_scores = []
    huberts_scores = []
    rmssdt_scores = []
    sd_scores = []
    avgScat_scores = []
    totalSep_scores = []

    for nCluster in range(minClusters, maxClusters+1):
        
        numClusters = nCluster
        kmeans = KMeans(init="k-means++", n_clusters=numClusters)  # Por defecto, n_init=1 
        cluster_labels = kmeans.fit_predict(data)
        
        # Obtener propiedades de los clusters
        cluster_centroids = kmeans.cluster_centers_
        dictClusters = getPointsInClusters(data, cluster_labels)
        size_clusters = getNumElementsClusters_withDict(dictClusters)
        distMatrix = getDistanceMatrix(data)
        dictMeans = getMeanFromFeatures(data)
        numData = len(data)
        numFeatures = len(data[0]) 


        # Calcular indices
        silhouette = getSilhouetteScore(data, cluster_labels)
        ch = getCalinskiHarabaszScore(data, cluster_labels)
        db = getDaviesBouldinScore(data, cluster_labels)
        ssw = getSSW(data, cluster_labels, cluster_centroids)
        ssb = getSSB(data, dictClusters, size_clusters, cluster_centroids)
        bh = getBH_Index(ssw, numClusters)
        hartigan = getHartigan_Index(ssw, ssb)
        xu = getXu_coefficient(ssw, numFeatures, numData, numClusters)
        dunn = getDunn_index(dictClusters)
        xb = getXB_score(dictClusters, cluster_centroids, numClusters)
        huberts = getHuberts_statistic(data, dictClusters, cluster_centroids, distMatrix, numData)
        rmssdt = getRMSSTD(dictClusters, dictMeans, numFeatures)
        sd, avgScattering, totalSeparation = getSD_index(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData)
        
        sil_scores.append(silhouette)
        ch_scores.append(ch)
        db_scores.append(db)
        ssw_scores.append(ssw)
        ssb_scores.append(ssb)
        bh_scores.append(bh)
        hartigan_scores.append(hartigan)
        xu_scores.append(xu)
        dunn_scores.append(dunn)
        xb_scores.append(xb)
        huberts_scores.append(huberts)
        rmssdt_scores.append(rmssdt)
        sd_scores.append(sd)
        avgScat_scores.append(avgScattering)
        totalSep_scores.append(totalSeparation)

    validationScores = []
    # Compactness
    pairSSW = (getNameValidationScore('ssw'), ssw_scores)
    validationScores.append(pairSSW)
    pairBH = (getNameValidationScore('bh'), bh_scores)
    validationScores.append(pairBH)
    # Separation
    pairSSB = (getNameValidationScore('ssb'), ssb_scores)
    validationScores.append(pairSSB)
    pairDB = (getNameValidationScore('db'), db_scores)
    validationScores.append(pairDB)
    # Both
    pairCH = (getNameValidationScore('ch'), ch_scores)
    validationScores.append(pairCH)
    pairDunn = (getNameValidationScore('dunn'), dunn_scores)
    validationScores.append(pairDunn)
    pairXB = (getNameValidationScore('xb'), xb_scores)
    validationScores.append(pairXB)
    pairHartigan = (getNameValidationScore('hartigan'), hartigan_scores)
    validationScores.append(pairHartigan)
    pairXU = (getNameValidationScore('xu'), xu_scores)
    validationScores.append(pairXU)
    pairHuberts = (getNameValidationScore('huberts'), huberts_scores)
    validationScores.append(pairHuberts)
    pairSilhouette = (getNameValidationScore('sil'), sil_scores)
    validationScores.append(pairSilhouette)
    pairRMSSDT = (getNameValidationScore('rmssdt'), rmssdt_scores)
    validationScores.append(pairRMSSDT)
    pairSD = (getNameValidationScore('sd'), sd_scores)
    validationScores.append(pairSD)
    pairAvgScat = (getNameValidationScore('avgScat'), avgScat_scores)
    validationScores.append(pairAvgScat)
    pairTotalSep = (getNameValidationScore('totalSep'), totalSep_scores)
    validationScores.append(pairTotalSep)


    return validationScores 


    
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
    
    # Visualization
    visualizationOfClusterWithReducedData(data, numClusters, secondMain_parameters['h'], similarityMethod)



    # Obtener propiedades de los clusters
    cluster_centroids = kmeans.cluster_centers_
    dictClusters = getPointsInClusters(data, cluster_labels)
    size_clusters = getNumElementsClusters_withDict(dictClusters)
    distMatrix = getDistanceMatrix(data)
    dictMeans = getMeanFromFeatures(data)
    numData = len(data)
    numFeatures = len(data[0]) 


    # Calcular indices
    silhouette = getSilhouetteScore(data, cluster_labels)
    ch = getCalinskiHarabaszScore(data, cluster_labels)
    db = getDaviesBouldinScore(data, cluster_labels)
    ssw = getSSW(data, cluster_labels, cluster_centroids)
    ssb = getSSB(data, dictClusters, size_clusters, cluster_centroids)
    bh = getBH_Index(ssw, numClusters)
    hartigan = getHartigan_Index(ssw, ssb)
    xu = getXu_coefficient(ssw, numFeatures, numData, numClusters)
    dunn = getDunn_index(dictClusters)
    xb = getXB_score(dictClusters, cluster_centroids, numClusters)
    huberts = getHuberts_statistic(data, dictClusters, cluster_centroids, distMatrix, numData)
    rmssdt = getRMSSTD(dictClusters, dictMeans, numFeatures)
    sd, avgScattering, totalSeparation = getSD_index(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData)


    return dictClusters_pairs, cluster_centroids, silhouette, ch, db, ssw, ssb, bh, hartigan, xu, dunn, xb, huberts, rmssdt, sd, avgScattering, totalSeparation


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


###################################################################################################################################################################




'''
# Get validation scores for each K
def searchForNClusters(data, minClusters, maxClusters, numIters):   # La version de bis hace lo mismo porque la 'seed' no influye en el resultado

    sil_scores_avg = []
    ch_scores_avg = []
    db_scores_avg = []
    ssw_scores_avg = []
    ssb_scores_avg = []
    bh_scores_avg = []
    hartigan_scores_avg = []
    xu_scores_avg = []
    dunn_scores_avg = []
    xb_scores_avg = []
    huberts_scores_avg = []
    rmssdt_scores_avg = []
    sd_scores_avg = []
    avgScat_scores_avg = []
    totalSep_scores_avg = []

    for nCluster in range(minClusters, maxClusters+1):
        
        seeds = []    
        
        sil_scores = []
        ch_scores = []
        db_scores = []
        ssw_scores = []
        ssb_scores = []
        bh_scores = []
        hartigan_scores = []
        xu_scores = []
        dunn_scores = []
        xb_scores = []
        huberts_scores = []
        rmssdt_scores = []
        sd_scores = []
        avgScat_scores = []
        totalSep_scores = []


        for i in range(numIters):       # Todas las seeds dan el mismo resultado, así que no haría falta este bucle for     ->  Se podría dejar "random_state" por defecto y darle un valor >1 a "n_init"   # The final results is the best output of n_init consecutive runs in terms of inertia
            seed = getSeed(i, seeds)
            numClusters = nCluster
            kmeans = KMeans(init="k-means++", n_clusters=numClusters, random_state=seed, n_init=10)  # Por defecto, n_init=1 
            cluster_labels = kmeans.fit_predict(data)
            print(cluster_labels)

            # Obtener propiedades de los clusters
            cluster_centroids = kmeans.cluster_centers_
            print("\t Centroids:")
            print(cluster_centroids)
            dictClusters = getPointsInClusters(data, cluster_labels)
            size_clusters = getNumElementsClusters_withDict(dictClusters)
            distMatrix = getDistanceMatrix(data)
            dictMeans = getMeanFromFeatures(data)
            numData = len(data)
            numFeatures = len(data[0]) 


            # Calcular indices
            silhouette = getSilhouetteScore(data, cluster_labels)
            ch_score = getCalinskiHarabaszScore(data, cluster_labels)
            db_score = getDaviesBouldinScore(data, cluster_labels)
            ssw = getSSW(data, cluster_labels, cluster_centroids)
            ssb = getSSB(data, dictClusters, size_clusters, cluster_centroids)
            bh_index = getBH_Index(ssw, numClusters)
            hartigan_index = getHartigan_Index(ssw, ssb)
            xu_index = getXu_coefficient(ssw, numFeatures, numData, numClusters)
            dunn_index = getDunn_index(dictClusters)
            xb_score = getXB_score(dictClusters, cluster_centroids, numClusters)
            huberts_statistic = getHuberts_statistic(data, dictClusters, cluster_centroids, distMatrix, numData)
            rmssdt = getRMSSTD(dictClusters, dictMeans, numFeatures)
            sd_index, avgScattering, totalSeparation = getSD_index(dictClusters, data, cluster_centroids, numClusters, numFeatures, numData)
            
            sil_scores.append(silhouette)
            ch_scores.append(ch_score)
            db_scores.append(db_score)
            ssw_scores.append(ssw)
            ssb_scores.append(ssb)
            bh_scores.append(bh_index)
            hartigan_scores.append(hartigan_index)
            xu_scores.append(xu_index)
            dunn_scores.append(dunn_index)
            xb_scores.append(xb_score)
            huberts_scores.append(huberts_statistic)
            rmssdt_scores.append(rmssdt)
            sd_scores.append(sd_index)
            avgScat_scores.append(avgScattering)
            totalSep_scores.append(totalSeparation)
            
            print("\tsilhouette: {}".format(silhouette))
            print("\tch_score: {}".format(ch_score))
            print("\tdb_score: {}".format(db_score))
            print("\tssw: {}".format(ssw))
            print("\tssb: {}".format(ssb))
            print("\tbh_index: {}".format(bh_index))
            print("\thartigan_index: {}".format(hartigan_index))
            print("\txu_index: {}".format(xu_index))
            print("\tdunn_index: {}".format(dunn_index))
            print("\txb_score: {}".format(xb_score))
            print("\thuberts_statistic: {}".format(huberts_statistic))
            print("\trmssdt: {}".format(rmssdt))
            print("\tsd_index: {}".format(sd_index))
            print("\tavgScatering: {}".format(avgScattering))
            print("\totalSeparation: {}".format(totalSeparation))
            print()

        
        silhouette_avg = sum(sil_scores) / len(sil_scores)
        ch_avg = sum(ch_scores) / len(ch_scores)
        db_avg = sum(db_scores) / len(db_scores)
        ssw_avg = sum(ssw_scores) / len(ssw_scores)
        ssb_avg = sum(ssb_scores) / len(ssb_scores)
        bh_avg = sum(bh_scores) / len(bh_scores)
        hartigan_avg = sum(hartigan_scores) / len(hartigan_scores)
        xu_avg = sum(xu_scores) / len(xu_scores)
        dunn_avg = sum(dunn_scores) / len(dunn_scores)
        xb_avg = sum(xb_scores) / len(xb_scores)
        huberts_avg = sum(huberts_scores) / len(huberts_scores)
        rmssdt_avg = sum(rmssdt_scores) / len(rmssdt_scores)
        sd_avg = sum(sd_scores) / len(sd_scores)
        avgScat_avg = sum(avgScat_scores) / len(avgScat_scores)
        totalSep_avg = sum(totalSep_scores) / len(totalSep_scores)
        print("* Final: \t Silhouette: {:.6f}, \t- Calinski: {:.6f}, \t- Davies: {:.6f}".format(silhouette_avg, ch_avg, db_avg))
    
        sil_scores_avg.append(silhouette_avg)
        ch_scores_avg.append(ch_avg)
        db_scores_avg.append(db_avg)
        ssw_scores_avg.append(ssw_avg)
        ssb_scores_avg.append(ssb_avg)
        bh_scores_avg.append(bh_avg)
        hartigan_scores_avg.append(hartigan_avg)
        xu_scores_avg.append(xu_avg)
        dunn_scores_avg.append(dunn_avg)
        xb_scores_avg.append(xb_avg)
        huberts_scores_avg.append(huberts_avg)
        rmssdt_scores_avg.append(rmssdt_avg)
        sd_scores_avg.append(sd_avg)
        avgScat_scores_avg.append(avgScat_avg)
        totalSep_scores_avg.append(totalSep_avg)

    validationScores = []
    # Compactness
    pairSSW = ("Sum of Squared Errors Within Clusters (SSW)", ssw_scores_avg)
    validationScores.append(pairSSW)
    pairBH = ("Ball-Hall index", bh_scores_avg)
    validationScores.append(pairBH)
    # Separation
    pairSSB = ("Sum of Squares Between Clusters (SSB)", ssb_scores_avg)
    validationScores.append(pairSSB)
    pairDB = ("Davies-Bouldin score", db_scores_avg)
    validationScores.append(pairDB)
    # Both
    pairCH = ("Calinski-Harabasz score", ch_scores_avg)
    validationScores.append(pairCH)
    pairDunn = ("Dunn index", dunn_scores_avg)
    validationScores.append(pairDunn)
    pairXB = ("Xie-Beni score", xb_scores_avg)
    validationScores.append(pairXB)
    pairHartigan = ("Hartigan index", hartigan_scores_avg)
    validationScores.append(pairHartigan)
    pairXU = ("Xu coefficient", xu_scores_avg)
    validationScores.append(pairXU)
    pairHuberts = ("Modified Huberts statistic", huberts_scores_avg)
    validationScores.append(pairHuberts)
    pairSilhouette = ("Silhouette coefficient", sil_scores_avg)
    validationScores.append(pairSilhouette)
    pairRMSSDT = ("RMSSDT", rmssdt_scores_avg)
    validationScores.append(pairRMSSDT)
    pairSD = ("Standard Desviation", sd_scores_avg)
    validationScores.append(pairSD)
    pairAvgScat = ("Average Scatteing for Clusters", avgScat_avg)
    validationScores.append(pairAvgScat)
    pairTotalSep = ("Total Separation between Clusters", totalSep_avg)
    validationScores.append(pairTotalSep)

    return validationScores 
'''

###################################################################################################################################################################



if __name__ == "__main__":
    
    
    minClusters = 2
    maxClusters = 6
    numIters = 100#25 #100
    numRows = 4     # Numero de filas de la imagen con las distancias
    markPoints = None
    markPoints = []
    for i in range(15):
        point = [4, 0.2]
        markPoints.append(point)
    check_ValidationScores_ForRangeKClusters(None, minClusters, maxClusters, numRows, markPoints, None)
    
    '''
    data, y = make_blobs(
        n_samples=20,
        n_features=4,
        centers=4,  # Para que haya 4 Blobs (Gotas - Clusters)
        cluster_std=1,
        center_box=(1.0, 20.0),
        shuffle=True,
        random_state=1,
    )
    cluster_labels, dictClusters, cluster_centroids  = clusteringKMeans(data, 4)
    print(dictClusters)
    print(len(dictClusters))
    for label, values in dictClusters.items():
        print(label)
        print(values)'''

    #h = 0.05
    #numClusters = 4
    #visualizationOfClusterWithReducedData(data, numClusters, h)






