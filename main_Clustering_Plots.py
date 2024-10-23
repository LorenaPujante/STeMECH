
import numpy as np

from config import *
from writer_reader_similarities import *
from clustering import *
from plot import *


def main():
    
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

    for ind in range(len(required_parameters['similarityFunctions'])):
        simMethod = required_parameters['similarityFunctions'][ind]
        k = secondMain_parameters['Ks'][ind]
    
        # Read data from files
        data, pats = readCSVToSimilarityMatrix(simMethod)
        print(pats)
        print(data)
        print(type(data))
        data_np = np.array(data)
        
        # Diccionario en el que cada paciente tiene sus similitudes
        dictPatSims = {}
        for i in range(len(pats)):
            pat = pats[i]
            sims = data[i]
            dictPatSims[pat] = sims
        for label, pat in dictPatSims.items():
            print("{}: {}".format(label, pat))

        # Clustering
        dictClusters_pairs, cluster_centroids, silhouette, ch, db, ssw, ssb, bh, hartigan, xu, dunn, xb, huberts, rmssdt, sd, avgScattering, totalSeparation = clusteringKMeans(data_np, dictPatSims, k, simMethod)
        print(cluster_centroids)

        # Mostrar resultados Clustering
        writeResultsClustering_specificK(simMethod, k, dictClusters_pairs, cluster_centroids)

        # Para mostrar las Validation scores de cada Similarity method
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
    

    visualizationValidationScores_bis(validationScores, required_parameters['numRows'])






if __name__ == "__main__":
    main() 
    