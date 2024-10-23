

required_parameters = {
    # Graph and trajectories
    'zero': False,                  
    'repository': "Test3",    # TODO
    'dateStart': "2024-03-01T00:00:00",#"2024-03-12T00:00:00", #"2024-01-13T00:00:00" #Test2,     # "2023-01-06T00:00:00",    # Para Test1
    'dateEnd': "2024-03-13T23:59:59", #"2024-01-19T23:59:59" #Test2,       # "2023-01-12T23:59:59",    # Para Test1
    'idLoc': 1306, #1339 #Test2,      # "1335",  # Para Test1   
    'idMicroorg': 1434, #1435 #Test2,     #" 1429",   # Para Test1
    'maxDaysTrajForward': 7,#7,
    'timeInFile': True, 
    # Similarity methods
    'beta': 0.45,
    'alfa': 0.5,
    'maxStepsBackwardLCSS': 5,  # DEFINICION DE BROTE   # 1 Día = 3 steps -> Dos Eventos están en el mismo día cuando la diferencia es, como máximo, 2. 
    'margin': 5,
    'similarityFunctions': ['tsJoin', 'tsJoin_2',  'dtw', 'dtw_st', 'lcss', 'lcss_2'],
    # Visualization heatmap
    'heatmap': True,
    'annotated': True,
    # Clustering
    'normalice01': True,
    'minClusters': 4,
    'maxClusters': 4,
    'maxClustersPats': True,
    'numRows': 4,
    # Folders
    'nameFolder_Matrix': './matrixes',
    'nameFolder_SimArrays': './similarityArrays',
    'nameFolder_SimArrays_0': './similarityArrays_0',
    'nameFolder_Figures': './figures',
    'nameFolder_Outputs': './outputs',
    'nameFolder_Time': './time'
}

secondMain_parameters = {
    'Ks': [3,3], #[3,3, 3,3, 3,3],
    'barColors': ['cornflowerblue', 'royalblue'] #['mediumturquoise', 'lightseagreen',  'salmon', 'tomato',  'cornflowerblue', 'royalblue']
}


matrixes = {
    'matrixDTW': [[None] * 50] * 50,
    'matrixDTW_SP': [[None] * 50] * 50,
    'matrixLCSS': [[None] * 50] * 50,
    'matrixLCSS_2': [[None] * 50] * 50
}


####################################################################################################################

def getSimilarityMethodName_Long(similarity):
    if similarity == 'dtw':
        return "Dynamic Time Warping (DTW)"
    if similarity == 'dtw_st':
        return "Spatiotemporal Dynamic Time Warping (ST-DTW)"
    if similarity == 'lcss':
        return "Spatiotemporal Longest Common Subsequence (ST-LCSS)"
    if similarity == 'lcss_2':
        return "Spatiotemporal Longest Common Subsequence With Time Window (ST-LCSS-WTW)"
    if similarity == 'tsJoin':
        return "Spatiotemporal Linear Combine (STLC)"
    if similarity == 'tsJoin_2':
        return "Joint Spatiotemporal Linear Combine (JSTLC)"
    
def getSimilarityMethodName_Short(similarity):
    if similarity == 'dtw':
        return "DTW"
    if similarity == 'dtw_st':
        return "DTW_ST"
    if similarity == 'lcss':
        return "ST_LCSS"
    if similarity == 'lcss_2':
        return "ST_LCSS_WTW"
    if similarity == 'tsJoin':
        return "STLC"
    if similarity == 'tsJoin_2':
        return "JSTLC"

def getNameValidationScore(vs):
    if vs == 'ssw':
        return "Sum of Squared Errors Within Clusters (SSW)"
    if vs == 'bh':
        return "Ball-Hall index"
    if vs == 'ssb':
        return "Sum of Squares Between Clusters (SSB)"
    if vs == 'db':
        return "Davies-Bouldin score"
    if vs == 'ch':
        return "Calinski-Harabasz score"
    if vs == 'dunn':
        return "Dunn index"
    if vs == 'xb':
        return "Xie-Beni score"
    if vs == 'hartigan':
        return "Hartigan index"
    if vs == 'xu':
        return "Xu coefficient"
    if vs == 'huberts':
        return "Modified Huberts statistic"
    if vs == 'sil':
        return "Silhouette coefficient"
    if vs == 'rmssdt':
        return "RMSSDT"
    if vs == 'sd':
        return "Standard Desviation"
    if vs == 'avgScat':
        return "Average Scatteing for Clusters"
    if vs == 'totalSep':
        return "Total Separation between Clusters"

# margin = 2  # TODO: 1, 2 o 3 ¿?     ->  2 Serían hasta 24 horas

