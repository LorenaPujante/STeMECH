
##############
# PARAMETERS #
##############

required_parameters = {
    # Graph and trajectories
    'zero': False,                  
    'repository': "Test3",    # TODO
    'dateStart': "2024-03-01T00:00:00",#"2024-03-12T00:00:00", #"2024-01-13T00:00:00" #Test2,     # "2023-01-06T00:00:00",    # Para Test1
    'dateEnd': "2024-03-13T23:59:59", #"2024-01-19T23:59:59" #Test2,       # "2023-01-12T23:59:59",    # Para Test1
    'idLoc': 1306, #1339 #Test2,      # "1335",  # Para Test1   
    'idMicroorg': 1434, #1435 #Test2,     #" 1429",   # Para Test1
    'maxDaysTrajForward': 7,#7, 
    # Similarity methods
    'similarityFunctions': ['tsJoin', 'tsJoin_2',  'dtw', 'dtw_st', 'lcss', 'lcss_2'],
    'beta': 0.45,
    'alfa': 0.5,
    'maxStepsBackwardLCSS': 5,  # DEFINICION DE BROTE   # 1 Día = 3 steps -> Dos Eventos están en el mismo día cuando la diferencia es, como máximo, 2. 
    'margin': 5,
    # Folders
    'nameFolder_Matrix': './matrixes',
    'nameFolder_SimArrays': './similarityArrays',
    'nameFolder_Figures': './figures',
    'nameFolder_Outputs': './outputs',
    'timeInFile': True,
    'nameFolder_Time': './time',
    # Others
    'normalice01': True
}

required_parameters_heatmap = {
    'annotated': True
}

required_parameters_clustering = {
    'maxClustersPats': True,
    'numRows': 4,
    'meshSize': 0.02,
    'Ks': [3,3, 3,3, 3,3],
    'barColors':  ['mediumturquoise', 'lightseagreen',  'salmon', 'tomato',  'cornflowerblue', 'royalblue'] # ['royalblue']
}


####################################################################################################################


matrixes_opt = {
    'matrixDTW': [[None for x in range(100)] for y in range(100)],
    'matrixDTW_SP': [[None for x in range(100)] for y in range(100)],
    'matrixLCSS': [[None for x in range(100)] for y in range(100)],
    'matrixLCSS_2': [[None for x in range(100)] for y in range(100)]
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

