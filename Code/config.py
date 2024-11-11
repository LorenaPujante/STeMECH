
##############
# PARAMETERS #
##############

required_parameters = {
    # Graph and trajectories
    'zero': False,                  
    'repository': "YourRepositoryName",    # TODO
    'dateStart': "2024-03-02T00:00:00",
    'dateEnd': "2024-03-06T23:59:59",
    'idLoc': 1340,  
    'idMicroorg': 1436, 
    'maxDaysTrajForward': 3,
    # Similarity methods
    'similarityFunctions': ['dtw', 'dtw_st', 'lcss', 'lcss_2, 'tsJoin', 'tsJoin_2'],
    'beta': 0.45,
    'alfa': 0.5,
    'maxStepsBackwardLCSS': 5,   
    'margin': 5,
    'maxSpDist': 40/94, 
    'maxDiffStepsSTLC': True,
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
    'annotated': True,
    'heatColors': "RdYlBu_r"
}

required_parameters_clustering = {
    'maxClustersPats': True,
    'numRows': 4,
    'meshSize': 0.02,
    'Ks': [3,3, 3,3, 3,3],
    'reducedColors': "gist_rainbow",
    'barColors':  ['mediumturquoise', 'lightseagreen',  'salmon', 'tomato',  'cornflowerblue', 'royalblue']
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

