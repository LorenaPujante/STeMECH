
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import colormaps
from sklearn.decomposition import PCA

from config import *


####################################
# VALIDATION SCORES FOR TESTING Ks #
####################################

def visualizationValidationScores(validationScores, minClusters, maxClusters, numRows, markPoints, similarityMethod):

    numValScores = len(validationScores)
    plotsPerRow = math.ceil(numValScores/numRows)
    
    # Get number of k's
    ks = []
    for i in range(minClusters, maxClusters+1):
        ks.append(i)
    
    fig, axes = plt.subplots(plotsPerRow, numRows, figsize=(20, 10))
    for i in range(len(validationScores)):
        valScore = validationScores[i]
        a = int(i/numRows)
        b = i%numRows
        ax = axes[a][b]
        
        if markPoints is not None:
            markPoint = markPoints[i]    # Por si se quiere resaltar algún punto
        else:
            markPoint = None
        
        title = getSimilarityMethodName_Long(similarityMethod)
        plotterValidationScores(ax, axes, fig, ks, valScore[1], valScore[0], markPoint, title)
    

    # Eliminar subplots en blanco
    removeLastSubplots(numValScores, numRows, plotsPerRow, fig, axes)

    nameFile = "./figure_ValidationScores"
    if similarityMethod is not None:
        nameFile += "_{}".format(getSimilarityMethodName_Short(similarityMethod))
    nameFile += ".png"
    plt.savefig(required_parameters['nameFolder_Figures'] + nameFile)
    plt.show()
    #plt.clf()   # Borra la figura
    
    return 0


def visualizationValidationScores_sil(pairSilhouette, minClusters, maxClusters, similarityMethod):

    # Get number of k's
    ks = []
    for i in range(minClusters, maxClusters+1):
        ks.append(i)
    
    plt.figure(figsize=(8,5))
    title = getSimilarityMethodName_Long(similarityMethod)
    dataX = ks
    dataY = pairSilhouette[1]

    plt.plot(dataX, dataY,
                  color='darkcyan',
                  linewidth=1.5, 
                  marker='.', markersize=10)
    
    plt.title(title, fontweight='bold', fontsize=11)
    
    plt.grid(True)

    # Para que en todos los subplots la leyenda del eje X (abajo) sean las k's
    plt.xlabel('k', fontsize=10, style= "italic", color='dimgrey', loc='right')
    plt.xticks(dataX, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Margenes
    plt.subplots_adjust(left=0.1,
                    right=0.95,
                    bottom=0.1, 
                    top=0.82,
                    hspace=0.6)  

    textTitle = "Validation scores"
    if similarityMethod is not None:
        textTitle += " - {}".format(similarityMethod)
    plt.suptitle(textTitle, fontsize=20, bbox={'facecolor': 'aliceblue', 'pad':5})
    

    nameFile = "./figure_SilhouetteScores"
    if similarityMethod is not None:
        nameFile += "_{}".format(getSimilarityMethodName_Short(similarityMethod))
    nameFile += ".png"
    plt.savefig(required_parameters['nameFolder_Figures'] + nameFile)
    plt.show()
    


#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def plotterValidationScores(ax, axes, fig, dataX, dataY, title, markedPoint, similarityMethod):

    out = ax.plot(dataX, dataY,
                  color='darkcyan',
                  linewidth=1.5, 
                  marker='.', markersize=10)#, **param_dict)
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    
    ax.grid(True)

    # Para que en todos los subplots la leyenda del eje X (abajo) sean las k's
    if type(dataX[0]) is int:
        ax.set_xlabel('k', fontsize=10, style= "italic", color='dimgrey', loc='right')
        plt.setp(axes, xticks=dataX, xticklabels=dataX)
    else:
        ax.set_xlabel('Similarity method', fontsize=10, style= "italic", color='dimgrey', loc='right')

    plt.yticks(fontsize=11)
    plt.xticks(fontsize=11)

    # Linea vertical
    if markedPoint is not None:
        ax.axvline(x=markedPoint[0], color='orangered', linewidth=1, linestyle='--')
        ax.scatter(markedPoint[0], markedPoint[1], marker='x', color='red')

    # Margenes
    plt.subplots_adjust(left=0.05,
                    right=0.95,
                    bottom=0.1, 
                    top=0.9,
                    hspace=0.6) #, 0.4
                    #wspace=0.4) 

    textTitle = "Validation scores"
    if similarityMethod is not None:
        textTitle += " - {}".format(similarityMethod)
    fig.suptitle(textTitle, fontsize=20, bbox={'facecolor': 'aliceblue', 'pad':8})

    return out


def removeLastSubplots(numTrueSubplots, numRows, plotsPerRow, fig, axes):
    numSubplots = plotsPerRow*numRows
    restSubplots = numTrueSubplots%numRows
    if numSubplots > numTrueSubplots:
        diff = numSubplots-numTrueSubplots
        for i in range(diff):
            fig.delaxes(axes[numRows-1][restSubplots+i])
        

###################################################################################################################################################################


##############################################
# VALIDATION SCORES FOR COMPARING SIMILARITY #
##############################################

def visualizationValidationScores_bis(validationScores, numRows):
    numValScores = len(validationScores)
    plotsPerRow = math.ceil(numValScores/numRows)


    fig, axes = plt.subplots(plotsPerRow, numRows, figsize=(20, 10))
    for i in range(numValScores):
        valScore = validationScores[i]
        a = int(i/numRows)
        b = i%numRows
        ax = axes[a][b]

        simMethods = []
        for sm in required_parameters['similarityFunctions']:
            simMethods.append(getSimilarityMethodName_Short(sm))

        barLabels = []
        for value in valScore[1]:
            value_round = round(value, 2)
            barLabels.append(value_round)

        plotterValidationScores_bis(ax, fig, simMethods, valScore[1], valScore[0], barLabels)
        
    # Eliminar subplots en blanco
    removeLastSubplots(numValScores, numRows, plotsPerRow, fig, axes)

    nameFile = required_parameters['nameFolder_Figures'] + "./figure_SimMethods_ValidationScores.png"
    plt.savefig(nameFile)
    plt.show()


def visualizationValidationScores_bis_sil(pairSilhouette, numRows):
    simMethods = []
    for sm in required_parameters['similarityFunctions']:
        simMethods.append(getSimilarityMethodName_Short(sm))
    
    barLabels = []
    for value in pairSilhouette[1]:
        value_round = round(value, 2)
        barLabels.append(value_round)

    dataX = simMethods 
    dataY = barLabels

    plt.figure(figsize=(8,5))
    out = plt.bar(dataX, dataY,
                    color=required_parameters_clustering['barColors'],
                    width=0.8,
                    label=barLabels)
        
    plt.bar_label(out, labels=barLabels, label_type='edge', padding=1)
        
    # Para aumentar un poco el eje Y
    plt.ylim(-1.1, 1.1)
        
    plt.tight_layout()

    plt.title(pairSilhouette[0], fontweight='bold', fontsize=11)

    plt.grid(axis='y')
    plt.xlabel('Similarity method', fontsize=10, style= "italic", color='dimgrey', loc='right')

    plt.yticks(fontsize=11)
    plt.xticks(fontsize=11)

    # Margenes
    plt.subplots_adjust(left=0.1,
                    right=0.95,
                    bottom=0.1, 
                    top=0.9,
                    hspace=0.6) 

    nameFile = required_parameters['nameFolder_Figures'] + "./figure_SimMethods_SilhouetteScores.png"
    plt.savefig(nameFile)
    plt.show()



#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def plotterValidationScores_bis(ax, fig, dataX, dataY, title, barLabels):

    out = ax.bar(dataX, dataY,
                  color=required_parameters_clustering['barColors'],
                  width=0.8,
                  label=barLabels)
    
    ax.bar_label(out, labels=barLabels, label_type='edge', padding=1)
    
    # Para aumentar un poco el eje Y
    if title == getNameValidationScore('sil'):
        ax.set_ylim(-1.1, 1.1)
    else:
        maxDataY = max(dataY)
        if maxDataY>0:
            if maxDataY < 1:
                maxYLim = round(maxDataY+0.2, 1)
            else:
                maxYLim = math.ceil(maxDataY)
            ax.set_ylim(top=maxYLim)

        minDataY = min(dataY)
        if minDataY<0:
            if minDataY > -1:
                minYLim = round(minDataY-0.2, 1)    # Antes habia un minYLim
            else:
                minYLim = math.floor(minDataY)  
            ax.set_ylim(bottom=minYLim)
    fig.tight_layout()

    ax.set_title(title, fontweight='bold', fontsize=11)

    ax.grid(axis='y')
    ax.set_xlabel('Similarity method', fontsize=10, style= "italic", color='dimgrey', loc='right')

    plt.yticks(fontsize=11)
    plt.xticks(fontsize=11)

    # Margenes
    plt.subplots_adjust(left=0.05,
                    right=0.95,
                    bottom=0.1, 
                    top=0.9,
                    hspace=0.6) #, 0.4
                    #wspace=0.4) 

    textTitle = "Validation scores"
    fig.suptitle(textTitle, fontsize=20, bbox={'facecolor': 'aliceblue', 'pad':8})

    return out


###################################################################################################################################################################


#############################
# VISUALIZATION OF CHOSEN K #
#############################

def linkPats_Clusters_Data_RedData(data, reduced_data, dictClusters_pairs):
    pat_cluster_data_rData = []
    for i in range(len(data)):
        rd = reduced_data[i]
        d = data[i]
        pat = None
        clustersKeys = list(dictClusters_pairs.keys())
        found1 = False
        j = 0
        while j<len(clustersKeys) and not found1:
            key = clustersKeys[j]
            cluster = dictClusters_pairs[key]
            found2 = False
            k = 0
            while k<len(cluster) and not found2:
                patDists = cluster[k][1]
                if (patDists == d).all():
                    found2 = True
                    pat = cluster[k][0]
                else:
                    k += 1
            if found2:
                found1 = True
            else:
                j += 1
        dato = [pat, key, d.tolist(), rd.tolist()]
        pat_cluster_data_rData.append(dato)

    return pat_cluster_data_rData

# https://scikit-learn.org/1.5/auto_examples/cluster/plot_digits_linkage.html
def visualizationOfClusterWithReducedData(data, cluster_centroids, numClusters, dictClusters_pairs, similarityMethod):
    
    # Reduce data to 2 dimensions
    data_and_centroids = np.concatenate((data, cluster_centroids))
    pca = PCA(n_components=2)
    pca = pca.fit(data)
    data_and_centroids_red = pca.transform(data_and_centroids)
    reduced_data = data_and_centroids_red[:-numClusters]
    cluster_centroids_red = data_and_centroids_red[len(data_and_centroids_red)-numClusters:]

    # Link each Patient with its cluster and data
    pat_cluster_data_rData = linkPats_Clusters_Data_RedData(data, reduced_data, dictClusters_pairs)

    # Normalize reduced data to [0,1]
    x_min, x_max = data_and_centroids_red[:, 0].min(), data_and_centroids_red[:, 0].max()
    reduced_data_norm = (reduced_data - x_min) / (x_max - x_min)
    cluster_centroids_red_norm = (cluster_centroids_red - x_min) / (x_max - x_min)
    pat_cluster_data_rDataNorm = []
    for i in range(len(pat_cluster_data_rData)):
        dato = pat_cluster_data_rData[i]
        datoNorm = [dato[0],dato[1],dato[2],reduced_data_norm[i]]
        pat_cluster_data_rDataNorm.append(datoNorm)

    # Figure size
    plt.figure(figsize=(9,8))

    # Plot centroids of the clusters and the legend
    cmap = colormaps[required_parameters_clustering['reducedColors']]
    norm = plt.Normalize(vmin=0, vmax=6)
    for i in range(len(cluster_centroids_red_norm)):
        cluster = i
        centroid = cluster_centroids_red_norm[i]
        plt.scatter(
            centroid[0], centroid[1],
            marker="x",
            s=75,
            c = cmap(norm(cluster)),
            #edgecolors='black',
            alpha=0.75
        )
    labels = np.arange(numClusters).tolist()
    for i in range(len(labels)):
        labels[i] = "Cluster {}".format(labels[i])
    plt.legend(labels, bbox_to_anchor=(1.04, 1), loc='upper left')

    # Plot each Patient
    for i in range(len(pat_cluster_data_rDataNorm)):
        cluster = pat_cluster_data_rDataNorm[i][1]
        pat = pat_cluster_data_rDataNorm[i][0]
        rd = pat_cluster_data_rDataNorm[i][3]
        plt.scatter(
            rd[0], rd[1],
            marker="o", #marker=f"·${cluster}$",
            s=100,
            c = cmap(norm(cluster)),
            edgecolors='black'
        )
        plt.text(
            rd[0]+0.01, rd[1], " {}".format(pat)
        )
           
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(left=0.05, right=0.8)

    plt.title("{}: Clusters with K = {}\n".format(getSimilarityMethodName_Short(similarityMethod), numClusters), fontsize=18)

    plt.savefig(required_parameters['nameFolder_Figures'] + './figure_ClusterReducedData_k{}_{}.png'.format(numClusters, getSimilarityMethodName_Short(similarityMethod)))
    plt.show()



###################################################################################################################################################################



###########
# HEATMAP #
###########

def visualizationOfHeatMapBtwPatients(simMethod, data, patients, annotated):
    fig = plt.figure(figsize=(12, 11))

    title = getSimilarityMethodName_Long(simMethod)

    im, cbar = plotterHeatmap(data, patients, patients, simMethod, cbarlabel="Similarity (normalized to [0,1])\n")
    if annotated:
        texts = plotter_AnnotateHeatmap(im)

    plt.savefig(required_parameters['nameFolder_Figures'] + './figure_similarityHeatmap_{}.png'.format(getSimilarityMethodName_Short(simMethod)))
    plt.show()


#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def plotterHeatmap_orig(data, row_labels, col_labels, simMethod, ax=None, cbarlabel=""):

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    #cmap = mpl.cm.ScalarMappable(cmap=required_parameters_heatmap['heatColors'])
    im = ax.imshow(data, cmap=required_parameters_heatmap['heatColors'])   # cmap: Esquema de color (https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=np.arange(0, 1, 0.1))
    #cmap.set_clim(0, 1.0)
    #mesh = ax.pcolormesh(data, cmap = required_parameters_heatmap['heatColors'])
    #cbar.ax.set_xlim(0,1.0) #mesh.set_clim(0,1.0)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12, style= "italic", color='dimgrey')    # cbarlabel: The label for the colorbar

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Set the alignment of tick labels
    plt.setp(ax.get_xticklabels(), ha="center")

    plt.title("({}) Heatmap of similarities between patients' trajectories\n".format(getSimilarityMethodName_Short(simMethod)), fontsize=18)

    plt.subplots_adjust(left=0.1,
                    right=0.95,
                    bottom=0.1, 
                    top=0.95,
                    )

    return im, cbar


def plotterHeatmap(data, row_labels, col_labels, simMethod, ax=None, cbarlabel=""):

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap=required_parameters_heatmap['heatColors'], vmin=0, vmax=1)   # cmap: Esquema de color (https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=np.arange(0, 1.1, 0.1))
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12, style= "italic", color='dimgrey')    # cbarlabel: The label for the colorbar

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Set the alignment of tick labels
    plt.setp(ax.get_xticklabels(), ha="center")

    plt.title("({}) Heatmap of similarities between patients' trajectories\n".format(getSimilarityMethodName_Short(simMethod)), fontsize=18)

    plt.subplots_adjust(left=0.1,
                    right=0.95,
                    bottom=0.1, 
                    top=0.95,
                    )

    return im, cbar


def func(x, pos):
        if type(x) == str:
            return x
        return f"{x:.2f}".replace("0.", ".").replace("1.00", "")
def plotter_AnnotateHeatmap(im, textcolors=("white", "black"), threshold=None):

    data = im.get_array()
    
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/3

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              ha="center", va="center", color="w",
              size=12)

    # Text formatter
    valfmt = tck.FuncFormatter(func) #tck.StrMethodFormatter("{x:.2f}")  # Image's data is used

    # Loop over the data and create a `Text` for each cell.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if i==j:
                value = "-"
            else:
                value = data[i,j]
            text = im.axes.text(j, i, valfmt(value, None), kw)
            texts.append(text)

    return texts
