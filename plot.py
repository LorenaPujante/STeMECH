
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as tck
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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


#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def plotterValidationScores_bis(ax, fig, dataX, dataY, title, barLabels):

    out = ax.bar(dataX, dataY,
                  color=secondMain_parameters['barColors'],
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
                minYLim = round(minYLim-0.2, 1)
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

def visualizationOfClusterWithReducedData(data, numClusters, h, similarityMethod):
    reduced_data = PCA(n_components=2).fit_transform(data)
    
    # ELIGE EL MINIMO Y EL MAXIMO DE CADA UNA DE LAS DIMENSIONES REDUCIDAS
    # ARRAY QUE VA DEL MIN AL MAX CON SALTOS DE TAMAÑO h
    # LUEGO HACE UNA GRID DE TAMAÑO LEN(ARRAY(DIM_X))xLEN(ARRAY(DIM_Y)) CON LOS VALORES SACADOS DE LOS ARRAYS 'ARANGE'
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1   # [:, 0] = Todos los valores de la primera columna
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    kmeans = KMeans(init="k-means++", n_clusters=numClusters, n_init=4)  # Se obtienen los mismos clusters con los mismos puntos en cada cluster, solo cambia el nombre de los clusters: es posible que los puntos del cluster_1 estén en el cluster_3 de datos reducidos
    cluster_labels_reducedData = kmeans.fit_predict(reduced_data)
    centroids_reducedData = kmeans.cluster_centers_

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    # Show image
    plotterClustersReducedData(reduced_data, centroids_reducedData, Z, xx, yy, x_min, x_max, y_min, y_max, numClusters, similarityMethod)

    plt.savefig(required_parameters['nameFolder_Figures'] + './figure_ClusterReducedData_k{}_{}.png'.format(numClusters, getSimilarityMethodName_Short(similarityMethod)))
    plt.show()
    #plt.clf()


#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def correspondenciaDataReduced(data, reduced_data):
    correspondenciaPuntos = []
    for i in range(len(data)):
        pair = (data[i], reduced_data[i])
        correspondenciaPuntos.append(pair)

    return correspondenciaPuntos


def plotterClustersReducedData(reduced_data, centroids, Z, xx, yy, x_min, x_max, y_min, y_max, numClusters, similarityMethod):
    
    plt.figure(figsize=(8, 7))

    # Para crear una imagen como un grid
    plt.imshow(     
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=6, color=(49/255, 45/255, 140/255))#(30/255, 45/255, 125/255)

    # Plot the centroids as a white X
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]
    plt.scatter(
        centroids_x,
        centroids_y,
        marker="X",
        s=150,
        linewidths=1,
        color="white",
        zorder=10,
    )
    # Label for the centroids
    for i in range(numClusters):
        nCluster = i+1
        plt.text(centroids_x[i], centroids_y[i], "   Cluster {}".format(nCluster), 
                 fontsize=12, fontweight='bold', color='0.1', 
                 zorder=20) # zorder : "Eje Z"
        

    plt.title("{}: Clusters with K = {}\n".format(getSimilarityMethodName_Short(similarityMethod), numClusters), fontsize=18)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # get rid of the frame
    for spine in plt.gca().spines.values(): # gca() : Get the current Axes
        spine.set_visible(False)
    plt.xticks(())
    plt.yticks(())
    
    plt.subplots_adjust(left=0.1,
                    right=0.9,
                    bottom=0.2, 
                    top=0.8,
                    hspace=0.6)
    

###################################################################################################################################################################


###########
# HEATMAP #
###########

def visualizationOfHeatMapBtwPatients(simMethod, data, patients, annotated):
    fig = plt.figure(figsize=(8, 7))

    title = getSimilarityMethodName_Long(simMethod)

    im, cbar = plotterHeatmap(data, patients, patients, simMethod, cbarlabel="Similarity (normalized to [0,1])\n")
    if annotated:
        texts = plotter_AnnotateHeatmap(im)

    plt.savefig(required_parameters['nameFolder_Figures'] + './figure_similarityHeatmap_{}.png'.format(getSimilarityMethodName_Short(simMethod)))
    plt.show()


#--------------------#
# AUXILIAR FUNCTIONS #
#--------------------#

def plotterHeatmap(data, row_labels, col_labels, simMethod, ax=None, cbarlabel=""):

    if ax is None:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, cmap="RdYlBu_r")   # cmap: Esquema de color (https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
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

    plt.subplots_adjust(left=0.2,
                    right=0.95,
                    bottom=0.1, 
                    top=0.85,
                    )

    return im, cbar


def plotter_AnnotateHeatmap(im):

    data = im.get_array()
    
    # Text formatter
    valfmt = tck.StrMethodFormatter("{x:.2f}")  # Image's data is used

    # Loop over the data and create a `Text` for each cell.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, valfmt(data[i, j], None), ha="center", va="center", color="w")
            texts.append(text)

    return texts