
from config import *
from writer_reader_similarities import *
from clustering import *
from plot import *


def main():

    # READ DATA
    arrayDataframes = []
    for simMethod in required_parameters['similarityFunctions']:
        simMethod = 'tsJoin'

        df = readCSVToSimilarityMatrix_df(simMethod)
        arrayDataframes.append(df)

    heatmapAllSimilarities(arrayDataframes)



#----------------------#
# FUNCIONES AUXILIARES #
#----------------------#

def heatmapAllSimilarities(arrayDataframes_01):
    j = 0
    for i in range(len(arrayDataframes_01)):
        sims_df = arrayDataframes_01[i]
        if sims_df is not None:
            simMethod = required_parameters['similarityFunctions'][j]
            patients = list(sims_df.index)
            sims_list = sims_df.values.tolist()
            sims_numpy = np.array(sims_list)
            
            j += 1

            visualizationOfHeatMapBtwPatients(simMethod, sims_numpy, patients, required_parameters['annotated'])


############################################################################################################################        


if __name__ == "__main__":
    main() 
