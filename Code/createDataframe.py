
import pandas as pd
import numpy as np

def createDataframePatXPatZeros_givenListPats(listPats):
    nPats = len(listPats)
    dfSimilarities = pd.DataFrame(np.zeros((nPats, nPats)), columns=listPats, index=listPats, dtype=float)

    return dfSimilarities