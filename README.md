# StESPT

TODO

**IMPORTANT:** Also read the file [PARAMS.md](https://github.com/LorenaPujante/STeMECH/blob/main/PARAMS.md)

## 0. Related Repositories
Below, we present some other related repositories that may be of interest to you:
- [**HospitalKG_changes**](https://github.com/LorenaPujante/HospitalKG_Changes): It is also linked to [~~doi: TODO~~](NULL).
- [**HospitalEdgeWeigths**](https://github.com/LorenaPujante/HospitalEdgeWeigths): It is also linked to [~~doi: TODO~~](NULL).
- [**HospitalGeneratorRDF_V2**](https://github.com/LorenaPujante/HospitalGeneratorRDF_V2): Code used to generate the input dataset in RDF* for STeMECH based on the output from [**H-Outbreak**](https://github.com/denissekim/Simulation-Model).


## 1. Other sections
TODO


## 2. Installation
The source code is currently hosted on [github.com/LorenaPujante/STeMECH/Code](https://github.com/LorenaPujante/STeMECH/Code).

The code is in Python 3.10. The following packages are needed:
- matplotlib v3.9.2
- networkx v3.2.1
- numpy v2.1.2
- pandas v1.5.3
- scikit_learn v1.5.2
- scipy v1.14.1
- SPARQLWrapper v2.0.0
 

## 3. Input
The code doesn't need any input files to read but requires a repository in [GraphDB Semantic Graph Database](https://www.ontotext.com/products/graphdb/) to query the data about patients. 

This repository must be an RDF* ontology following the data model described in [10.1109/JBHI.2024.3417224](https://ieeexplore.ieee.org/document/10568325) and [HospitalKG_changes](https://github.com/LorenaPujante/HospitalKG_Changes). [HospitalGeneratorRDF_V2](https://github.com/LorenaPujante/HospitalGeneratorRDF_V2) has been used to generate the data for the repository.

The RDF* ontology with the dataset for the experiments of [~~doi: TODO~~](NULL) can be found in [**dataset/HospitalGeneratorRDF_V2_output**](https://github.com/LorenaPujante/STeMECH/tree/main/dataset/HospitalGeneratorRDF_V2_output). In addition, the input data to generate the ontology is in [dataset/H-Outbreak_output](https://github.com/LorenaPujante/STeMECH/tree/main/dataset/H-Outbreak_output).


## 4. Execution
There are 4 _main_ python files to execute the different parts of the framework. Each file must be executed separately. Go to the folder containing the folder and run: `python name_of_file.py`. All the parameters for STeMECH are in the file [config.py](https://github.com/LorenaPujante/STeMECH/blob/main/Code/config.py), which are described in the [next section](#5-configuration-params).

The parts of the frameworks are:
- [**main.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main.py): TODO
- [**main_Heatmap.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Heatmap.py): TODO
- [**main_Clustering_Ks_sil.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Ks.py): TODO
- [**main_Clustering_Plots_sil.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Plots.py): TODO

The file [**main_NumCases.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_NumCases.py) can be used to search the number of positive cases for a microorganism for each week of the dataset. It also searches the cases by week and floor. It can be used to have an approximate idea of the number of patients whose trajectories will be studied depending on the parameters' values.  


## 5. Configuration params
Here there are the parameters for the execution of, mainly, _main.py_ but also the rest of the _main_files:
- **zero**: It indicates if it is necessary to ask the database for the matrixes to calculate the spatial distance. If _False_, they must be stored in _Code/matrixes_.
- **repository**: The name of the GraphDB repository with the input dataset.
- **dateStart**: The date and time to start searching for patients with a positive _TestMicro_ for a specific _Microorganism_.
- **dateEmd**: The date and time to stop the search for patients with a positive _TestMicro_ for a specific _Microorganism_.
- **idLoc**: Value for the _id_ attribute of the _Floor_ where to search the infected patients.
- **idMicroorg**: The value for the _id_ attribute of the _Microorganism_ whose infected patients we are searching.
- **maxDaysTrajForward**: When we already have found the patients infected during a period, we will also search for other events of these patients, at most, during the indicated days.
- **similarityFunctions**: The _ids_ of the _Trajectory similarity measurement_ algorithms to be run. The allowed values are:
  - _dtw_: for Dynamic Time Warping (DTW).
  - _dtw_st_: for Spatiotemporal DTW (ST-DTW).
  - _lcss_: for Spatiotemporal Longest Common Subsequence (ST-LCSS).
  - _lcss_2_: for ST-LCSS With Time Window (ST-LCSS-WTW).
  - _tsJoin_: for Spatiotemporal Linear Combine (STLC).
  - _tsJoin_2_: for Joint Spatiotemporal Linear Combine (JSTLC).    
- **beta**: The β parameter of the equation for _temporal similarity_ between sampling points.
- **alfa**: The α parameter of the equation for the _spatiotemporal similarity_ between sampling points.
- **maxStepsBackwardLCSS**: For the _LCSS_ and _LCSS_WTW_ algorithms, the maximum allowed number of difference between two steps. If the distance in steps is bigger than this value, there won't be a match between the sampling points.
- **margin**: For the _LCSS_WTW_ algorithm, the number of steps with which we do the match check forward and backwards.
- **maxSpDist**: For the _LCSS_ and _LCSS_WTW_ algorithms, the maximum spatial distance allowed between two _Beds_ to allow matching two sampling points. 
- **maxDiffStepsSTLC**: For the _STLC_ and _JSTLC_ algorithms, if it is _True_, the temporal and spatial similarities will be divided between the number of steps of our search time. If it is False, they will be divided (as the original STLC) between the number of steps of the compared trajectories.
- **nameFolder_Matrix**: The path to the folder where to save the matrixes to calculate the spatial similarity. They are saved in CSV files. The headers of the matrixes (the id of the locations) are saved in this folder.   
- **nameFolder_SimArrays**: The path to the folder where to save the similarity matrixes between patients' trajectories. They are saved in CSV files. The result of each algorithm is saved in a separate file. The arrays with the similarities normalised to range [0,1] are also saved here with the name ending with "__01_".
- **nameFolder_Figures**: The path to the folder where to save the figures with the resume of the results of all the _main_ files.
- **nameFolder_Outputs**: The path to the folder where to save the matrixes to calculate the spatial similarity.
- **timeInFile**: If it is _True_, the execution time of each similarity measurement algorithm for each pair of trajectories will be saved in a file. 
- **nameFolder_Time**: The path to the folder with the file with the execution time of each similarity measurement algorithm for each pair of trajectories.

Here there are the parameters for some of the _main_ files.
- **annotated**: This parameter is only used in _main_Heatmap.py_. If it is _True_, the heatmaps with the trajectory similarities between patients will also show in each cell the value of the trajectory similarity between the pair of patients.
- **heatColors**: This parameter is only used in _main_Heatmap.py_. The name of the [colour scheme](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for the heatmaps.
- **maxClustersPats**: This parameter is only used in _main_Clustering_Ks.py_. If it is _True_, when searching for the best value of _K_ for the K-Means clustering algorithm, the tested _Ks_ will be in the range [2, *_*numPatients-1_**]. If it is _False_, the _Ks_ will be in the range [2, **_numPatients/2_**].
- **numRows**: This parameter is used in _main_Clustering_Ks.py_ and _main_Clustering_Plots.py_. It determines how many rows will have the image that shows the bar charts of the clustering metrics for each value of _K_ or trajectory similarity algorithm.
- **meshSize**: This parameter is only used in _main_Clustering_Plots.py_. It determines the "definition" of the chart showing the points of each cluster in a bi-dimensional chart.
- **Ks**: This parameter is only used in _main_Clustering_Plots.py_. It is an array that for each trajectory similarity measurement algorithm saves which value of _K_ returns the clusters with the optimum cohesion and separation. It must be the same size as _similarityFunctions_ and follow its order.
- **reducedColors**: This parameter is only used in _main_Clustering_Plots.py_. The name of the [colour scheme](https://matplotlib.org/stable/users/explain/colors/colormaps.html) for the bi-dimensional representation of the clusters. 
-  **barColors**: This parameter is only used in _main_Clustering_Plots.py_. It is an array with the name of the colours for the bars of the charts that show the clustering metrics for each trajectory similarity algorithm.

In the file [PARAMS.md](https://github.com/LorenaPujante/STeMECH/blob/main/PARAMS.md) we present the values for all these parameters used to create the dataset for the work [~~doi: TODO~~](NULL).


## 6. Output
TODO
