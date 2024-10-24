# STeMECH

TODO

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
- [**main_Clustering_Ks.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Ks.py): TODO
- [**main_Clustering_Plots.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Plots.py): TODO

The file [**main_NumCases.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_NumCases.py) can be used to search the number of positive cases for a microorganism for each week of the dataset. It also searches the cases by week and floor. It can be used to have an approximate idea of the number of patients whose trajectories will be studied depending on the parameters' values.  


## 5. Configuration params
The parameters for STeMECH are the following:
- **zero**: It indicates if it is necessary to ask the database for the matrixes to calculate the spatial distance. If _False_, they must be stored in _Code/matrixes_.
- **repository**: The name of the GraphDB repository with the input dataset.
- **dateStart**: The date and time to start searching for patients with a positive _TestMicro_ for a specific _Microorganism_.
- **dateEmd**: The date and time to stop the search for patients with a positive _TestMicro_ for a specific _Microorganism_.
- **idLoc**: Value for the _id_ attribute of the _Floor_ where to search the infected patients.
- **idMicroorg**: The value for the _id_ attribute of the _Microorganism_ whose infected patients we are searching.
- **maxDaysTrajForward**: When we already have found the patients infected during a period, we will also search for other events of these patients, at most, during the indicated days.
- **similarityFunctions**: The _ids_ of the _Trajectory similarity measurement_ algorithms to be run. The allow values are:
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
- **nameFolder_Matrix**: The path to the folder where to store the matrixes to calculate the spatial similarity.
- **nameFolder_Matrix**: The path to the folder where to store the matrixes to calculate the spatial similarity.
- **nameFolder_Matrix**: The path to the folder where to store the matrixes to calculate the spatial similarity.
- **nameFolder_Matrix**: The path to the folder where to store the matrixes to calculate the spatial similarity.
- **timeInFile**: 
- **nameFolder_Matrix**: The path to the folder where to store the matrixes to calculate the spatial similarity.
## 6. Output
TODO
