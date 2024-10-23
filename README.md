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
The code doesn't need any input files, but it requires a repository in [GraphDB Semantic Graph Database](https://www.ontotext.com/products/graphdb/). This repository must be an RDF* ontology following the data model described in [10.1109/JBHI.2024.3417224](https://ieeexplore.ieee.org/document/10568325) and [HospitalKG_changes](https://github.com/LorenaPujante/HospitalKG_Changes). [HospitalGeneratorRDF_V2](https://github.com/LorenaPujante/HospitalGeneratorRDF_V2) has been used to generate the data for the repository.

## 4. Execution
There are 4 _main_ python files to execute the different parts of the framework. Each file must be executed separately. Just go to the folder containing the folder and run: `python name_of_file.py`. All the parameters for STeMECH are in the file [config.py](https://github.com/LorenaPujante/STeMECH/blob/main/Code/config.py), which are described in the [next section](#5-configuration-params).

The parts of the frameworks are:
- [**main.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main.py): TODO
- [**main_Heatmap.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Heatmap.py): TODO
- [**main_Clustering_Ks.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Ks.py): TODO
- [**main_Clustering_Plots.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_Clustering_Plots.py): TODO

The file [**main_NumCases.py**](https://github.com/LorenaPujante/STeMECH/blob/main/Code/main_NumCases.py) can be used to search the number of positive cases for a microorganism for each week of the dataset. It also searches the cases by week and floor. It can be used to have an approximate idea of the number of patients whose trajectories will be studied depending on the parameters' values.  

## 5. configuration Params
TODO

## 6. Output
TODO
