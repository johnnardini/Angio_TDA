# angiogenesis_TDA
 
Code for "Topological data analysis distinguishes parameter regimes in the Anderson-Chaplain model of angiogenesis" by John Nardini, Bernadette Stolz, Kevin Flores, Heather Harrington, and Helen Byrne. 
angio.py is the main code for this document, as it includes the angio_abm class, simulates the Anderson-Chaplain agent-based model of tumor-induced angiogenesis. 

## Code description

In this study, we 1. perform simulations of the Anderson Chaplain Model over many parameter values, 2. perform data analysis by measuring standard descriptors and applying topological filtrations, and 3. perform basic clustering analysis usint the descriptor vectors from part (2). 

Part (1) of our analysis can be performed for one model parameter combination in Chaplain_Anderson_TDA.ipynb, or you can loop over many parameter combinations using Chaplain_Anderson_TDA.py (Note: if you ran the latter code as is, it will take multiple days to run). 

The code used to for step (2) to create the standard and topological descriptor vectors are also provided in angio.py. and called during or after model implementation in Chaplain_Anderson_TDA.ipynb and Chaplain_Anderson_TDA.py.

Step (3) of our analysis is performed in bio_clustering_standard_descriptors.ipynb for the standard descriptor vectors and for TDA_clustering_clustering_plane_flood.ipynb for the topological descriptor vectors. This code relies on the script custom_functions.py.

## Code dependencies:

We used Gudhi for implementation of our data analysis. 

Please contact John Nardini at jtnardin@ncsu.edu if you have any questions.
