# PhD Study 2 Machine Learning Application
This repository holds python and R code files for an exploration of data from my second PhD study (first timepoint/wave) with machine learning algorithms.

The data is from my second PhD study, available on psycharchives: https://doi.org/10.23668/psycharchives.14066.
Note that the first folder uses data from the first wave only, because the first wave holds the largest dataset (~ 400 participants) and is therefore the most appropriate for machine learning exploration (although still on the smaller side of datasets for deep learning). The second folder uses the data from all waves and uses data imputation to fill missing values to be able to train a recurrent neural network. The published data only holds data from the first wave that combines with either the second or third wave, so will differ slightly from the one used for the code here. 

This includes:

Depression Folder
- Code for data preparation: selection of demographic and wave 1 variables, data cleaning (incl. exclusion of participants with existing mental health disorder or taking pain medication) and data imputation in case of missing values
- Code for a DBSCAN to identify and remove outliers
- Code for a neural net with a continuous outcome (regression)
- Code for a neural net with binary outcome (classification)

Burnout Folder
- Code for data cleaning and imputation in R (imputation was carried out in R in order to use Amelia's EM algorithm with bootstrapping that is appropriate for time series data)
- Code for a recurrent neural network to predict burnout at timepoint 3 (total burnout, continuous, regression) based on features measured at timepoint 1 and 2
- Code for a standard deep neural network to predict burnout at timepoint 3 (total burnout, continous, regression) based on features at timepoint 1 only - a more practical solution

Both neural nets in the first folder are focused on the identification of depressive symptoms. The first NN focuses on predicting degree of depressive symptoms (regression), while the second focuses on identification of individuals of risk for clinical depression, which is based on the CES Depression scale (score of 16 or over considered at risk; see Radloff, 1977).

The recurrent neural network is focused on predicting burnout at 6 months after baseline measurement (wave 3) via burnout, physical symptoms, illness symptoms, depressive symptoms, sleep disruptions, and life satisfaction (measurement at waves 1 and 2). Another standard deep neural network was added her as a more practical solution in the case that coaches wish to evaluate burnout risk in their athletes based on measures taken prior to/at the beginning of the season. As such, this second model predicts burnout at the end of the season (timepoint 3) based on measures taken at timepoint 1 only. 

References
Radloff, L. S. (1977). The CES-D scale: A self report depression scale for research in the general population. Applied Psychological Measurements, 1, 385-401.
