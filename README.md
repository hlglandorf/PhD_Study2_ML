# PhD Study 2 Machine Learning Application
This repository holds python code files for an exploration of data from my second PhD study (first timepoint/wave) with machine learning algorithms.

The data is from my second PhD study, available on psycharchives: https://doi.org/10.23668/psycharchives.14066.
Note that only the data from the first wave is used for the algorithms here, because the first wave holds the largest dataset (~ 400 participants) and is therefore the most appropriate for machine learning exploration (although still on the smaller side of datasets for deep learning). The published data only holds data from the first wave that combines with either the second or third wave, so will differ slightly from the one used for the code here. 

This includes:
- Code for data preparation: selection of demographic and wave 1 variables, data cleaning (incl. exclusion of participants with existing mental health disorder or taking pain medication) and data imputation in case of missing values
- Code for a DBSCAN to identify and remove outliers
- Code for a neural net with a continuous outcome (regression)
- Code for a neural net with binary outcome (classification)

Both neural nets are focused on the identification of depressive symptoms. The first NN focuses on predicting degree of depressive symptoms (regression), while the second focuses on identification of individuals of risk for clinical depression, which is based on the CES Depression scale (score of 16 or over considered at risk; see Radloff, 1977).

References
Radloff, L. S. (1977). The CES-D scale: A self report depression scale for research in the general population. Applied Psychological Measurements, 1, 385-401.
