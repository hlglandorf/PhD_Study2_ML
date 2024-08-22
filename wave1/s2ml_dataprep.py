import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Import data from second PhD study
df = pd.read_csv("s2full.csv")

# Select Variables (demographic information and data from wave 1)
df1 = df[['id', 'bio_sex', 'age', 'sport_type', 'mental_disorder', 'pain_meds', 'injury', 'comp_years', 't1trainingload', 't1season', 't1BURN', 't1DEV', 't1EXH', 't1RSA', 't1DS', 't1WURSSc', 't1PSQIgr', 't1LS']]
df1 = df1.dropna(subset=['mental_disorder'])
df1 = df1.loc[df1['mental_disorder'] == 1]
df1 = df1.loc[df1['pain_meds'] == 1]
df2 = df1[['bio_sex', 'age', 'sport_type', 'injury', 'comp_years', 't1trainingload', 't1season', 't1BURN', 't1DEV', 't1EXH', 't1RSA', 't1DS', 't1WURSSc', 't1PSQIgr', 't1LS']]

# Impute missing values
imputer = IterativeImputer(max_iter=10, random_state=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df2), columns=df2.columns)
print(df_imputed)

# Write out datafile to use in subsequent analyses
df_imputed.to_csv('s2w1imputed.csv', index=False)
