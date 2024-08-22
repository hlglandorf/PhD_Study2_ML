import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Data
df1 = pd.read_csv("s2w1impdb.csv")
print(df1)

# Create risk category for clinical depression (based on CES-D scale where score at 16 or above is risk for clinical depression)
df1['t1dsclass'] = np.where( df1.t1DS >= 16, 1, 0) 
print(df1.groupby(['t1dsclass']).size())

# Pick X and y
X = df1[['bio_sex', 'age', 'sport_type', 'injury', 'comp_years', 't1trainingload', 't1season', 't1DEV', 't1EXH', 't1RSA', 't1WURSSc', 't1PSQIgr', 't1LS']]
# using demographic data, recent injury history and current data on training load, season, devaluation, exhaustion, reduced sense of accomplishment, illness symptoms, sleep disruptions, and life satisfaction
y = df1['t1dsclass']

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# The groups (label y) are not equal in size, so resampling is required for training set 
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
y_train = y_resampled # after upsampling, y train needs to be updated

# Scale the data to avoid weighting of features pre model
scaler = StandardScaler()
scaler.fit(X_resampled)
X_train = scaler.transform(X_resampled)
X_test = scaler.transform(X_test)

# Data size check to make sure X and y are the same size after resampling
print("X resampled: ", len(X_resampled))
print("Y resampled: ", len(y_resampled))
print("X train: ", len(X_train))
print("Y train: ", len(y_train))
print("X test: ", len(X_test))
print("Y test: ", len(y_test))

# Build a neural net with tensorflow 
# Start with defining the input shape
input_shape = [X_train.shape[1]]
print(input_shape)

# Build  a basic model first
model1 = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Dense(units=104, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
print(model1.summary())

# Build on the basic model by creating a multilayer model with ReLu activation
model2 = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=128, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
print(model2.summary())

# Set up earlystopping and adam optimiser
callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
Adam = tf.keras.optimizers.AdamW(learning_rate=0.001)

# Compile the models with parameters to be shown during epochs
model1.compile(optimizer='adam', # using adam optimizer 
               loss = 'mae', # using mean absolute error
               metrics = ['accuracy']) # asking for accuracy for epochs
model2.compile(optimizer=Adam, # using adam (w) optimizer 
               loss = 'binary_crossentropy', # using binary crossentropy for binary outcome
               metrics = ['accuracy']) # asking for accuracy for epochs

# Fit model 1
losses1 = model1.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=256, 
                     epochs=100)
# Fit model 2
losses2 = model2.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=256, 
                     epochs=100,
                     callbacks=[callback]) #implementing early stopping rule to avoid overfitting
# models are similar in validation accuracy, but model 2 performs slightly better 

# Now analyse the loss and figure out if it is overfitting
loss_df1 = pd.DataFrame(losses1.history) #history stores the loss/val loss in each epoch
loss_df1.loc[:,['loss','val_loss']].plot()
plt.show()

loss_df2 = pd.DataFrame(losses2.history) #history stores the loss/val loss in each epoch
loss_df2.loc[:,['loss','val_loss']].plot()
plt.show() # validation loss seems to mostly level out but not increase again
