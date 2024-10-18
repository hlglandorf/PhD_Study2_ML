import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Import data from second PhD study
df = pd.read_csv("s2wideimputed.csv")
print(df)

# Pick X and y
X = df[['EXH_1', 'EXH_2', 'DEV_1', 'DEV_2', 'RSA_1', 'RSA_2', 'trainload_1', 'trainload_2', 'PS_1', 'PS_2', 'LS_1', 'LS_2', 'PSQIg_1', 'PSQIg_2', 'WURSSg_1', 'WURSSg_2']]
y = df[['EXH_3', 'DEV_3', 'RSA_3']]

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
dump(scaler, 'scaler.joblib') # save the scaler to a file

# Reformat X to feed into recurrent layer
X_train = X_train.reshape((X_train.shape[0], 2, 8))
X_test = X_test.reshape((X_test.shape[0], 2, 8))

print(X_test)
print(X_train)

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation = 'tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), 
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(units=3) #output without activation function since this is a regression problem
])
print(model.summary())

# Set up earlystopping and adam optimiser
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
Adam = tf.keras.optimizers.Adam(learning_rate=0.003)

# Compile other parameters for model
model.compile(optimizer='adam', # using adam optimizer 
               loss = 'mae', # using mean absolute error
               metrics = ['mean_absolute_error', 'mean_squared_error']) # asking for mae and mse for epochs (appropriate for regression)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'checkpoint.keras', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min'
)

history = model.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=64, 
                     epochs=100,
                     callbacks=[callback, checkpoint])

