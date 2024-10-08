import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import data from second PhD study
df = pd.read_csv("s2wideimputed.csv")
print(df)

# Pick X and y
X = df[['trainload_1', 'trainload_2', 'BURN_1', 'BURN_2', 'PS_1', 'PS_2', 'LS_1', 'LS_2', 'PSQIg_1', 'PSQIg_2', 'WURSSg_1', 'WURSSg_2']]
y = df['BURN_3']

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Reformat X
X_train = X_train.reshape((X_train.shape[0], 2, 6))
X_test = X_test.reshape((X_test.shape[0], 2, 6))

print(X_test)
print(X_train)

# Create a model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation = 'tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(50, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(units=1) #output without activation function since this is a regression problem
])
print(model.summary())

# Set up earlystopping and adam optimiser
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
Adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile other parameters for model
model.compile(optimizer='adam', # using adam optimizer 
               loss = 'mae', # using mean absolute error
               metrics = ['mean_absolute_error', 'mean_squared_error']) # asking for mae and mse for epochs (appropriate for regression)

losses = model.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=256, 
                     epochs=100,
                     callbacks=[callback])

# Check whether model does better than baseline prediction
baseline_pred = np.mean(y_train)
baseline_mae = np.mean(np.abs(y_test - baseline_pred))
baseline_mse = np.mean((y_test - baseline_pred)**2)
print('baseline mae: ', baseline_mae, 'baseline mse: ', baseline_mse)
