import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import data from second PhD study
df = pd.read_csv("s2wideimputed.csv")
print(df)

# Pick X and y
X = df[['BURN_1', 'trainload_1', 'PS_1', 'LS_1', 'PSQIg_1', 'WURSSg_1']]
y = df['BURN_3']

# Split the dataset into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler on training data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build a neural net with tensorflow 
# Start with defining the input shape
input_shape = [X_train.shape[1]]
print(input_shape)

# Create a model
model = tf.keras.Sequential([
    tf.keras.Input(shape=input_shape),
    tf.keras.layers.Dense(units=42, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=18, activation='elu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(units=6, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(units=1) #no activation/only linear activation on output layer
])
print(model.summary())

# Set up earlystopping and adam optimiser
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
Adam = tf.keras.optimizers.Adam(learning_rate=0.005)

# Compile other parameters for model
model.compile(optimizer='adam', # using adam optimizer 
               loss = 'mae', # using mean absolute error
               metrics = ['mean_absolute_error', 'mean_squared_error']) # asking for mae and mse for epochs (appropriate for regression)

losses = model.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     batch_size=64, 
                     epochs=200,
                     callbacks=[callback])

# Now analyse the loss and figure out if it is overfitting
loss_df = pd.DataFrame(losses.history) #history stores the loss/val loss in each epoch
loss_df.loc[:,['loss','val_loss']].plot()
plt.show()

# Check whether model does better than baseline prediction (based on burnout at timepoint 1)
baseline_pred = np.mean(X_train[:, 0])
baseline_mae = np.mean(np.abs(y_test - baseline_pred))
baseline_mse = np.mean((y_test - baseline_pred)**2)
print('baseline mae: ', baseline_mae, 'baseline mse: ', baseline_mse)