


from tensorflow import keras as ks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sas
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("/content/drive/MyDrive/gold_cad (1).csv")
df

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')


df.sort_values('Date', inplace=True)


scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df[['Close']])


df = df[['Close']]

df = np.asarray(df)

training_data_len = int(len(df) * 0.6)
xtrain = df[0:training_data_len].reshape((-1, 1, 1))
ytrain = df[1:training_data_len+1].reshape((-1, 1))
xtest = df[training_data_len:-1].reshape((-1, 1, 1))
ytest = df[training_data_len+1:].reshape((-1, 1))



model = ks.Sequential()
model.add(ks.layers.Input((1,1)))
model.add(ks.layers.LSTM(units=78, activation="relu", return_sequences=True))
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.LSTM(units=78, activation="relu", return_sequences=False))
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.Dense(units=10, activation="relu"))
model.add(ks.layers.Dense(units=1))

model.compile(optimizer=tf.optimizers.Adam(), loss='mean_squared_error',)

hist = model.fit(xtrain, ytrain, epochs=100, validation_data=(xtest, ytest),callbacks=ks.callbacks.EarlyStopping(patience=20,restore_best_weights=True) ,verbose=1)

loss= model.evaluate(xtest, ytest)
print(f"Test Loss: {loss}")

predictions = model.predict(xtest)
plt.figure(figsize=(20,10))
plt.plot(ytest,linewidth=3, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()







