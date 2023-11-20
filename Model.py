import pandas as pd
import os
import zipfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

folder = r'C:\Users\thynnea\Downloads\heartfailure'

files = os.listdir(folder)
csv_file_name = [file for file in files if file.endswith('.csv')][0]
data = pd.read_csv(os.path.join(folder, csv_file_name))
data['death_event'] = data['DEATH_EVENT'].replace({1: 'yes', 0: 'no'})
print(data.head())
print(Counter(data['death_event']))
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42) 
ct = ColumnTransformer([('numeric', StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)
le = LabelEncoder()
y_train = le.fit_transform(y_train.astype(str))
y_test = le.transform(y_test.astype(str))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(InputLayer(input_shape = (x_train.shape[1],)))

model.add(Dense(12, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size = 16, verbose = 1)
loss, acc = model.evaluate(x_test, y_test, verbose = 0)
print("Loss: ", loss)
print("Accuracy: ", acc)
y_hat = model.predict(x_test, verbose = 0)
y_estimate = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis = 1)
print(classification_report(y_true, y_estimate))
