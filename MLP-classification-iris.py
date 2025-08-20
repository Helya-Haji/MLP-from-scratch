import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"D:\my-github\MLP\Iris.csv")

sns.pairplot(data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species' )
plt.show()

x = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = data['Species']

scaler = StandardScaler()
scaled_data = scaler.fit_transform (x.values)

scaled_data = pd.DataFrame(scaled_data, index = x.index, columns = x.columns)

fig, (ax1, ax2) = plt.subplots(ncols=2 , figsize=(6,5))

ax1.set_title('before scaling')
sns.kdeplot(x['SepalLengthCm'], ax=ax1)
sns.kdeplot(x['SepalWidthCm'], ax=ax1)
sns.kdeplot(x['PetalLengthCm'],ax=ax1)
sns.kdeplot(x['PetalWidthCm'], ax=ax1)

ax2.set_title('after standard scaler')
sns.kdeplot(scaled_data['SepalLengthCm'], ax=ax2)
sns.kdeplot(scaled_data['SepalWidthCm'], ax=ax2)
sns.kdeplot(scaled_data['PetalLengthCm'], ax=ax2)
sns.kdeplot(scaled_data['PetalWidthCm'], ax=ax2)

plt.show()


x_train, x_test, y_train, y_test = train_test_split(scaled_data.values, y, test_size=0.3, random_state=40)

label_encoder = preprocessing.LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=4))
model.add(Dense(3, activation= 'softmax'))

sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = 1000, batch_size=32)

score = model.evaluate (x_test, y_test, verbose=0)

print('test score:', score[0])
print('test accuracy:', score[1])

y_pred = model.predict(x_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

confiusion= confusion_matrix(y_test_class, y_pred_class)
sns.heatmap(confiusion, annot=True, fmt='d', cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel(True)
plt.show()

results = pd.DataFrame.from_dict(history.history)

results['accuracy'].plot()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.show()


results['loss'].plot()
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('lOSS')
plt.show()