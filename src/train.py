import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn
import scipy
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix


from PIL import Image



k=2
pic = Image.open("content/DataSet Signos/Y/IMG_1930.JPG")
pix = np.array(pic)
print(type(pic))

plt.imshow(pic.resize((28*k,28*k)))


# Clasificamos las imagenes
bs = 16  # bach size
k = 2
# Generador de imágenes de entrenamiento.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=(0.3),
    zoom_range=(0.3),
    width_shift_range=(0.2),
    height_shift_range=(0.2),
    validation_split=0.2,
    brightness_range=(0.05, 0.85),
    horizontal_flip=False)

# Carga de imágenes al generador de entrenamiento desde directorio.
train_generator = train_datagen.flow_from_directory(
    'content/DataSet Signos',
    class_mode='categorical',
    shuffle=True,
    target_size=(28 * k, 28 * k),
    color_mode='rgb',
    subset='training',
    batch_size=bs)

valid_generator = train_datagen.flow_from_directory(
    'content/DataSet Signos',
    class_mode='categorical',
    shuffle=True,
    target_size=(28 * k, 28 * k),
    color_mode='rgb',
    subset='validation',
    batch_size=bs)

#Printeado de imagenes
plt.imshow(next(train_generator)[0][2,...,0])
plt.show()

#Para saber a qué clase se le asocia a cada letra
print(train_generator.class_indices)

model = tf.keras.applications.VGG19()
model.summary()


num_classes = 24
epochs = 25

# VGG19
# Importamos el modelo que queremos utilizar.
VGG19_model = tf.keras.applications.VGG19(input_shape=(28*k,28*k,3),
                                          include_top=False,
                                          weights='imagenet')

print(len(VGG19_model.layers))
#Congelamos las 6 primeras capas del modelo que no seran reentrenadas
for layer in VGG19_model.layers[:6]:
  layer.trainable = False

# Creamos un nuevo modelo vacio.
model = tf.keras.Sequential()

# Añadimos el modelo preentrenado como si se tratase de una capa.
model.add(VGG19_model)

# Continuamos añadiendo más capas que sí serán entrenadas...
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))


## EJECUCION DEL MODELO

model.compile(loss="categorical_crossentropy",
              optimizer= SGD(learning_rate=0.01),
              metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='model/model.h5', verbose=1, save_best_only=True,
                               monitor='val_accuracy', mode = 'max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.000001)
#Procedemos a entrenar
history= model.fit(train_generator,
                             validation_data = valid_generator,
                             callbacks=[reduce_lr, checkpointer],
                             epochs=epochs)
