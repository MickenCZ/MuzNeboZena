#dataset = https://www.kaggle.com/ashwingupta3012/male-and-female-faces-dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf


import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

data_dir = "D:\Code VS projects\image classifier\Male and Female face dataset"
batch_size = 32
img_height = 180
img_width = 180
#definuje charakteristiky obrázků

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#definuje, jaká část dat je na trénování a jaká na ověřování

#Jména složek jsou jména skupin obrázků (žena, muž)
class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
#data jsou upraveny automaticky

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
#konvertuje RGB hodnoty do decimální podoby.

num_classes = 2
#kolik skupin mám (muž, žena, takže dvě)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
#definuje model

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#kompiluje model

epochs=7
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
#model.fit trénuje model. Čím víc epoch, tím déle to trvá ale je to potom přesnější. Pro domácí potřeby doporučuju 2-3

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show() 

#pokud toto odkomentujete tak to vygeneruje graf přesnosti.

face_url = "https://i.imgur.com/Qo6qESS.png"
face_path = tf.keras.utils.get_file('Qo6qESS', origin=face_url)
#importuje obrázek na detekci. Jen na testování, predikce přímo v pythonu se dělají v druhém souboru. (savedModelTest.py)

img = keras.preprocessing.image.load_img(
    face_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) #Vytvoří z obrázku array a z něj array čtyřdimenzionální, protože to je datatyp kterej se používá.

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
     "Clovek na obrazku je {} s jistotou {:.2f}%"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

#tfjs.converters.save_keras_model(model, "saved_model/my_modeljs_7epochs")
#uloží model do formátu přijatelného javascriptem.

#model.save('saved_model/my_model1_7epochs')
#uloží model do souboru