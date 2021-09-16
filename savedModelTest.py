import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras

#importuje model
model = tf.keras.models.load_model('saved_model\my_model1_2epochs')
img_height = 180
img_width = 180
class_names = ["Muz", "Zena"]

face_url = "https://i.imgur.com/nVLyony.png"
face_path = tf.keras.utils.get_file('nVLyony', origin=face_url)


img = keras.preprocessing.image.load_img(
    face_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

#vytvoření predikce
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Clovek na obrazku je {} s jistotou {:.2f}%"
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)