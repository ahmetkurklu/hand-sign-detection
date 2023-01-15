import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

train_folder = "./images/train"
data_dir = pathlib.Path(train_folder)
image_count = len(list(data_dir.glob('*/*.png')))
print("Nombre d'image pour le training: {}".format(image_count))

#############################################################
batch_size = 32
h_image = 180
l_image = 180

#Création du dataset de training
train_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(h_image, l_image),
  batch_size=batch_size)

#Création du dataset de validation
val_dataset = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(h_image, l_image),
  batch_size=batch_size)


normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
# Normalisation du dataset -> valeur RGB (0,255) => valeur [0,1] (plus opti pour le réseau de neurones)


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
# Optimisation du dataset en perf.

num_classes = 8
# Création du modèle
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=6
)
# epochs : nombre d'itération du training
