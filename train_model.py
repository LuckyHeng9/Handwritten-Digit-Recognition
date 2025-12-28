import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = "dataset"
MODEL_PATH = "digit_cnn_model.h5"
IMG_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 100

train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR + "/train",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR + "/val",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR + "/test",
    color_mode="grayscale",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

norm = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (norm(x), y))
val_ds = val_ds.map(lambda x, y: (norm(x), y))
test_ds = test_ds.map(lambda x, y: (norm(x), y))

model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

model.save(MODEL_PATH)
