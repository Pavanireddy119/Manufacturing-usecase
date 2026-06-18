import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)

x = Dense(128, activation="relu")(x)

x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

model.save("vehicle_damage_model.h5")

print("Model Saved Successfully")