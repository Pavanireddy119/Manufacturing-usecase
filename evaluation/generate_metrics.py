import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

IMG_SIZE = (224,224)
BATCH_SIZE = 32

model = load_model("models/vehicle_damage_model.h5")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    predictions = (predictions > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predictions.flatten())

print(classification_report(
    y_true,
    y_pred,
    target_names=["Damage", "Whole"]
))
