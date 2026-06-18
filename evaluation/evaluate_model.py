import tensorflow as tf
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load saved model
model = load_model("models/vehicle_damage_model.h5")

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

loss, accuracy = model.evaluate(test_ds)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)
print("\nModel Loaded Successfully")
