from PIL import Image
import os

INPUT_DIR = "datasets/Dataset/data1a"

OUTPUT_DIR = "datasets/preprocessed_dataset"

TARGET_SIZE = (224, 224)

processed = 0

for root, dirs, files in os.walk(INPUT_DIR):

    for file in files:

        try:
            input_path = os.path.join(root, file)

            relative_path = os.path.relpath(input_path, INPUT_DIR)

            output_path = os.path.join(
                OUTPUT_DIR,
                relative_path
            )

            os.makedirs(
                os.path.dirname(output_path),
                exist_ok=True
            )

            img = Image.open(input_path)

            img = img.convert("RGB")

            img = img.resize(TARGET_SIZE)

            img.save(output_path)

            processed += 1

        except Exception as e:
            print("Error:", e)

print(f"Processed {processed} images")
