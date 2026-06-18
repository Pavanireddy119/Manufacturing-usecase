from PIL import Image
import os

dataset_path = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\Dataset\data1a"

corrupted = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        try:
            path = os.path.join(root, file)
            Image.open(path).verify()
        except:
            corrupted += 1

print("Corrupted Images:", corrupted)