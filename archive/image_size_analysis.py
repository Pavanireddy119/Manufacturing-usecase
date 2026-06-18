from PIL import Image
import os

dataset_path = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\Dataset\data1a"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        try:
            img_path = os.path.join(root, file)

            img = Image.open(img_path)

            print("Image Size:", img.size)

            exit()  # show first image size only

        except:
            pass