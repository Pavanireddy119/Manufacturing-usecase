import os

dataset_path = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\Dataset\data1a"

total_images = 0

for root, dirs, files in os.walk(dataset_path):
    if len(files) > 0:
        print(f"{root} : {len(files)} images")
        total_images += len(files)

print("\nTotal Images:", total_images)