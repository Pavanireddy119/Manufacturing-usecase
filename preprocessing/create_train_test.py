import shutil
import os

SOURCE = "datasets/preprocessed_dataset"

TRAIN_SOURCE = os.path.join(SOURCE, "training")
TEST_SOURCE = os.path.join(SOURCE, "validation")

TRAIN_DEST = "datasets/train"
TEST_DEST = "datasets/test"

shutil.copytree(TRAIN_SOURCE, TRAIN_DEST, dirs_exist_ok=True)
shutil.copytree(TEST_SOURCE, TEST_DEST, dirs_exist_ok=True)

print("Train/Test dataset created successfully")
