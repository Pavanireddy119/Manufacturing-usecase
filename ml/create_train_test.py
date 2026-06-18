import shutil
import os

SOURCE = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\preprocessed_dataset"

TRAIN_SOURCE = os.path.join(SOURCE, "training")
TEST_SOURCE = os.path.join(SOURCE, "validation")

TRAIN_DEST = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\train"
TEST_DEST = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\test"

shutil.copytree(TRAIN_SOURCE, TRAIN_DEST, dirs_exist_ok=True)
shutil.copytree(TEST_SOURCE, TEST_DEST, dirs_exist_ok=True)

print("Train/Test dataset created successfully")