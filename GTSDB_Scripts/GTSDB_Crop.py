import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

original_train_dir = "GTSDB/Train"
new_train_dir = "GTSDB_Classification/Train"
new_test_dir = "GTSDB_Classification/Test"
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

# Loop class folders and split data
for class_name in os.listdir(original_train_dir):
    class_path = os.path.join(original_train_dir, class_name)
    
    if os.path.isdir(clss_path):  # Ignore non-folder files
        images = os.listdir(class_path)

        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
        os.makedirs(os.path.join(new_train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(new_test_dir, class_name), exist_ok=True)

        # Move images
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(new_train_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(new_test_dir, class_name, img))

print("Dataset cropped")
