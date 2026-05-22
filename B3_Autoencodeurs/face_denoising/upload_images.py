import os
import random
import shutil

# dossier contenant toutes les images
source_folder = "data/archive/images"

# dossiers de sortie
train_folder = "anime_faces_split/train"
test_folder = "anime_faces_split/test"

# création des dossiers
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# récupérer toutes les images
images = [
    f for f in os.listdir(source_folder)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

# mélanger les images
random.shuffle(images)

# ratio train/test
split_ratio = 0.8

split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
test_images = images[split_index:]

# copier dans train
for img in train_images:
    src = os.path.join(source_folder, img)
    dst = os.path.join(train_folder, img)
    shutil.copy(src, dst)

# copier dans test
for img in test_images:
    src = os.path.join(source_folder, img)
    dst = os.path.join(test_folder, img)
    shutil.copy(src, dst)

print("Train images :", len(train_images))
print("Test images :", len(test_images))