"""For already made datasets in YOLO format, sort them into train and val folders."""

import os
import random
import shutil

OUTPUT_FOLDER = "DATASET DIR HERE"
LABELS_FOLDER = f"{OUTPUT_FOLDER}/labels"
YOLO_TRAIN_FOLDER = f"{LABELS_FOLDER}/train"
YOLO_VAL_FOLDER = f"{LABELS_FOLDER}/val"

IMAGE_FOLDER = OUTPUT_FOLDER
IMAGE_TRAIN_FOLDER = f"{IMAGE_FOLDER}/train"
IMAGE_VAL_FOLDER = f"{IMAGE_FOLDER}/val"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(YOLO_TRAIN_FOLDER, exist_ok=True)
os.makedirs(YOLO_VAL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_TRAIN_FOLDER, exist_ok=True)
os.makedirs(IMAGE_VAL_FOLDER, exist_ok=True)

# Create train/val images by spiltting 30% of images in image folder into val folder and copying the 70% into train folder
duplicate_percent=0.3

# Get all image files
files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if os.path.isfile(os.path.join(IMAGE_FOLDER, f))
]

# Shuffle files
random.shuffle(files)

# Split index
split_index = int(len(files) * duplicate_percent)

val_files = files[:split_index]
train_files = files[split_index:]

# Copy Train Files
for f in train_files:
    # Image
    img_src = os.path.join(IMAGE_FOLDER, f)
    img_dst = os.path.join(IMAGE_TRAIN_FOLDER, f)
    shutil.copy(img_src, img_dst)

    # Label
    txt_file = os.path.splitext(f)[0] + ".txt"
    yolo_src = os.path.join(LABELS_FOLDER, txt_file)
    yolo_dst = os.path.join(YOLO_TRAIN_FOLDER, txt_file)

    if os.path.exists(yolo_src):
        shutil.copy(yolo_src, yolo_dst)

# Copy Val Files
for f in val_files:
    # Image
    img_src = os.path.join(IMAGE_FOLDER, f)
    img_dst = os.path.join(IMAGE_VAL_FOLDER, f)
    shutil.copy(img_src, img_dst)

    # Label
    txt_file = os.path.splitext(f)[0] + ".txt"
    yolo_src = os.path.join(LABELS_FOLDER, txt_file)
    yolo_dst = os.path.join(YOLO_VAL_FOLDER, txt_file)

    if os.path.exists(yolo_src):
        shutil.copy(yolo_src, yolo_dst)

print(f"Train images: {len(train_files)}")
print(f"Val images: {len(val_files)}")