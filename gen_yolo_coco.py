import csv
import json
import os
import random
import shutil

OUTPUT_FOLDER = "dataset"
LABELS_FOLDER = f"{OUTPUT_FOLDER}/labels"
YOLO_TRAIN_FOLDER = f"{LABELS_FOLDER}/train"
YOLO_VAL_FOLDER = f"{LABELS_FOLDER}/val"

IMAGE_FOLDER = f"{OUTPUT_FOLDER}/images"
IMAGE_TRAIN_FOLDER = f"{IMAGE_FOLDER}/train"
IMAGE_VAL_FOLDER = f"{IMAGE_FOLDER}/val"

CSV_FILE = f"{OUTPUT_FOLDER}/annotations.csv"
COCO_FILE = f"{OUTPUT_FOLDER}/annotations_coco.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(YOLO_TRAIN_FOLDER, exist_ok=True)
os.makedirs(YOLO_VAL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_TRAIN_FOLDER, exist_ok=True)
os.makedirs(IMAGE_VAL_FOLDER, exist_ok=True)

assign_index = {
    'person': 0,
    # 'gun': 80,
    'smartphone': 81,
    # 'machinegun': 82,
    # 'knife': 83
}

rows = []
with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

for label, idx in assign_index.items():
    coco["categories"].append({
        "id": idx,
        "name": label
    })

images_added = set()
ann_id = 1

for row in rows:
    filename = row["file"]
    label = row["label"]

    if label not in assign_index:
        print(f"WARNING: Label [ {label} ] not in assign_index\nSKIPPING . . .")
        continue

    class_id = assign_index[label]

    # Parse bounding box
    coords = row["coordinates"].replace("(", "").replace(")", "")
    x1, y1, x2, y2 = map(int, coords.split(","))

    box_width = int(row["box_width"])
    box_height = int(row["box_height"])

    img_width = int(row["image_width"])
    img_height = int(row["image_height"])

    # YOLO format: class_id x_center y_center width height (all normalized)
    x_center = (x1 + box_width / 2) / img_width
    y_center = (y1 + box_height / 2) / img_height
    w_norm = box_width / img_width
    h_norm = box_height / img_height

    # Clamp values between 0 and 1
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))

    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(LABELS_FOLDER, txt_filename)

    with open(txt_path, "a") as yolo_f:
        yolo_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    if filename not in images_added:
        coco["images"].append({
            "id": filename,
            "file_name": filename,
            "width": img_width,
            "height": img_height
        })
        images_added.add(filename)

    coco["annotations"].append({
        "id": ann_id,
        "image_id": filename,
        "category_id": class_id,
        "bbox": [x1, y1, box_width, box_height],
        "area": box_width * box_height,
        "iscrowd": 0
    })

    ann_id += 1

with open(COCO_FILE, "w") as fjson:
    json.dump(coco, fjson, indent=4)

print("Complete!")
print(f"YOLO files: {YOLO_TRAIN_FOLDER}")
print(f"COCO file: {OUTPUT_FOLDER}")


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