import os
import csv
from PIL import Image, ImageDraw

OUTPUT_FOLDER = "dataset"
IMAGE_FOLDER = f"{OUTPUT_FOLDER}/images"
ANNOTATED_DIR = f"{OUTPUT_FOLDER}/annotated_images"
CSV_FILE = f"{OUTPUT_FOLDER}/annotations.csv"

os.makedirs(ANNOTATED_DIR, exist_ok=True)

def parse_coordinates(coord_str):
    # "(x1, y1, x2, y2)" → tuple of ints
    coord_str = coord_str.replace("(", "").replace(")", "")
    x1, y1, x2, y2 = coord_str.split(",")
    return int(x1), int(y1), int(x2), int(y2)

def draw_boxes():
    # Read annotations into memory
    annotations = {}

    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file"]
            label = row["label"]
            coords = parse_coordinates(row["coordinates"])

            annotations.setdefault(fname, []).append((label, coords))

    # Process each image with boxes
    for fname, items in annotations.items():
        img_path = os.path.join(IMAGE_FOLDER, fname)

        if not os.path.exists(img_path):
            print(f"Skipping missing file: {fname}")
            continue

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        for label, (x1, y1, x2, y2) in items:
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label background + text
            text = str(label)
            text_width = draw.textlength(text)
            text_height = 14
            draw.rectangle([x1, y1 - text_height, x1 + text_width + 4, y1], fill="red")
            draw.text((x1 + 2, y1 - text_height), text, fill="white")

        save_path = os.path.join(ANNOTATED_DIR, fname)
        img.save(save_path)

        print(f"Saved annotated image: {save_path}")


if __name__ == "__main__":
    draw_boxes()
