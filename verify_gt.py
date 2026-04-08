import os
import json
from PIL import Image, ImageDraw
from collections import defaultdict

STATS_JSON = "output/stats/neo_image_stats.json"
TEST_IMAGE_PATH = "datasets/image_test_gt/005753_jpg.rf.6af0bc88a1c368557756505de4a497a2.jpg" 
OUTPUT_DIR = "resources/ground_truth/gt_images"
gt_ver_dir = "resources/ground_truth/gt_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(gt_ver_dir, exist_ok=True)

# Ground truth boxes (pixel coords)
# Example: [[x1, y1, x2, y2], ...]
GT_BOXES = [
    [54, 47, 249, 259]
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_unique_square_sizes(stats_json):
    with open(stats_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    resolution_counts = data["statistics"][0]

    sizes = set()

    for res in resolution_counts.keys():
        w, h = map(int, res.split("x"))

        if w == h and w % 320 == 0:
            sizes.add(w)

    return sorted(list(sizes))


def scale_boxes(boxes, original_size, new_size):
    scale = new_size / original_size

    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box

        scaled.append([
            x1 * scale,
            y1 * scale,
            x2 * scale,
            y2 * scale
        ])

    return scaled


def scale_non_square(boxes, original_size, new_width, new_height):
    scale_w = new_width / original_size
    scale_h = new_height / original_size

    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box

        scaled.append([
            x1 * scale_w,
            y1 * scale_h,
            x2 * scale_w,
            y2 * scale_h
        ])

    return scaled


def generate_tests(image_path, sizes, gt_boxes):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size[0]  # square

    mapping = {
        "original_size": original_size,
        "gt_boxes": gt_boxes,
        "variants": {}
    }

    for size in sizes:
        # Resize image
        resized = img.resize((size, size), Image.LANCZOS)

        filename = f"{size}x{size}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        resized.save(save_path)

        # Scale GT boxes
        scaled_boxes = scale_boxes(gt_boxes, original_size, size)

        mapping["variants"][filename] = {
            "size": size,
            "scaled_boxes": scaled_boxes
        }

    return mapping


def save_mapping(mapping):
    output_path = os.path.join(OUTPUT_DIR, "gt_mapping.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    return output_path


def image_test_process():
    sizes = get_unique_square_sizes(STATS_JSON)
    print(f"Found sizes: {sizes}")
    mapping = generate_tests(TEST_IMAGE_PATH, sizes, GT_BOXES)
    output_path = save_mapping(mapping)
    print("Test images + GT mapping generated")
    
    return output_path


def draw_gtboxes(mapping_json_path:str, image_dir:str, color:tuple, output_dir:str, file_title:str="verify_box", dataset_title:str=""):
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    variants = mapping["variants"]

    for filename, data in variants.items():
        add_name = f"{dataset_title}_{filename}" if dataset_title else f"{filename}"
        image_path = os.path.join(image_dir, add_name)

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for box in data["scaled_boxes"]:
            x1, y1, x2, y2 = box

            # Draw rectangle (color outline, thickness simulated)
            for i in range(3):  # thickness
                draw.rectangle(
                    [x1 - i, y1 - i, x2 + i, y2 + i],
                    outline=color
                )

        verify_name = f"{file_title}_{filename}"
        save_path = os.path.join(output_dir, verify_name)

        img.save(save_path)

    print("Verification images created")


def main():
    color = (0, 255, 0)  # Green for GT boxes
    output_path = image_test_process()
    draw_gtboxes(output_path, OUTPUT_DIR, color, gt_ver_dir)

if __name__ == "__main__":
    main()