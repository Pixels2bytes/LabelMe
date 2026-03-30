import os
import cv2
import glob

IMAGE_DIR = "create dataset/dangerous_weapons/images/train"
LABEL_DIR = "create dataset/dangerous_weapons/labels/train"
OUTPUT_DIR = "output/verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["person", "gun", "smartphone", "hand", "knife"]
VALID_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")


def draw_yolo_boxes(image_path, label_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"[WARNING] Bad label format: {label_path}")
            continue

        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])

        # Convert YOLO to pixel coords
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Draw box
        color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


image_paths = []
for ext in VALID_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

total_images = len(image_paths)
successful = 0
missing_labels = 0
bad_images = 0

print(f"Found {total_images} images\n")

for img_path in image_paths:
    print(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"[MISSING LABEL] {base_name}.txt")
        missing_labels += 1
        continue

    result = draw_yolo_boxes(img_path, label_path)

    if result is None:
        bad_images += 1
        continue

    successful += 1

    ext = os.path.splitext(img_path)[1]
    output_path = os.path.join(OUTPUT_DIR, base_name + ext)
    cv2.imwrite(output_path, result)
    
    """
    if verbose:
        cv2.imshow("YOLO Label Check", result)
        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break

        cv2.destroyAllWindows()
    """

print("\n===== SUMMARY =====")
print(f"Total Images:        {total_images}")
print(f"Successful Draws:    {successful}")
print(f"Missing Labels:      {missing_labels}")
print(f"Bad Images:          {bad_images}")