import csv
import cv2
import re
import os
from collections import defaultdict
from main_utils import load_config


def parse_index(frame_name: str) -> int:
    frame_name = frame_name.replace(",", "").strip()
    return int(frame_name.split("_")[1].split(".")[0])


def parse_coords(coord_str: str, classes: dict = None):
    detections = []

    if not coord_str:
        return detections

    # Remove surrounding brackets if present
    coord_str = coord_str.strip().lstrip("[").rstrip("]")

    # Split on commas NOT inside parentheses
    parts = re.split(r",(?![^()]*\))", coord_str)

    pattern = re.compile(r"(.+?)\s+\d+\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")

    for part in parts:
        part = part.strip().strip("'").strip('"')

        match = pattern.search(part)
        if not match:
            continue

        label, x1, y1, x2, y2 = match.groups()
        label = label.strip()

        if classes and label not in classes:
            continue

        detections.append({
            "label": label,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": 1.0
        })

    return detections


def load_detects(csv_path: str, class_colors: dict=None):
    frame_detections = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_idx = parse_index(row["Frame"])
            detections = parse_coords(row["Coordinates"], class_colors)
            frame_detections[frame_idx].extend(detections)

    return frame_detections


def draw_detections(frame, detections, class_colors=None):
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf = det["confidence"]

        if label in ["gun", "smartphone", "person"]:
            color = (0, 255, 0)
            if class_colors and label in class_colors:
                color = class_colors[label]

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


def create_video(video_path: str, csv_path: str, output_path: str, class_colors=None):
    detections_by_frame = load_detects(csv_path, class_colors)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        draw_detections(
            frame,
            detections_by_frame.get(frame_idx, []),
            class_colors
        )

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def numeric_sort(filename: str) -> int:
    name = os.path.splitext(filename)[0]
    return int(name.split("_")[-1])


def load_yolo_txt(label_path: str, img_w: int, img_h: int, class_map: dict):
    detections = []

    if not os.path.exists(label_path):
        return detections

    with open(label_path, "r") as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.split())

            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            detections.append({
                "label": class_map[int(cls)],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": 1.0
            })

    return detections


def create_video_from_dataset(
    image_dir: str,
    label_dir: str,
    output_path: str,
    class_map: dict,
    class_colors=None,
    fps: int = 10
):
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))],
        key=numeric_sort
    )

    first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width = first_frame.shape[:2]

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_idx = 0

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        frame = cv2.imread(img_path)
        detections = load_yolo_txt(label_path, width, height, class_map)

        draw_detections(frame, detections, class_colors)

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        out.write(frame)
        frame_idx += 1

    out.release()


def extract_video_name_from_csv(csv_name: str) -> str:
    name = os.path.splitext(csv_name)[0]
    return name.replace("yoloweapons_detection_model_", "")


def create_video_from_csv_frames(frames_dir: str,csv_path: str,output_path: str,class_colors=None,fps: int = 10,verbose: bool = False):
    frame_detections = load_detects(csv_path, class_colors)
    if verbose:
        print(frame_detections)

    image_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))],
        key=numeric_sort
    )

    first_frame = cv2.imread(os.path.join(frames_dir, image_files[0]))
    height, width = first_frame.shape[:2]

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_idx = 0

    for img_name in image_files:
        frame = cv2.imread(os.path.join(frames_dir, img_name))

        draw_detections(
            frame,
            frame_detections.get(frame_idx, []),
            class_colors
        )

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        out.write(frame)
        frame_idx += 1

    out.release()


def video_annon_process():
    config = load_config("config.json")

    video_stream = config.get("VIDEO_STREAM")
    csv_path = config.get("VIDEO_CSV")
    output_path = "annotated_output.mp4"
    verbose = config.get("DEBUG", False)

    CLASS_COLORS = {
        "person": (0, 255, 0),
        "smartphone": (0, 0, 255),
        "gun": (255, 0, 0),
    }

    create_video(video_stream, csv_path, output_path, CLASS_COLORS)
    if verbose:
        print("Annotated video saved to", output_path)


def dataset_annon_process():
    config = load_config("config.json")

    img_dir = config.get("IMAGE_DIR")
    label_dir = config.get("LABEL_DIR")
    output_path = "dataset_annotated.mp4"
    verbose = config.get("DEBUG", False)

    CLASS_MAP = {
        0: "person",
        1: "gun",
        2: "smartphone"
    }

    CLASS_COLORS = {
        "person": (0, 255, 0),
        "smartphone": (0, 0, 255),
        "gun": (255, 0, 0),
    }

    create_video_from_dataset(
        img_dir,
        label_dir,
        output_path,
        CLASS_MAP,
        CLASS_COLORS
    )
    if verbose:
        print("Dataset video saved to", output_path)


def dataset_csv_annon_process():
    config = load_config("config.json")

    model_name = "yoloweapons_detection_model"

    csv_root = config.get("DATASET_CSV", f"resources/{model_name}")
    frames_root = config.get("IMAGE_DIR", "resources/frames")
    output_root = "output/annotations/videos"
    verbose = config.get("DEBUG", False)

    os.makedirs(output_root, exist_ok=True)

    CLASS_COLORS = {
        "person": (0, 255, 0),
        "smartphone": (0, 0, 255),
        "gun": (255, 0, 0),
    }

    for csv_file in os.listdir(csv_root):
        if not csv_file.endswith(".csv"):
            continue

        video_name = extract_video_name_from_csv(csv_file)

        frames_dir = os.path.join(
            frames_root,
            video_name,
            "original"
        )

        print("Processing", frames_dir)

        if not os.path.exists(frames_dir):
            continue

        output_path = os.path.join(
            output_root,
            f"{video_name}_annotation.mp4"
        )

        create_video_from_csv_frames(
            frames_dir,
            os.path.join(csv_root, csv_file),
            output_path,
            CLASS_COLORS
        )
        if verbose:
            print("Annotated dataset video saved to", output_path)


def main():
    #video_annon_process() # For a video stream with yolo/coco .txt annotations
    #dataset_annon_process() # For images with yolo/coco .txt annotations
    dataset_csv_annon_process() # For images with CSV annotations


if __name__ == "__main__":
    main()