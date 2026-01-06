import csv
import cv2
import re
from collections import defaultdict
from main_utils import load_config

def parse_index(frame_name: str) -> int:
    frame_name = frame_name.replace(",", "").strip()
    return int(frame_name.split("_")[1].split(".")[0])


def parse_coords(coord_str: str):
    pattern = re.compile(
        r"(.+?)\s+\d+\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
    )

    detections = []
    for match in pattern.finditer(coord_str):
        label, x1, y1, x2, y2 = match.groups()
        detections.append({
            "label": label.strip(),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": 1.0
        })

    return detections


def load_detects(csv_path: str):
    frame_detections = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_idx = parse_index(row["Frame"])
            detections = parse_coords(row["Coordinates"])

            frame_detections[frame_idx].extend(detections)

    return frame_detections


def draw_detections(frame, detections, class_colors=None):
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf = det["confidence"]

        if label in ["gun", "smartphone", "person"]: # Only draw specified labels
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


def create_video(
    video_path: str,
    csv_path: str,
    output_path: str,
    class_colors=None
):
    detections_by_frame = load_detects(csv_path)

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

        # Draw detections for this frame
        draw_detections(
            frame,
            detections_by_frame.get(frame_idx, []),
            class_colors
        )

        # Optional frame counter
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


def video_annon_process():
    config = load_config("config.json")
    video_stream = config.get["VIDEO_STREAM"]

    CSV_PATH = "annotations.csv"
    OUTPUT_PATH = "annotated_output.mp4"

    CLASS_COLORS = {
        "person": (0, 255, 0),
        "smartphone": (0, 0, 255),
        "gun": (255, 0, 0),
    }

    create_video(video_stream, CSV_PATH, OUTPUT_PATH, CLASS_COLORS)

    return print("Annotated video saved to", OUTPUT_PATH)
    

def main():
    msg = video_annon_process()


if __name__ == "__main__":
    main()
