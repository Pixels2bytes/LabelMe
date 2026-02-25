"""Auto-label videos using Google Gemini Robotics ER 1.5 Preview model for object detection."""

import os
import json
import cv2
import numpy as np
import time
import shutil
import csv
import base64
from datetime import datetime
from dotenv import load_dotenv
from google.genai import types
from google.genai.errors import ServerError
from pathlib import Path
from utils import load_service_account

load_dotenv(dotenv_path="utils/.env")
PROJECT_NAME = os.getenv("PROJECT_NAME")
LOCATION = os.getenv("LOCATION")

USE_VERTEX = False  # Flip to False for API key mode
USE_DITO = False  # Set to True to use DITO model from vertex, False to use Gemini
MODEL_ID = "gemini-robotics-er-1.5-preview"
# MODEL_ID = "gemini-3-pro-preview" #In case preview model is unavailable, can switch to Gemini 1.5 Pro
LABEL_CLASSES = {"gun", "person", "smartphone", "knife", "hand"}

if USE_VERTEX:
    load_service_account(PROJECT_NAME, os.getenv("VERTEX_SERVICE_ACCT"))

import vertexai
from google import genai

if USE_DITO:
    MODEL_ID = "dito-1.0-preview"
    USE_VERTEX = True  # DITO is only available via Vertex AI, so we force this to True if DITO is selected

if USE_VERTEX:
    # Load Vertex AI environment
    vertexai.init(project=PROJECT_NAME, location=LOCATION)

    client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION, api_key=os.getenv("VERTEX_GEMINI_API_KEY"))
else:
    # Load API Environment Keys
    #load_dotenv(dotenv_path="utils/.env")

    # Initialize the GenAI client and specify the model
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Pricing for Gemini Robotics ER 1.5 Preview (adjust based on actual pricing)
# Estimated pricing per 1M tokens
INPUT_PRICE_PER_1M = 0.0  # Update with actual pricing
OUTPUT_PRICE_PER_1M = 0.0  # Update with actual pricing

# Prompt for weapons detection - Optimized for gun and person detection
PROMPT = """
Detect and identify all 'gun', 'person', 'smartphone', 'knife', or 'hand' objects in this image.

Instructions:
1.  Identify all instances by class: "gun", "smartphone", "knife", "hand", or "person".
2.  Provide tight bounding boxes for each detected object.
3.  Assign a confidence score between 0.0 and 1.0 for each detection.

Return ONLY a JSON array in this exact format, no code fencing, no additional text:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "<label>", "status": "<status>", "confidence": <score>}]

JSON Format Details:
* label: Must be "gun", "smartphone", "knife", "hand", or "person".
* Confidence: A float between 0.0 and 1.0 representing detection confidence.
* Coordinates must be integers normalized to 0-1000 range.
"""

PROMPT2 = """
Detect and identify all 'gun', 'person', 'smartphone', 'knife', or 'hand' objects in this image.

Instructions:
1.  Identify all instances by class: "gun", 'person', 'smartphone', 'knife', or 'hand'.
2.  Provide tight bounding boxes for each detected object.
3.  Assign a confidence score between 0.0 and 1.0 for each detection.

Return a CSV row in this exact format, no code fencing, no additional text:
"['label 1 (ymin, xmin, ymax, xmax)', 'label 2 (ymin, xmin, ymax, xmax)', ...]", "['label1', 'label2', ...]", "['confidence1', 'confidence2', ...]"

Example Output:
"['person 1 (0, 670, 179, 1070)', 'gun 2 (16, 700, 32, 729)']", "['person', 'gun']", "['0.95', '0.89']"

Example Output for No Detections:
[], [], []

CSV Format Details:
* label: Must be "gun", "person", "smartphone", "knife", or "hand".
* box_2d: List of four integers (ymin, xmin, ymax, xmax) representing bounding box coordinates.
* confidence: A float between 0.0 and 1.0 representing detection confidence.
* Coordinates must be integers normalized to 0-1000 range.

"""

# Token and cost tracking
class UsageTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_frames = 0

    def add_usage(self, response):
        """Extract and add token usage from response."""
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            self.total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        self.total_frames += 1

    def calculate_cost(self):
        """Calculate total cost based on token usage."""
        input_cost = (self.total_input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        output_cost = (self.total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
        return input_cost + output_cost

    def print_summary(self):
        """Print usage summary."""
        total_cost = self.calculate_cost()
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Total Input Tokens: {self.total_input_tokens:,}")
        print(f"Total Output Tokens: {self.total_output_tokens:,}")
        print(f"Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}")
        if INPUT_PRICE_PER_1M > 0 or OUTPUT_PRICE_PER_1M > 0:
            print(f"Estimated Cost: ${total_cost:.4f}")
        else:
            print("Cost: Not available (update pricing in script)")
        print("=" * 60)


tracker = UsageTracker()


def gem_detect_weapons(frame_bytes, max_retries=2):
    """Detect weapons in a frame using Gemini Robotics model with retry logic."""
    retry_count = 0
    base_delay = 2  # Start with 2 seconds delay

    while retry_count <= max_retries:
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Part.from_bytes(
                        data=frame_bytes,
                        mime_type="image/jpeg",
                    ),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent detection
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            # Track usage
            tracker.add_usage(response)

            # Parse JSON response
            try:
                # Check if response has text
                if not response.text:
                    print(f"Warning: Empty response from API")
                    return []

                # Clean up response text - remove code fences if present
                response_text = response.text.strip()

                # If response is just text saying no weapons, return empty array
                if not response_text.startswith("[") and not response_text.startswith(
                    "{"
                ):
                    print(f"Raw response: {response_text[:200]}...")
                    return []

                if response_text.startswith("```"):
                    # Remove markdown code fencing
                    lines = response_text.split("\n")
                    response_text = (
                        "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                    )

                detections = json.loads(response_text)

                # Validate the structure
                if not isinstance(detections, list):
                    print(f"Warning: Response is not a list")
                    return []

                return detections
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(
                    f"Raw response: {response.text[:200] if response.text else 'None'}..."
                )
                return []

        except ServerError as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\nMax retries ({max_retries}) reached. Skipping frame.")
                return []

            # Exponential backoff: 2, 4, 8, 16, 32 seconds
            delay = base_delay * (2 ** (retry_count - 1))
            print(
                f"\nAPI timeout/error (attempt {retry_count}/{max_retries}). Retrying in {delay}s..."
            )
            time.sleep(delay)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            return []

    return []


def dito_detect_weapons(frame_bytes, max_retries=5):
    """Detect weapons in a frame using DITO model from Vertex AI with retry logic."""
    retry_count = 0
    base_delay = 2  # Start with 2 seconds delay

    while retry_count <= max_retries:
        try:
            response = client.models.predict(
                model=MODEL_ID,
                instances_list = [
                    {
                        "image_jpeg_bytes_inputs": {
                            "b64": base64.b64encode(frame_bytes).decode("utf-8")
                        }
                    }
                ],
            )
            tracker.total_frames += 1
            if not response or not hasattr(response, "predictions"):
                print("Warning: Empty response from DITO")
                return []

            predictions = response.predictions
            

            if not predictions:
                return []

            detections = []

            # DITO returns detections in standard OD format
            pred = predictions[0]

            boxes = pred.get("bboxes", [])
            scores = pred.get("scores", [])
            classes = pred.get("classes", [])

            # Map DITO class names to required labels
            valid_labels = LABEL_CLASSES

            for box, score, cls in zip(boxes, scores, classes):

                label = str(cls).lower()

                if label not in valid_labels:
                    continue

                # DITO boxes are normalized 0-1 → convert to 0-1000
                ymin = int(box[0] * 1000)
                xmin = int(box[1] * 1000)
                ymax = int(box[2] * 1000)
                xmax = int(box[3] * 1000)

                detections.append(
                    {
                        "box_2d": [ymin, xmin, ymax, xmax],
                        "label": label,
                        "status": "detected",
                        "confidence": float(score),
                    }
                )

            return detections

        except ServerError as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\nMax retries ({max_retries}) reached. Skipping frame.")
                return []

            delay = base_delay * (2 ** (retry_count - 1))
            print(
                f"\nDITO API timeout/error (attempt {retry_count}/{max_retries}). Retrying in {delay}s..."
            )
            time.sleep(delay)

        except Exception as e:
            print(f"\nUnexpected DITO error: {e}")
            return []

    return []


def write_detections_to_csv(writer, frame_name, detections, image_width, image_height):
    """
    Write each detection as a separate row in the CSV:
    Frame, Label, Coordinates, Box Width, Box Height, Image Width, Image Height, Confidence
    """
    for det in detections:
        box = det.get("box_2d", [])
        label = det.get("label", "")
        confidence = det.get("confidence", 0.0)

        if not box or len(box) != 4 or not label:
            continue

        # Calculate box width and height
        box_width = box[3] - box[1]
        box_height = box[2] - box[0]

        # Format coordinates as a string tuple "(ymin, xmin, ymax, xmax)"
        coords_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"

        # Write row
        writer.writerow([frame_name, label, coords_str, f"{box_width}", f"{box_height}", f"{image_width}", f"{image_height}", f"{confidence:.2f}"])


def draw_weapon_bboxes(frame, detections):
    """Draw bounding boxes on the frame with label and confidence."""
    height, width = frame.shape[:2]

    # Color coding by label type
    label_colors = {
        "gun": (0, 0, 255),  # Red for guns
        "person": (0, 255, 0),  # Green for persons
        "smartphone": (255, 255, 0),  # Cyan for smartphones
        "knife": (255, 0, 0),  # Blue for knives
        "hand": (255, 0, 255),  # Purple for hands
    }

    # Draw each bounding box
    for idx, detection in enumerate(detections):
        try:
            box = detection.get("box_2d")
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)

            if not box or len(box) != 4:
                continue

            # Convert normalized coordinates (0-1000) to pixel coordinates
            ymin = int(box[0] * height / 1000)
            xmin = int(box[1] * width / 1000)
            ymax = int(box[2] * height / 1000)
            xmax = int(box[3] * width / 1000)

            # Ensure coordinates are within image bounds
            ymin = max(0, min(ymin, height))
            xmin = max(0, min(xmin, width))
            ymax = max(0, min(ymax, height))
            xmax = max(0, min(xmax, width))

            # Get color based on label
            color = label_colors.get(label.lower(), (255, 0, 0))

            # Draw rectangle with thick line for visibility
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)

            # Create label text with confidence
            label_text = f"{label} {confidence:.2f}"

            # Draw label background
            label_size, baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_ymin = max(ymin, label_size[1] + 10)

            cv2.rectangle(
                frame,
                (xmin, label_ymin - label_size[1] - 10),
                (xmin + label_size[0] + 5, label_ymin + baseline - 10),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (xmin + 2, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        except Exception as e:
            print(f"Error drawing bbox {idx}: {e}")
            continue

    return frame


def process_images_folder(dataset_images_folder, output_folder, output_images_folder):
    """Process all images in a folder and save annotations + CSV."""
    annotations_folder = f"{output_folder}/annotations"
    json_folder = f"{output_folder}/json"
    #json_path = f"{json_folder}/detections.json"

    # Create necessary folders
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    if not dataset_images_folder:
        print(f"Error: Images folder {dataset_images_folder} does not exist")
        return

    # Convert to Path object
    images_path = Path(dataset_images_folder)

    # Collect image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]

    # Sort files based on numbers in filename to ensure correct order (e.g., frame1.jpg, frame2.jpg, ..., frame10.jpg)
    import re
    def numerical_sort_key(path_obj):
        numbers = re.findall(r'\d+', path_obj.stem)
        return int(numbers[-1]) if numbers else 0

    image_files = sorted(image_files, key=numerical_sort_key)

    if not image_files:
        print(f"No images found in {dataset_images_folder}")
        return

    print(f"Found {len(image_files)} images to process\n")

    csv_path = f"{output_folder}/image_annotations.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Label", "Coordinates", "Box_Width", "Box_Height", "Image_Width", "Image_Height", "Confidence"])

        for idx, img_file in enumerate(image_files, 1):

            # JSON check to skip already processed images
            json_path = f"{json_folder}/{img_file.stem}.json"
            if os.path.exists(json_path):
                print(f"\nSkipping {img_file.name} (JSON already exists)")
                continue

            print(f"\nProcessing image {idx}/{len(image_files)}: {img_file.name}")

            # Read image
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"Error reading {img_file.name}, skipping")
                continue

            # Encode to bytes for detection
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print(f"Error encoding {img_file.name}, skipping")
                continue
            frame_bytes = buffer.tobytes()

            if USE_DITO:
                # Detect objects from DITO model
                detections = dito_detect_weapons(frame_bytes)
            else:
                # Detect objects from Gemini models
                detections = gem_detect_weapons(frame_bytes)

            # Convert JSON to CSV format
            write_detections_to_csv(writer, img_file.name, detections, frame.shape[1], frame.shape[0]) # frame.shape[1] = image width, frame.shape[0] = image height

            # Draw bounding boxes
            annotated_frame = draw_weapon_bboxes(frame.copy(), detections)

            # Save annotated image
            annotated_path = f"{annotations_folder}/{img_file.name}"
            cv2.imwrite(annotated_path, annotated_frame)

            # Copy original image to dataset images folder
            shutil.copy(str(img_file), f"{output_images_folder}/{img_file.name}")

            # Save detections to JSON (always save, even if no detections)
            """frames_with_detections = sum(1 for d in detections if d["num_detections"] > 0)
            total_objects_detected = sum(d["num_detections"] for d in detections)"""
            width = frame.shape[1]
            height = frame.shape[0]

            with open(json_path, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "image_name": img_file.name,
                            "image_path": str(img_file),
                            "resolution": f"{width}x{height}",
                            "image_width": width,
                            "image_height": height,
                            "detections": detections,

                            "processed_date": datetime.now().isoformat(),
                        },
                        "detections": detections,
                    },
                    f,
                    indent=2,
                )
            """# Write detections to CSV
            for det in detections:
                box = det.get("box_2d", [])
                label = det.get("label", "")
                confidence = det.get("confidence", 0.0)
                coords = f"{box}" if box else ""
                writer.writerow([img_file.name, label, coords, confidence])"""

    print(f"\nAll images processed. CSV saved to {csv_path}")
    print(f"Annotated images saved to {annotations_folder}")
    print(f"Original images copied to {output_images_folder}")

    #return annotated_frame, csv_path, images_output_folder, annotations_folder


def process_video(video_path, output_folder, frame_skip=1):
    """Process video frame by frame for weapons detection.

    Args:
        video_path: Path to input video
        output_folder: Folder to save output video and results
        frame_skip: Process every Nth frame (1 = process all frames)
    """
    video_path = Path(video_path)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video {video_path} does not exist")
        return

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing video: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")

    # Setup video writer
    output_video_path = output_path / f"detected_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    all_detections = []
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every Nth frame
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue

        processed_count += 1
        print(
            f"\rProcessing frame {frame_count}/{total_frames} (analyzed: {processed_count})",
            end="",
            flush=True,
        )

        # Encode frame to JPEG
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            print(f"\nError encoding frame {frame_count}")
            out.write(frame)
            continue

        frame_bytes = buffer.tobytes()

        # Detect weapons
        detections = gem_detect_weapons(frame_bytes)

        # Store detections (including frames with no detections)
        all_detections.append(
            {
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "num_detections": len(detections),
                "detections": detections,
            }
        )

        # Draw bounding boxes
        frame_with_bbox = draw_weapon_bboxes(frame.copy(), detections)

        # Write frame to output video
        out.write(frame_with_bbox)

    # Release resources
    cap.release()
    out.release()

    print(f"\n\nVideo processing complete!")
    print(f"Output video saved to: {output_video_path}")

    # Save detections to JSON (always save, even if no detections)
    json_path = output_path / f"detections_{video_path.stem}.json"

    frames_with_detections = sum(1 for d in all_detections if d["num_detections"] > 0)
    total_objects_detected = sum(d["num_detections"] for d in all_detections)

    with open(json_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "video_name": video_path.name,
                    "video_path": str(video_path),
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "total_frames": total_frames,
                    "frames_analyzed": processed_count,
                    "frame_skip": frame_skip,
                    "frames_with_detections": frames_with_detections,
                    "total_objects_detected": total_objects_detected,
                    "processed_date": datetime.now().isoformat(),
                },
                "detections": all_detections,
            },
            f,
            indent=2,
        )

    print(f"Detections saved to: {json_path}")
    print(f"Frames with detections: {frames_with_detections}/{processed_count}")
    print(f"Total objects detected: {total_objects_detected}")

    # Print usage summary
    tracker.print_summary()

    return all_detections


def process_videos_folder(videos_folder, output_folder, image_folder, frame_skip=1):
    """Process all videos in a folder."""
    videos_path = Path(videos_folder)

    if not videos_path.exists():
        print(f"Error: Folder {videos_folder} does not exist")
        return

    # Get all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = [
        f for f in videos_path.iterdir() if f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No videos found in {videos_folder}")
        return

    print(f"Found {len(video_files)} videos to process\n")

    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Video {idx}/{len(video_files)}")
        print(f"{'='*60}")
        process_video(video_file, output_folder, frame_skip)

    print(f"\n\nAll videos processed!")


def main():
    """Main function to process videos and images"""
    # Configuration
    OUTPUT_DIR = "model dataset"
    INPUT_FOLDER = "resources/training_files"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    DATASET_IMAGES_FOLDER = f"{INPUT_FOLDER}/images" # Folder containing images to be labelled # testme" #images"
    VIDEOS_FOLDER = f"{INPUT_FOLDER}/videos"  # Folder containing videos to be labelled
    OUTPUT_FOLDER = f"{OUTPUT_DIR}/llm_detections"  # Folder to save results
    OUTPUT_IMAGE_FOLDER = f"{OUTPUT_DIR}/images"
    os.makedirs(DATASET_IMAGES_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    FRAME_SKIP = 1  # Process every frame (set to 2 to process every other frame, etc.)

    # Process all videos in folder
    process_images_folder(DATASET_IMAGES_FOLDER, OUTPUT_FOLDER, OUTPUT_IMAGE_FOLDER)
    #process_videos_folder(VIDEOS_FOLDER, OUTPUT_FOLDER, OUTPUT_IMAGE_FOLDER, FRAME_SKIP)

if __name__ == "__main__":
    main()