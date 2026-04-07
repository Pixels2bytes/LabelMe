from collections import defaultdict
import os
import json
from typing import List, Tuple
import cv2
from glob import glob
import numpy as np
from datetime import datetime, time
from dotenv import load_dotenv
from google.genai import types
from google.genai.errors import ServerError
from pathlib import Path
from utils import load_service_account
from prompts import img_ground_truth_prompt
from schemas import test_llm_schema
from verify_gt import scale_boxes, scale_non_square, draw_gtboxes


gt_dir = "resources/ground_truth"
trials_name = f"llm_images test *" # Main name and contains each test trial run
gt_images_dir = f"{gt_dir}/gt_images"
llm_map_path = f"{gt_dir}/size_reference_map test 10.json"
gt_map_path = f"{gt_images_dir}/gt_mapping.json" # Ground truth mapping with scaled boxes for all image size variants
pattern = os.path.join(gt_dir, "size_reference_map*.json") # Grab all test runs for averaging
master_map_path = os.path.join(gt_dir, "final_scaling_map.json")
llm_dir = f"{gt_dir}/llm_images"

# Create necessary folders
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(llm_dir, exist_ok=True)
os.makedirs(gt_images_dir, exist_ok=True)

MODEL_ID = "gemini-robotics-er-1.5-preview"
LABEL_CLASSES = {"person"}
PROMPT = img_ground_truth_prompt(list(LABEL_CLASSES))
load_dotenv(dotenv_path="utils/.env")
PROJECT_NAME = os.getenv("PROJECT_NAME")
LOCATION = os.getenv("LOCATION")

USE_VERTEX = False

if USE_VERTEX:
    load_service_account(PROJECT_NAME, os.getenv("VERTEX_SERVICE_ACCT"))

import vertexai
from google import genai

if USE_VERTEX:
    # Load Vertex AI environment
    vertexai.init(project=PROJECT_NAME, location=LOCATION)

    client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION, api_key=os.getenv("VERTEX_GEMINI_API_KEY"))
else:
    # Initialize the GenAI client and specify the model
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def image_to_bytes(img_file: Path) -> bytes:
    frame = cv2.imread(str(img_file))
    if frame is None:
        print(f"Error reading {img_file.name}, skipping")
        return None

    # Encode to bytes for detection
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print(f"Error encoding {img_file.name}, skipping")
        return None

    frame_bytes = buffer.tobytes()
    return frame_bytes, frame


def gem_response(frame_bytes, model_id, prompt, max_retries=2):
    """Detect labels in a frame using Gemini Robotics model with retry logic."""
    retry_count = 0
    base_delay = 2  # Start with 2 seconds delay

    while retry_count <= max_retries:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(
                        data=frame_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent detection
                    response_mime_type="application/json",
                    response_schema=test_llm_schema(),
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            print(f":::::::::::::::\n{response.text}\n")

            if not response.text:
                print("Warning: Empty response")
                return None, response

            try:
                detections = json.loads(response.text)
                return detections, response

            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")

                retry_count += 1
                if retry_count > max_retries:
                    print("Max retries reached (JSON issue). Skipping.")
                    return None, response

                delay = base_delay * (2 ** (retry_count - 1))
                print(f"Retrying JSON parse in {delay}s...")
                time.sleep(delay)

        except ServerError as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Skipping.")
                return None, None

            delay = base_delay * (2 ** (retry_count - 1))
            print(f"API error. Retrying in {delay}s...")
            time.sleep(delay)

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None, None

    return None, None


def process_llm_images(images_folder, output_folder):
    """Process all images in the folder and convert them to bytes."""
    if not images_folder:
        print(f"Error: Images folder {images_folder} does not exist")
        return

    # Convert to Path object
    images_path = Path(images_folder)

    # Collect image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {images_folder}")
        return
    
    for idx, img_file in enumerate(image_files, 1):

        # JSON check to skip already processed images
        json_path = f"{output_folder}/{img_file.stem}.json"
        if os.path.exists(json_path):
            print(f"\nSkipping {img_file.name} (JSON already exists)")
            continue

        print(f"\nProcessing image {idx}/{len(image_files)}: {img_file.name}")

        result = image_to_bytes(img_file)
        if result is None:
            continue

        frame_bytes, frame = result

        # Detect person from Gemini model
        detections, response = gem_response(frame_bytes, MODEL_ID, PROMPT)

        if detections is None:
            print(f"Warning: No valid response for {img_file.name}")
            continue

        # Save raw JSON response directly
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

    print("\nAll images processed")


def load_gt_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_gt(gt_map_path:str):
    with open(gt_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Original size (square image)
    width = data["original_size"]
    height = data["original_size"]

    # Original GT bounding box
    gt_box = data["gt_boxes"][0]

    # Extract variant names without .jpg
    variants = [
        os.path.splitext(name)[0]
        for name in data["variants"].keys()
    ]

    return gt_box, width, height, variants


def extract_llm(gt_dir:str, trials_name:str):
    """
    Collect all bounding boxes from all trials, mapped by variant name.

    Args:
        gt_dir (str): path containing folders like "llm_images test 1", "llm_images test 2", ...
        trials_name (str): pattern for trial folder names

    Returns:
        dict: { "320x320": [(x1,y1,x2,y2), ...], "640x640": [...], ... }
    """
    all_boxes = defaultdict(list)

    # Find all trial folders
    trial_folders = sorted(glob(os.path.join(gt_dir, trials_name)))

    for folder in trial_folders:
        # Get all JSON files in this trial
        json_files = glob(os.path.join(folder, "*.json"))

        for json_path in json_files:
            # Extract variant name from file name (heightxwidth)
            variant_name = os.path.splitext(os.path.basename(json_path))[0]

            # Load JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract first detection bbox if it exists
            if "detections" in data and data["detections"]:
                bbox = tuple(data["detections"][0]["bbox"])
                all_boxes[variant_name].append(bbox)

    return dict(all_boxes)


def get_dimensions(gt_box: List[int], bb_box: List[int], orig_image_size: int = 320, tolerance: float = 0.05) -> Tuple[int, int]:
    """
    Compute the width and height of a resized image given:
    - gt_box: [x1g, y1g, x2g, y2g] from ground truth
    - bb_box: [x1p, y1p, x2p, y2p] from model prediction
    - orig_image_size: width/height of original square image
    - tolerance: since AI predictions can be noisy, we apply a tolerance threshold to the scaling factors

    Returns:
        (width, height) of the new image as integers
    """
    x1g, y1g, x2g, y2g = gt_box
    x1p, y1p, x2p, y2p = bb_box

    # Compute raw scale factors
    sx = (x2p - x1p) / (x2g - x1g) # Scale for width
    sy = (y2p - y1p) / (y2g - y1g) # Scale for height

    # Apply threshold clipping to avoid extreme scaling due to box noise
    sx = max(min(sx, 1 + tolerance), 1 - tolerance)
    sy = max(min(sy, 1 + tolerance), 1 - tolerance)

    # Compute new image dimensions
    widthp = round(orig_image_size * sx)
    heightp = round(orig_image_size * sy)

    return widthp, heightp


def test_llm_process():
    # process_llm_images(gt_images_dir, llm_dir)

    # Find Width and Height of the new image size variants of the LLM
    variants_json = {}
    gt_box, widthg, heightg, variants = extract_gt(gt_map_path)
    orig_image_size = widthg # square images
    all_boxes = extract_llm(gt_dir, trials_name)
    # Compute averaged boxes per variant
    avg_boxes = {
        variant: tuple(
            sum(coord[i] for coord in boxes) // len(boxes)  # integer division
            for i in range(4)
        )
        for variant, boxes in all_boxes.items()
    }
    for variant in variants:
        bb_box = avg_boxes[variant] # Get the averaged predicted box for this variant
        widthp, heightp = get_dimensions(gt_box, bb_box, orig_image_size, tolerance = 0.05)
        # Scale coordinates
        if widthp == heightp:
            scaled_boxes = [scale_boxes(bb_box, orig_image_size, widthp)]
        else:
            scaled_boxes = [scale_non_square(bb_box, orig_image_size, widthp, heightp)]

        variants_json[f"{variant}.jpg"] = {
            "height": heightp,
            "width": widthp,
            "scaled_boxes": scaled_boxes,
            "averaged_boxes": [bb_box]
        }

        # Final JSON
        save_llm_map = {
            "original_size": orig_image_size,
            "gt_boxes": [gt_box],
            "variants": variants_json
        }

        # Save to file
        os.makedirs(os.path.dirname(master_map_path), exist_ok=True)
        with open(master_map_path, "w", encoding="utf-8") as f:
            json.dump(save_llm_map, f, indent=4)

        print(f"Final JSON saved to {master_map_path}")
        
        draw_gtboxes(master_map_path, llm_dir, file_title="llm_verify_box")
    
    return


def main():
    test_llm_process()


if __name__ == "__main__":
    main()