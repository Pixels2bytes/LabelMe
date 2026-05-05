from collections import defaultdict
import os
import json
import re
import shutil
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
from prompts import img_ground_truth_prompt, img_multi_prompt
from schemas import test_llm_schema
from verify_gt import image_test_process, scale_boxes, scale_non_square, draw_gtboxes
from resize_dataset import resize_images_process


# Pricing for Gemini Robotics ER 1.5 Preview (adjust based on actual pricing)
# Estimated pricing per 1M tokens
INPUT_PRICE_PER_1M = 0.0  # Update with actual pricing
OUTPUT_PRICE_PER_1M = 0.0  # Update with actual pricing

MODEL_ID = "gemini-robotics-er-1.6-preview"
# PROMPT = img_ground_truth_prompt(list(LABEL_CLASSES))
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

def image_to_bytes(img_file: Path) -> bytes:
    frame = cv2.imread(str(img_file))
    frame_height, frame_width = frame.shape[:2] # height = 0 index, width = 1 index
    if frame is None:
        print(f"Error reading {img_file.name}, skipping")
        return None

    # Encode to bytes for detection
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        print(f"Error encoding {img_file.name}, skipping")
        return None

    frame_bytes = buffer.tobytes()
    return frame_bytes, frame, frame_height, frame_width


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
                )
            )
            """
            config={'service_tier': 'SERVICE_TIER_FLEX', 'types': types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent detection
                    response_mime_type="application/json",
                    response_schema=test_llm_schema(),
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )  
                    },
            )
            """

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


def process_llm_images(images_folder, output_folder, norm_range=1000, LABEL_CLASSES={"person"}, gt:bool=False):
    """Process all images in the folder and convert them to bytes."""
    if not images_folder:
        print(f"Error: Images folder {images_folder} does not exist")
        return

    # Convert to Path object
    images_path = Path(images_folder)

    # Collect image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP", ".avif", ".AVIF", ".webp", ".WEBP", ".tif", ".tiff", ".TIF", ".TIFF", ".svg", ".SVG", ".svgz", ".SVGZ"]
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {images_folder}")
        return
    
    for idx, img_file in enumerate(image_files, 1):

        # JSON check to skip already processed images
        json_path = f"{output_folder}/{img_file.stem}.json"
        if os.path.exists(json_path):
            #print(f"\nSkipping {img_file.name} (JSON already exists)")
            continue

        print(f"\nProcessing image {idx}/{len(image_files)}: {img_file.name}")

        result = image_to_bytes(img_file)
        if result is None:
            continue

        frame_bytes, frame, frame_height, frame_width = result

        # Detect person from Gemini model
        if gt:
            PROMPT = img_ground_truth_prompt(list(LABEL_CLASSES), norm_range)
        else:
            PROMPT = img_multi_prompt(list(LABEL_CLASSES), norm_range)
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


def get_dimensions(gt_box: List[int], bb_box: List[int], orig_image_size: int = 320, tolerance: float = 0.05, pixel_dims: bool = False) -> Tuple[int, int]:
    """
    Compute the width and height of a resized image given:
    - gt_box: [y1g, x1g, y2g, x2g] from ground truth
    - bb_box: [y1p, x1p, y2p, x2p] from model prediction
    - orig_image_size: width/height of original square image
    - tolerance: since AI predictions can be noisy, we apply a tolerance threshold to the scaling factors

    Returns:
        (width, height) of the new image as integers
    """
    y1g, x1g, x2g, y2g = gt_box
    y1p, x1p, y2p, x2p = bb_box
    
    # Convert to pixel dimensions (if they are normalized, this step would be different)

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


def norm_scale_boxes(bb_box:list, image_height:int, image_width:int, norm_range:int=1000):
    """Draw bounding boxes on the image frame based on normalized coordinates (0 to norm_range)"""
    scaled = []
    for box in bb_box:
        try:
            if not box or len(box) != 4:
                continue

            # Convert normalized coordinates (0-normal_range) to pixel coordinates
            ymin = int(box[0] * image_height / norm_range)
            xmin = int(box[1] * image_width / norm_range)
            ymax = int(box[2] * image_height / norm_range)
            xmax = int(box[3] * image_width / norm_range)

            # Ensure coordinates are within image bounds
            ymin = max(0, min(ymin, image_height))
            xmin = max(0, min(xmin, image_width))
            ymax = max(0, min(ymax, image_height))
            xmax = max(0, min(xmax, image_width))
            scaled.append([ymin, xmin, ymax, xmax])

        except Exception as e:
            print(f"Error drawing bbox {box}: {e}")
            continue
    return scaled


def convert_to_yolo(bb_box:list, image_height:int, image_width:int, norm_range:int=1000):
    yolo_boxes = []
    for box in bb_box:
        try:
            if not box or len(box) != 4:
                continue
            
            y_min, x_min, y_max, x_max = box

            # normalize to 0–1
            x_min /= norm_range
            x_max /= norm_range
            y_min /= norm_range
            y_max /= norm_range

            # YOLO conversion
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            yolo_boxes.append([x_center, y_center, width, height])

        except Exception as e:
            print(f"Error converting to YOLO format {box}: {e}")
            continue
    return yolo_boxes


def process_llm_map(map_path: str, gt_map_path: str, gt_dir:str, trials_name: str, pixel_dims: bool = False, norm_range: int = 1000, tolerance: float = 0.02):
    # Find Width and Height of the new image size variants of the LLM
    variants_json = {}
    gt_box, widthg, heightg, variants = extract_gt(gt_map_path)

    orig_image_size = widthg # square images
    all_boxes = extract_llm(gt_dir, trials_name)
    # Compute averaged boxes per variant
    avg_boxes = {
        variant: tuple(
            round(sum(coord[i] for coord in boxes) / len(boxes))  # divide normally, then round
            for i in range(4)
        )
        for variant, boxes in all_boxes.items()
    }
    for variant in variants:
        # Extract gt size from filename like "320x320.jpg" to get 320 from square
        gt_size = int(re.search(r"(\d+)x\1", variant).group(1))

        # Get height and width from variant name (e.g., "320x400" to get height=320, width=400)
        match = re.search(r"(\d+)x(\d+)", variant)
        gt_height = int(match.group(1)) # Extract height (first number)
        gt_width = int(match.group(2)) # Extract width (second number)
        print(f"Processing variant {variant} with GT size {gt_size} and GT dimensions ({gt_width}x{gt_height})")
        bb_box = avg_boxes[variant] # Get the averaged predicted box for this variant
        
        # Scale coordinates
        if pixel_dims:
            widthp, heightp = get_dimensions(gt_box, bb_box, gt_size, tolerance = 0.02)
            if widthp == heightp:
                scaled_boxes = [round(scale_boxes([bb_box], widthp, gt_size)[0][i]) for i in range(4)]
                norm_range = 1 # Set norm_range to 1 to indicate that these are now pixel dimensions and should not be normalized further
            else:
                scaled_boxes = [round(coord) for coord in scale_non_square([bb_box], gt_size, widthp, heightp)[0]]
                norm_range = 1 # Set norm_range to 1 to indicate that these are now pixel dimensions and should not be normalized further
        else:
            # Normalized coordinates to pixel dimensions for the new image size
            scaled_boxes = norm_scale_boxes([bb_box], gt_height, gt_width, norm_range)[0]
        
        yolo_boxes = convert_to_yolo([bb_box], gt_height, gt_width, norm_range)[0]

        variants_json[f"{variant}.jpg"] = {
            "size": gt_size,
            "height": gt_height,
            "width": gt_width,
            "norm_range": norm_range,
            "yolo_boxes": [yolo_boxes],
            "scaled_boxes": [scaled_boxes],
            "averaged_boxes": [bb_box]
        }

        save_llm_map = {
            "original_size": orig_image_size,
            "gt_boxes": [gt_box],
            "variants": variants_json
        }

    # Save to file
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(save_llm_map, f, indent=4)

    print(f"LLM Map saved to {map_path}")
        
    return map_path


def copy_images(images_dir: str, folder_path: str, main_dir:str, add_extended:bool=False):
    if add_extended:
        horiz_dir = f"{main_dir}/horizontal"

        # Copy horizontal images
        for img_file in os.listdir(horiz_dir):
            src_path = os.path.join(horiz_dir, img_file)
            filename, ext = os.path.splitext(os.path.basename(src_path))

            if not os.path.isfile(src_path):
                continue

            if not filename.endswith("_horiz"):
                continue

            dst_path = os.path.join(images_dir, img_file)

            if os.path.exists(dst_path):
                print(f"Skipping {img_file} (already exists in images_dir)")
                continue

            shutil.copy(src_path, dst_path)
        else:
            print("Only copy images from the directory")

    # Copy original images from main_dir
    for img_file in os.listdir(main_dir):
        src_path = os.path.join(main_dir, img_file)

        if not os.path.isfile(src_path):
            continue

        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp", ".avif", ".svg", ".svgz")):
            continue

        dst_path = os.path.join(images_dir, img_file)

        if os.path.exists(dst_path):
            print(f"Skipping {img_file} (already exists in images_dir)")
            continue

        shutil.copy(src_path, dst_path)


def auto_llm_process(pixel_dims: bool = False, norm_range: int = 1000, LABEL_CLASSES = {"person"}, gt:bool = False):
    main_dir = "datasets/neo_weapons"
    images_dir = f"{main_dir}/images" # datasets/neo_weapons/images
    add_extended = True # True = Add horizontal and original images, False = Only use current images in folder
    folder_path= f"{main_dir}/image_annotations"
    llm_dir = f"{main_dir}/llm_images"
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(llm_dir, exist_ok=True)
    """
    trials_name = f"image_annotations *" # Main name and contains each test trial run
    gt_images_dir = f"{gt_dir}"
    llm_map_path = f"{gt_dir}/size_reference_map test 10.json"
    gt_map_path = f"{gt_images_dir}/gt_mapping.json" # Ground truth mapping with scaled boxes for all image size variants
    pattern = os.path.join(gt_dir, "size_reference_map*.json") # Grab all test runs for averaging
    master_map_path = os.path.join(gt_dir, "llm_scaling_map.json")
    images_dir = f"{gt_dir}"
    llm_dir = f"{gt_dir}/llm_images"

    # Create necessary folders
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(llm_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_images_dir, exist_ok=True)
    """

    # Copy images to new folder to process (dataset/images)
    #copy_images(images_dir, folder_path, main_dir, add_extended)

    process_llm_images(images_dir, folder_path, norm_range, LABEL_CLASSES)

    return main_dir, images_dir, folder_path, norm_range
2
def process_dataset_ready(main_dir, images_dir, folder_path, norm_range, LABEL_CLASSES, verbsoe:bool=False):
    # Get all JSON files from folder_path and create YOLO annotation files
    
    # bb_box = [[y1, x1, y2, x2], [y1, x1, y2, x2], ...] # List of bounding boxes from json file]
    #convert_to_yolo(bb_box, image_height, image_width, norm_range) # y1, x1, y2, x2
    #map_path = process_llm_map(master_map_path, pixel_dims=pixel_dims, norm_range=norm_range, tolerance=0.02)
    #draw_gtboxes(master_map_path, images_dir, (0, 0, 255), llm_dir, file_title="llm_verify_box", dataset_title="verify_box")
    return


def main():
    LABEL_CLASSES = {"person", "hand", "gun", "smartphone", "knife"}
    main_dir, images_dir, folder_path, norm_range = auto_llm_process(LABEL_CLASSES=LABEL_CLASSES)
    process_dataset_ready(main_dir, images_dir, folder_path, norm_range, LABEL_CLASSES)


if __name__ == "__main__":
    main()