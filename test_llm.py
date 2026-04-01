import os
import json
import cv2
import glob
import numpy as np
from datetime import datetime, time
from dotenv import load_dotenv
from google.genai import types
from google.genai.errors import ServerError
from pathlib import Path
from utils import load_service_account
from prompts import img_ground_truth_prompt
from schemas import test_llm_schema


gt_dir = "resources/ground_truth"
llm_dir = f"{gt_dir}/llm_images test 10"
gt_images_dir = f"{gt_dir}/gt_images"
llm_map_path = f"{gt_dir}/size_reference_map test 10.json"
gt_map_path = f"{gt_images_dir}/gt_mapping.json"
pattern = os.path.join(gt_dir, "size_reference_map*.json") # Grab all test runs for averaging
master_map_path = os.path.join(gt_dir, "final_scaling_map.json")

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


def compute_transform(gt_box, pred_box):
    x1g, y1g, x2g, y2g = gt_box
    x1p, y1p, x2p, y2p = pred_box

    sx = (x2p - x1p) / (x2g - x1g) # Scale X between GT and Pred
    sy = (y2p - y1p) / (y2g - y1g) # Scale Y between GT and Pred

    dx = x1p - (x1g * sx) # Shift X after scaling
    dy = y1p - (y1g * sy) # Shift Y after scaling

    return sx, sy, dx, dy


def compare_transforms(gt_map_path, llm_dir, llm_map_path):
    gt_data = load_gt_mapping(gt_map_path)
    results = {}

    for filename, data in gt_data["variants"].items():
        gt_box = data["scaled_boxes"][0]  # only one box

        json_name = filename.replace(".jpg", ".json")
        llm_path = os.path.join(llm_dir, json_name)

        if not os.path.exists(llm_path):
            print(f"Missing LLM output for {filename}")
            continue

        with open(llm_path, "r", encoding="utf-8") as f:
            llm_data = json.load(f)

        detections = llm_data.get("detections", [])

        if not detections:
            print(f"No detection for {filename}")
            continue

        pred_box = detections[0]["bbox"]  # assume single detection

        sx, sy, dx, dy = compute_transform(gt_box, pred_box)

        size = data["size"]

        results[size] = {
            "sx": sx,
            "sy": sy,
            "dx": dx,
            "dy": dy
        }

        print(f"\n{size}x{size}")
        print(f"sx={sx:.4f}, sy={sy:.4f}, dx={dx:.2f}, dy={dy:.2f}")

    with open(llm_map_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_all_maps():
    files = glob.glob(pattern)

    if not files:
        print("No mapping files found.")
        return []

    print(f"Found {len(files)} mapping files")

    maps = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            maps.append(json.load(f))

    return maps


def average_maps(maps):
    """
    Averages sx, sy, dx, dy across all test runs per size
    """
    combined = {}

    for m in maps:
        for size, vals in m.items():
            if size not in combined:
                combined[size] = {"sx": [], "sy": [], "dx": [], "dy": []}

            combined[size]["sx"].append(vals["sx"])
            combined[size]["sy"].append(vals["sy"])
            combined[size]["dx"].append(vals["dx"])
            combined[size]["dy"].append(vals["dy"])

    averaged = {}

    for size, vals in combined.items():
        averaged[size] = {
            "sx": float(np.mean(vals["sx"])),
            "sy": float(np.mean(vals["sy"])),
            "dx": float(np.mean(vals["dx"])),
            "dy": float(np.mean(vals["dy"])),
        }

    return averaged


def compute_global_equation(averaged_map):
    """
    Converts sx, sy into kx/size form
    """
    kx_vals = []
    ky_vals = []

    for size, vals in averaged_map.items():
        s = int(size)

        kx_vals.append(vals["sx"] * s)
        ky_vals.append(vals["sy"] * s)

    kx = float(np.mean(kx_vals))
    ky = float(np.mean(ky_vals))

    return kx, ky


def save_final_map(master_map_path, averaged_map, kx, ky):
    output = {
        "per_size": averaged_map,
        "global_equation": {
            "kx": kx,
            "ky": ky,
            "formula": "sx = kx / size, sy = ky / size"
        }
    }

    with open(master_map_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved final calibration → {master_map_path}")


def correct_bbox(pred_box, image_size, calibration_data):
    """
    Applies correction to LLM predicted bounding box

    Uses:
    - per-size if available
    - fallback to global equation
    """

    x1, y1, x2, y2 = pred_box

    # Center + size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    size_str = str(image_size)

    # Use per-size calibration if exists
    if size_str in calibration_data["per_size"]:
        vals = calibration_data["per_size"][size_str]
        sx = vals["sx"]
        sy = vals["sy"]

    else:
        # fallback to equation
        kx = calibration_data["global_equation"]["kx"]
        ky = calibration_data["global_equation"]["ky"]

        sx = kx / image_size
        sy = ky / image_size

    # Correct size
    w_corr = w / sx
    h_corr = h / sy

    # Rebuild box
    new_x1 = cx - w_corr / 2
    new_y1 = cy - h_corr / 2
    new_x2 = cx + w_corr / 2
    new_y2 = cy + h_corr / 2

    return [new_x1, new_y1, new_x2, new_y2]


def create_equation(gt_map_path, llm_map_path, master_map_path):
    """
    Combines:
    - Global equation (kx/ky)
    - Per-size calibration (sx, sy, dx, dy)
    - Variants with corrected boxes for draw_gtboxes
    """

    # Load LLM averaged map
    with open(llm_map_path, "r", encoding="utf-8") as f:
        averaged_map = json.load(f)

    # Compute global kx/ky
    kx = float(sum([v["sx"] * int(size) for size, v in averaged_map.items()]) / len(averaged_map))
    ky = float(sum([v["sy"] * int(size) for size, v in averaged_map.items()]) / len(averaged_map))

    # Load GT mapping to get original boxes
    with open(gt_map_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    variants = {}

    for filename, data in gt_data["variants"].items():
        size = data["size"]
        gt_box = data["scaled_boxes"][0]  # only one box

        # Get per-size calibration if exists
        size_str = str(size)
        if size_str in averaged_map:
            vals = averaged_map[size_str]
            sx, sy, dx, dy = vals["sx"], vals["sy"], vals["dx"], vals["dy"]
        else:
            # fallback to global equation
            sx = kx / size
            sy = ky / size
            dx = 0
            dy = 0

        # Correct the bounding box using the scaling
        x1, y1, x2, y2 = gt_box
        new_x1 = x1 * sx + dx
        new_y1 = y1 * sy + dy
        new_x2 = x2 * sx + dx
        new_y2 = y2 * sy + dy

        variants[filename] = {
            "size": size,
            "scaled_boxes": [[new_x1, new_y1, new_x2, new_y2]]
        }

    final_map = {
        "global_equation": {
            "kx": kx,
            "ky": ky,
            "formula": "sx = kx / size, sy = ky / size"
        },
        "per_size": averaged_map,
        "variants": variants
    }

    with open(master_map_path, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=2)

    print(f"Final verification map saved → {master_map_path}")


def test_llm_process():
    # process_llm_images(gt_images_dir, llm_dir)
    # compare_transforms(gt_map_path, llm_dir, llm_map_path)
    create_equation(gt_map_path, llm_map_path, master_map_path)
    return


def main():
    test_llm_process()


if __name__ == "__main__":
    main()