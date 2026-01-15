"""Auto-label images using Google Gemini Robotics ER 1.5 Preview model for object detection.

note: bounding box coordinates do not translate using draw_boxes.py Needs fix."""


import csv
from http.client import responses
import io
import os
import re
import PIL.Image
import json
import ast
import cv2
import base64
import sys
import math
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from main_utils import load_config

OUTPUT_FOLDER = "model output"
IMAGE_FOLDER = f"{OUTPUT_FOLDER}/images"
CSV_FILE = f"{OUTPUT_FOLDER}/llm_detections.csv"
ANNO_FILE = f"{OUTPUT_FOLDER}/annotations.csv"


# Pricing for Gemini Robotics ER 1.5 Preview (adjust based on actual pricing)
# Estimated pricing per 1M tokens
INPUT_PRICE_PER_1M = 0.0  # Update with actual pricing
OUTPUT_PRICE_PER_1M = 0.0  # Update with actual pricing

# Prompt for weapons detection - Optimized for gun and person detection
PROMPT = """
Detect and identify all 'gun', 'smartphone', and 'person' objects in this image.

Instructions:
1.  Identify all instances by class: "gun", "smartphone", "person".
2.  Provide tight bounding boxes for each detected object.
3.  Assign a confidence score between 0.0 and 1.0 for each detection.

Return a CSV row in this exact format, no code fencing, no additional text:
"['label 1 (xmin, ymin, xmax, ymax)', 'label 2 (xmin, ymin, xmax, ymax)', ...]", "['label1', 'label2', ...]", "['confidence1', 'confidence2', ...]"

Example Output:
"['person 1 (0, 670, 179, 1070)', 'gun 2 (16, 700, 32, 729)']", "['person', 'gun']", "['0.95', '0.89']"

Example Output for No Detections:
[], [], []

CSV Format Details:
* label: Must be "gun", "smartphone", or "person".
* box_2d: List of four integers (xmin, ymin, xmax, ymax) representing bounding box coordinates.
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


def gemini_response(model:str, api_key:str, prompt:str, image_path:str, csv_file:str, verbose:bool = False):
    client = genai.Client(api_key=api_key)
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: {image_path} missing")
        exit()
    
    """orig = PIL.Image.open(image_path)
    copy = orig.copy()

    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG")"""

    # Encode image properly as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    frame_bytes = buffer.getvalue()

    try:
        response = client.models.generate_content(
            model=model, # contents(image, prompt)
            contents=[
                    types.Part.from_bytes(
                        data=frame_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more consistent detection
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
        )

        print("Model Response:")
        print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

    return response.text


def gen_flat_annotations(input_csv:str, output_csv:str):
    with open(input_csv, "r", newline="", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        header = next(reader)  # skip header

        writer = csv.writer(outfile)
        writer.writerow(["Frame", "Label", "Coordinates", "Confidence"])

        for row in reader:
            try:
                print("\nRAW ROW:", row)

                # Frame name
                frame = row[0]
                coords_str = row[1]

                print("FRAME:", frame)
                print("COORDS STR:", coords_str)

                # Rejoin ONLY for regex scanning
                full_line = ",".join(row)

                # Extract quoted list literals: Labels + Confidences
                matches = re.findall(r'"(\[.*?\])"', full_line)

                print("REGEX MATCHES:", matches)

                if len(matches) != 2:
                    print("Skipping row, regex did not find 2 lists")
                    continue

                labels_str, conf_str = matches

                print("LABELS STR:", labels_str)
                print("CONF STR:", conf_str)

                # Safely parse lists
                coords_list = ast.literal_eval(coords_str)
                labels_list = ast.literal_eval(labels_str)
                conf_list = ast.literal_eval(conf_str)

                print("PARSED COORDS:", coords_list)
                print("PARSED LABELS:", labels_list)
                print("PARSED CONF:", conf_list)

                # Flatten detections
                for label, coord, conf in zip(labels_list, coords_list, conf_list):

                    # 'person 1' -> 'person'
                    label_clean = label.split()[0]

                    # 'person 1 (x,y,w,h)' -> '(x,y,w,h)'
                    if "(" in coord:
                        coord_clean = "(" + coord.split("(", 1)[1]
                    else:
                        coord_clean = coord

                    print("FLAT ROW:", frame, label_clean, coord_clean, conf)

                    writer.writerow([frame, label_clean, coord_clean, conf])

            except Exception as e:
                print("Skipping malformed row:", frame, e)
    return


def gen_flat_annotations2(input_csv:str, output_csv:str):
    with open(input_csv, "r", newline="", encoding="utf-8") as infile, \
         open(output_csv, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        header = next(reader)  # skip header

        writer = csv.writer(outfile)
        writer.writerow(["Frame", "Label", "Coordinates", "Confidence"])  # flat header

        for row in reader:
            # Rejoin the row in case some fields had extra commas
            line = ",".join(row)
            
            # Frame name
            frame = row[0]

            # Coordinates string is fine as-is
            coords_str = row[1]

            # Extract Labels and Confidences using regex
            pattern = r'\],\s*(\[.*?\]),\s*(\[.*?\])$'
            match = re.search(pattern, line)
            if match:
                labels_str = match.group(1)
                conf_str = match.group(2)

                # Convert to Python lists
                labels_list = ast.literal_eval(labels_str)
                conf_list = ast.literal_eval(conf_str)
                coords_list = ast.literal_eval(coords_str)  # now parse Coordinates

                # Flatten each detection
                for label, coord, conf in zip(labels_list, coords_list, conf_list):
                    # Remove numbering from label if needed: 'person 1' -> 'person'
                    label_clean = label.split()[0]

                    # Remove label prefix from coordinate: 'person 1 (x, y, ...)' -> '(x, y, ...)'
                    if "(" in coord:
                        coord_clean = "(" + coord.split("(", 1)[1]
                    else:
                        coord_clean = coord

                    writer.writerow([frame, label_clean, coord_clean, conf])
            else:
                print("No match for line:", line)
    return


def llm_process():
    # Load Environment Keys
    load_dotenv(dotenv_path="utils/.env")

    # Load configuration to get YOLO settings
    config = load_config("config.json")
    api_key = os.getenv("GEMINI_API_KEY")
    prompt = config.get("DEFAULT_PROMPT")
    model = config.get("AI_MODEL")
    verbose = config.get("DEBUG_MODE")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = PROMPT

    # Pull all images from the image folder
    image_paths = []
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.avif')):
            image_paths.append(os.path.join(IMAGE_FOLDER, filename))

    responses = []
    for image_path in image_paths:
        response = gemini_response(model, api_key, prompt, image_path, CSV_FILE, verbose)
        # Example: response = "['chair 1 (1756, 499, 1919, 648)']",['chair'],['0.85']
        row = f"{os.path.basename(image_path)},{response.strip()}" # Adds frame file name at the start
        responses.append(row)

    headers = ["Frame", "Coordinates", "Labels", "Confidences"]    
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as csvfile:
        csvfile.write(",".join(headers) + "\n")
        for row in responses:
            csvfile.write(row + "\n")

    return


def main():
    llm_process()
    gen_flat_annotations(CSV_FILE, ANNO_FILE)
    #gen_flat_annotations2(CSV_FILE, ANNO_FILE)

    return
    

if __name__ == "__main__":
    main()