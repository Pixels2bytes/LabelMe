import os
import json
from collections import defaultdict
from tabnanny import verbose
from PIL import Image
from tqdm import tqdm

INPUT_FOLDER = "datasets/neo_weapons" # "create dataset/dangerous_weapons/images"
OUTPUT_DIR = "output/stats"
OUTPUT_JSON = f"{OUTPUT_DIR}/neo_image_stats.json"
verbose = True
os.makedirs(OUTPUT_DIR, exist_ok=True)

def count_files(folder_path):
    total = 0
    for _, _, files in os.walk(folder_path):
        total += len(files)
    return total


def get_image_stats(folder_path):
    resolution_counts = defaultdict(int)
    extension_counts = defaultdict(int)
    breakdown = []

    total_files = 0
    total_to_process = count_files(folder_path)

    with tqdm(total=total_to_process, desc="Processing Images") as pbar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size

                    ext = os.path.splitext(file)[1].lower().replace('.', '')
                    resolution_key = f"{width}x{height}"

                    resolution_counts[resolution_key] += 1
                    extension_counts[ext] += 1

                    breakdown.append({
                        "filename": file,
                        "ext": ext,
                        "width": width,
                        "height": height,
                        "location": root
                    })

                    total_files += 1

                except Exception:
                    pass  # skip bad files

                pbar.update(1)

    return {
        "folder": folder_path,
        "num_of_files": total_files,
        "statistics": [
            dict(resolution_counts),
            dict(extension_counts)
        ],
        "breakdown": breakdown
    }


def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def stats_process(INPUT_FOLDER:str, OUTPUT_JSON:str="output/stats/image_stats.json", verbose:bool=False):
    stats = get_image_stats(INPUT_FOLDER)
    save_to_json(stats, OUTPUT_JSON)
    if verbose:
        print(f"Saved stats to {OUTPUT_JSON}")


def main():
    stats_process(INPUT_FOLDER, OUTPUT_JSON)

if __name__ == "__main__":
    main()