import os
import hashlib
import shutil
import csv
from pathlib import Path

def compute_file_hash(file_path, chunk_size=8192):
    """
    Compute SHA256 hash of a file.
    Reading in chunks allows large file handling.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_duplicate_images(dataset_path):
    """
    Walk through dataset directory and detect identical images.
    Returns:
        hash_dict: {hash: [file1, file2, ...]}
    """
    hash_dict = {}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                full_path = os.path.join(root, file)

                try:
                    file_hash = compute_file_hash(full_path)

                    if file_hash not in hash_dict:
                        hash_dict[file_hash] = [full_path]
                    else:
                        hash_dict[file_hash].append(full_path)

                except Exception as e:
                    print(f"Skipping {full_path} due to error: {e}")

    return hash_dict


def move_duplicates(hash_dict, duplicates_folder):
    """
    Moves duplicate files (keeping first occurrence untouched).
    Returns:
        duplicate_records: list of tuples (original, duplicate, hash)
    """
    os.makedirs(duplicates_folder, exist_ok=True)

    duplicate_records = []

    for file_hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            original = file_list[0]

            for duplicate in file_list[1:]:
                filename = os.path.basename(duplicate)
                destination = os.path.join(duplicates_folder, filename)

                # Handle filename collision inside duplicates folder
                counter = 1
                while os.path.exists(destination):
                    name, ext = os.path.splitext(filename)
                    destination = os.path.join(
                        duplicates_folder,
                        f"{name}_{counter}{ext}"
                    )
                    counter += 1

                shutil.move(duplicate, destination)

                duplicate_records.append((original, destination, file_hash))

    return duplicate_records


def write_csv_report(duplicate_records, output_csv_path):
    """
    Writes duplicate mapping report to CSV.
    """
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Original Image", "Duplicate Image (Moved)", "SHA256 Hash"])

        for record in duplicate_records:
            writer.writerow(record)


def main():
    """
    Main entry point.
    Modify dataset_path as needed.
    """

    dataset_path = "resources/dataset"  # Dataset path
    duplicates_folder = os.path.join(dataset_path, "duplicates")
    output_csv_path = os.path.join(dataset_path, "duplicate_report.csv")

    hash_dict = find_duplicate_images(dataset_path)

    print("Moving duplicates...")
    duplicate_records = move_duplicates(hash_dict, duplicates_folder)

    print("Writing CSV report...")
    write_csv_report(duplicate_records, output_csv_path)

    print(f"\nDone.")
    print(f"Total duplicates moved: {len(duplicate_records)}")
    print(f"CSV report saved to: {output_csv_path}")


if __name__ == "__main__":
    main()