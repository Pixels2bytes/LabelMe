import os
from pathlib import Path


def get_image_files(folder_path:str):
    """
    Returns a sorted list of image file Paths in the given folder.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

    folder = Path(folder_path)

    image_files = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    # Sort for consistent numbering
    image_files.sort()

    return image_files


def rename_images(folder_path:str, start_name:str, verbose:bool=False):
    """
    Renames all images in folder_path to:
    start_name_1, start_name_2, ...
    """
    image_files = get_image_files(folder_path)

    if not image_files:
        print("No image files found.")
        return

    for index, file_path in enumerate(image_files, start=1):
        new_filename = f"{start_name}_{index}{file_path.suffix.lower()}"
        new_path = file_path.parent / new_filename

        if verbose:
            print(f"Renaming: {file_path.name} TO {new_filename}")
        file_path.rename(new_path)

    print("Renaming complete.")


def main():
    folder_path = r"PATH_TO_DATASET_FOLDER"
    start_name = "batch000_2_24_2024"
    verbose = True

    rename_images(folder_path, start_name, verbose)


if __name__ == "__main__":
    main()