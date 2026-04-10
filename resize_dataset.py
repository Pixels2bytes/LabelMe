import os
from PIL import Image
from tqdm import tqdm

INPUT_FOLDER = "create dataset/dangerous_weapons/images" # "resources/test_llm" # "create dataset/dangerous_weapons/images"
OUTPUT_FOLDER = "datasets/image_test_gt" # "datasets/image_test_gt"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def make_square_with_padding(img):
    width, height = img.size
    max_dim = max(width, height)

    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2

    new_img.paste(img, (paste_x, paste_y))
    return new_img


def resize_to_multiple_of_target(img, target_size):
    width, height = img.size  # should already be square

    if width < target_size:
        # pad up to target_size
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        paste = ((target_size - width) // 2, (target_size - height) // 2)
        new_img.paste(img, paste)
        return new_img

    # find nearest lower multiple of target_size
    target_size = (width // target_size) * target_size
    if target_size == 0:
        target_size = 320

    if target_size != width:
        img = img.resize((target_size, target_size), Image.LANCZOS)

    return img


def process_image(input_path, output_path, target_size=320):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")

            width, height = img.size

            # Step 1: make square if needed
            if width != height:
                img = make_square_with_padding(img)

            # Step 2: resize/pad to multiple of target_size
            img = resize_to_multiple_of_target(img, target_size)

            img.save(output_path)

    except Exception:
        pass


def resize_images_process(INPUT_FOLDER:str, OUTPUT_FOLDER:str, target_size:int):
    image_files = []

    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            image_files.append((root, file))

    with tqdm(total=len(image_files), desc="Resizing Images", unit="img") as pbar:
        for root, file in image_files:
            input_path = os.path.join(root, file)

            rel_path = os.path.relpath(root, INPUT_FOLDER)
            save_dir = os.path.join(OUTPUT_FOLDER, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            output_path = os.path.join(save_dir, file)

            process_image(input_path, output_path)

            pbar.update(1)

def main():
    target_size = 320
    resize_images_process(INPUT_FOLDER, OUTPUT_FOLDER, target_size)

if __name__ == "__main__":
    main()