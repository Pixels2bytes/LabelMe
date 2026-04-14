"""Grabs the images in the dataset and the annotations file to generate 16 different versions of an image to balance the dataset."""
import os
import pandas as pd
import cv2
from tqdm import tqdm

def rotate_horizontal(image, orig, folder_path):
    # Code to rotate the image horizontally
    filename = os.path.basename(orig)
    name, ext = os.path.splitext(filename)
    new_name = f"{name}_horiz{ext}"
    path = os.path.join(folder_path, new_name)

    flipped = cv2.flip(image, 1)
    cv2.imwrite(path, flipped)

    return new_name


def grayscaled_images(image, img_path, folder_path):
    # Code to create grayscaled versions of the images and save them
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    cate_name = f"{name}_gray"
    imgs = []

    for i in range(5):
        new_name = f"{cate_name}_{i}{ext}"
        path = os.path.join(folder_path, new_name)
        imgs.append(new_name)
        
        # If file already exists in folder, skip
        if os.path.exists(path):
            continue

        if i == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale the image
            cv2.imwrite(path, gray)
        if i == 1:
            darkened = cv2.addWeighted(gray, 0.5, gray, 0, 0) # Darken the image
            cv2.imwrite(path, darkened)
        if i == 2:
            brightened = cv2.addWeighted(gray, 1.5, gray, 0, 0) # Brighten the image
            cv2.imwrite(path, brightened)
        if i == 3:
            inverted = cv2.bitwise_not(gray) # Invert the image
            cv2.imwrite(path, inverted)
        if i == 4:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Blur the image
            cv2.imwrite(path, blurred)

    return imgs


def contrast_images(image, img_path, folder_path):
    # Code to create contrast versions of the images and save them
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    cate_name = f"{name}_contrast"
    imgs = []

    for i in range(5):
        new_name = f"{cate_name}_{i}{ext}"
        path = os.path.join(folder_path, new_name)
        imgs.append(new_name)
        
        # If file already exists in folder, skip
        if os.path.exists(path):
            continue

        if i == 0:
            grainy = cv2.addWeighted(image, 0.5, image, 0.5, 0) # Add grain to the image
            cv2.imwrite(path, grainy)
        if i == 1:
            darkened = cv2.addWeighted(image, 0.5, image, 0, 0) # Darken the image
            cv2.imwrite(path, darkened)
        if i == 2:
            brightened = cv2.addWeighted(image, 1.5, image, 0, 0) # Brighten the image
            cv2.imwrite(path, brightened)
        if i == 3:
            inverted = cv2.bitwise_not(image) # Invert the image
            cv2.imwrite(path, inverted)
        if i == 4:
            blurred = cv2.GaussianBlur(image, (5, 5), 0) # Blur the image
            cv2.imwrite(path, blurred)

    return imgs


def create_more_images_process(annotations_file, target_labels, ignore_labels, combo_labels, img_dir):
    # Load annotations csv file
    annotations = pd.read_csv(annotations_file)

    frames = annotations["Frame"].tolist() # List of image names in the dataset
    labels = annotations["Label"].tolist() # List of labels corresponding to the images in the dataset

    images = []
    h_images = []
    orig_dir = f"{img_dir}/original"
    horiz_dir = f"{img_dir}/horizontal"
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(horiz_dir, exist_ok=True)

    # Find all images that have the target labels
    target_images = []
    image_key = {} # Dictionary to store the image name and its corresponding labels
    for frame, label in zip(frames, labels):
        if label in target_labels:
            if frame not in target_images:
                target_images.append(frame)
            if frame not in image_key:
                image_key[frame] = []
            image_key[frame].append(label)

    # If there are no ignore labels, then skip
    if ignore_labels:
        # If combo_labels is false, only use images that have the target labels
        if combo_labels == False:
            # Find all images that have ignore labels in image_key and remove them from the target_images list
            ignore_frames = set()
            for frame, label in zip(frames, labels):
                if label in ignore_labels:
                    ignore_frames.add(frame)

            filtered_images = []

            for frame in target_images:
                if frame in ignore_frames:
                    print(f"Removed {frame} from target images because it has ignore labels")
                    continue
                filtered_images.append(frame)

            target_images = filtered_images

    for img in tqdm(target_images, desc="Processing Images"):
        orig_path = os.path.join(img_dir, img)

        orig = cv2.imread(orig_path)
        if orig is None:
            continue

        # Create more images from the original image
        horiz_name = rotate_horizontal(orig, orig_path, horiz_dir)
        horiz_path = os.path.join(horiz_dir, horiz_name)
        h_images.append(horiz_name)
        
        horiz = cv2.imread(horiz_path)
        if horiz is None:
            continue

        # 5 versions of the original and horizontal images (10 total)
        orig_gray_images = grayscaled_images(orig, orig_path, orig_dir)
        horiz_gray_images = grayscaled_images(horiz, horiz_path, horiz_dir)

        # 5 versions of the original and horizontal images (10 total)
        orig_contra_images = contrast_images(orig, orig_path, orig_dir)
        horiz_contra_images = contrast_images(horiz, horiz_path, horiz_dir)

        for orig_gray in orig_gray_images:
            images.append(orig_gray)
        for horiz_gray in horiz_gray_images:
            images.append(horiz_gray)
        for orig_contra in orig_contra_images:
            images.append(orig_contra)
        for horiz_contra in horiz_contra_images:
            images.append(horiz_contra)
        
    msg = f"""\n==========================\n ORiginal Gray: {len(orig_gray_images)}\nHorizontal Gray: {len(horiz_gray_images)}\n
    Original Contrast: {len(orig_contra_images)}\nHorizontal Contrast: {len(horiz_contra)}\n
    Horizontal Images: {len(h_images)}\n Total Images Created: {len(images) + len(h_images)}\n==========================        
    """
    print(msg)
    return


def main():
    #target_labels = ["gun", "knife", "smartphone"]
    #ignore_labels = ["person", "hand"]
    #annotations_file = "create dataset/dangerous_weapons/image_annotations_this.csv"
    #img_dir = "datasets/neo_weapons"
    combo_labels = False # True = images with both target and ignore labels. False = only images with target labels and nothing else

    # Test
    target_labels = ["cow"]
    ignore_labels = ["tree"]
    annotations_file = "resources/ground_truth/gt_images/gt_annon.csv"
    img_dir = "resources/ground_truth/gt_images"

    create_more_images_process(annotations_file, target_labels, ignore_labels, combo_labels, img_dir)


if __name__ == "__main__":
    main()