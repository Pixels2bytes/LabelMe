"""Grabs the images in the dataset and the annotations file to generate 16 different versions of an image to balance the dataset."""
import os
from turtle import pd

import cv2

def rotate_horizontal(orig, folder_path):
    # Code to rotate the image horizontally and save it in images folder
    filename = os.path.basename(orig)
    name, ext = os.path.splitext(filename)
    new_name = f"{name}_horiz{ext}"
    

    return new_name


def grayscaled_images(img, folder_path):
    # Code to create grayscaled versions of the images and save them
    filename = os.path.basename(img)
    name, ext = os.path.splitext(filename)
    cate_name = f"{name}_gray"
    imgs = []

    for i in range(5):
        new_name = f"{cate_name}_{i}{ext}"
        path = os.path.join(folder_path, new_name)
        imgs.append(new_name)

        if i == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
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


def contrast_images(img, folder_path):
    # Code to create contrast versions of the images and save them
    filename = os.path.basename(img)
    name, ext = os.path.splitext(filename)
    cate_name = f"{name}_contrast"
    imgs = []

    for i in range(5):
        new_name = f"{cate_name}_{i}{ext}"
        path = os.path.join(folder_path, new_name)
        imgs.append(new_name)

        if i == 0:
            grainy = cv2.addWeighted(img, 0.5, img, 0.5, 0) # Add grain to the image
            cv2.imwrite(path, grainy)
        if i == 1:
            darkened = cv2.addWeighted(img, 0.5, img, 0, 0) # Darken the image
            cv2.imwrite(path, darkened)
        if i == 2:
            brightened = cv2.addWeighted(img, 1.5, img, 0, 0) # Brighten the image
            cv2.imwrite(path, brightened)
        if i == 3:
            inverted = cv2.bitwise_not(img) # Invert the image
            cv2.imwrite(path, inverted)
        if i == 4:
            blurred = cv2.GaussianBlur(img, (5, 5), 0) # Blur the image
            cv2.imwrite(path, blurred)


    return imgs


def create_more_images_process(annotations_file, target_labels, ignore_labels, combo_labels, img_dir):
    frames = annotations["Frame"].tolist() # List of image names in the dataset
    labels = annotations["Label"].tolist() # List of labels corresponding to the images in the dataset
    images = []
    h_images = []
    orig_dir = f"{img_dir}/original"
    horiz_dir = f"{img_dir}/horizontal"
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(horiz_dir, exist_ok=True)
    
    # Load annotations csv file
    annotations = pd.read_csv(annotations_file)

    # Find all images that have the target labels
    target_images = []
    image_key = {} # Dictionary to store the image name and its corresponding labels
    for frame, label in zip(frames, labels):
        if label in target_labels and frame in target_images:
            target_images.append(frame)
            if frame not in image_key:
                image_key[frame] = []
            image_key[frame].append(label)

    # If there are no ignore labels, then skip
    if ignore_labels:
        # If combo_labels is false, only use images that have the target labels
        if combo_labels == False:
            # Find all images that have ignore labels in image_key and remove them from the target_images list
            for frame in image_key:
                if any(label in ignore_labels for label in image_key[frame]):
                    if frame in target_images:
                        target_images.remove(frame)

    for img in target_images:
        orig = f"{img_dir}/original/{img}"
        # Create more images from the original image
        horiz = rotate_horizontal(orig, img_dir)
        h_images.append(horiz)        

        # 5 versions of the original and horizontal images (10 total)
        orig_gray_images = grayscaled_images(orig, orig_dir)
        horiz_gray_images = grayscaled_images(horiz, horiz_dir)

        # 5 versions of the original and horizontal images (10 total)
        orig_contra_images = contrast_images(orig, orig_dir)
        horiz_contra = contrast_images(horiz, horiz_dir)

        for orig_gray in orig_gray_images:
            filename = os.path.basename(orig_gray)
            name, ext = os.path.splitext(filename)
            images.append(orig_gray)
        for horiz_gray in horiz_gray_images:
            filename = os.path.basename(horiz_gray)
            name, ext = os.path.splitext(filename)
            images.append(horiz_gray)
        for orig_contra in orig_contra_images:
            filename = os.path.basename(orig_contra)
            name, ext = os.path.splitext(filename)
            images.append(orig_contra)
        for horiz_contra in horiz_contra:
            filename = os.path.basename(horiz_contra)
            name, ext = os.path.splitext(filename)
            images.append(horiz_contra)
        
    msg = f"""f"\n==========================\n ORiginal Gray: {len(orig_gray_images)}\nHorizontal Gray: {len(horiz_gray_images)}\n
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
    combo_labels = True # True = images with both target and ignore labels. False = only images with target labels and nothing else
    
    # Test
    target_labels = ["cow"]
    ignore_labels = ["tree"]
    annotations_file = "resources/ground_truth/gt_images/gt_annon.csv"
    img_dir = "resources/ground_truth/gt_images"

    create_more_images_process(annotations_file, target_labels, ignore_labels, combo_labels, img_dir)


if __name__ == "__main__":
    main()