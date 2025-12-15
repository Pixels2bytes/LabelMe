"""Reassigns the yolo label number from yolo text files."""

import os
import csv

OUTPUT_FOLDER = "dataset"
LABELS_FOLDER = f"{OUTPUT_FOLDER}/labels"
YOLO_TRAIN_FOLDER = f"{LABELS_FOLDER}/train"

assign_index = {
    '0': 81, # original label index 'person': 0 reassigned to 'smartphone': 81 index
}

def reassign_labels(file_path, assign_index):
    for fname in os.listdir(file_path):
        if not fname.endswith(".txt"):
            continue

        txt_path = os.path.join(file_path, fname)

        with open(txt_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                original_label = parts[0]
                if original_label in assign_index:
                    parts[0] = str(assign_index[original_label])
                new_lines.append(" ".join(parts) + "\n")

        with open(txt_path, "w") as f:
            f.writelines(new_lines)

        print(f"Reassigned labels in: {txt_path}")

    return


def main():
    file_path = "C:/Users/Ganbarou/Desktop/smartphones/labels" # YOLO_TRAIN_FOLDER
    reassign_labels(file_path, assign_index)

    return print("Label reassignment complete.")


if __name__ == "__main__":
    main()