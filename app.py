from flask import Flask, render_template, request, redirect, url_for, session
import os
import csv
import json
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = 'key123'

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "dataset"
IMAGE_FOLDER = f"{OUTPUT_FOLDER}/images"
IMAGE_TRAIN_FOLDER = f"{IMAGE_FOLDER}/train"
CSV_FILE = f"{OUTPUT_FOLDER}/annotations.csv"

RENAME_BASE = "smartphones"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(IMAGE_TRAIN_FOLDER, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "label", "coordinates",
            "box_width", "box_height",
            "image_width", "image_height"
        ])


def make_unique_name(folder, base, ext):
    rename = f"{base}{ext}"
    counter = 1

    while os.path.exists(os.path.join(folder, rename)):
        rename = f"{base}({counter}){ext}"
        counter += 1

    return rename


@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        images = request.files.getlist('images')
        session['queue'] = []

        for img in images:
            if img.filename == "":
                continue

            filename = secure_filename(img.filename)
            old_name, ext = os.path.splitext(filename)
            filename = make_unique_name(UPLOAD_FOLDER, old_name, ext)
            save_path = os.path.join(UPLOAD_FOLDER, filename)

            # Check if the image mode is RGBA or P (palette, which can also have transparency)
            if img.mode in ("RGBA", "P"):
                # Convert to RGB mode
                img = img.convert("RGB")
            img.save(save_path)

            session['queue'].append(filename)

        if session['queue']:
            return redirect(url_for('annotate_page', filename=session['queue'][0]))

    return render_template('index.html')


@app.route('/annotate/<filename>', methods=['GET', 'POST'])
def annotate_page(filename):
    queue = session.get('queue', [])

    if request.method == 'POST':
        if request.form.get("no_detections") == "1":
            if filename in queue:
                queue.remove(filename)
                session['queue'] = queue

            if queue:
                return redirect(url_for('annotate_page', filename=queue[0]))
            else:
                return "All images annotated!"

        boxes = json.loads(request.form.get('boxes'))
        labels = json.loads(request.form.get('labels'))

        # Load original image
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        img = Image.open(original_path)
        img_w, img_h = img.size

        # Generate renamed filename
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        new_name = make_unique_name(IMAGE_FOLDER, RENAME_BASE, ext)

        dataset_path = os.path.join(IMAGE_FOLDER, new_name)
        train_path = os.path.join(IMAGE_TRAIN_FOLDER, new_name)

        # Check if the image mode is RGBA or P (palette, which can also have transparency)
        if img.mode in ("RGBA", "P"):
            # Convert to RGB mode
            img = img.convert("RGB")
        img.save(dataset_path)
        img.save(train_path)

        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                coords = f"({x1}, {y1}, {x2}, {y2})"
                box_w = x2 - x1
                box_h = y2 - y1

                writer.writerow([
                    new_name,
                    label,
                    coords,
                    box_w,
                    box_h,
                    img_w,
                    img_h
                ])

        if filename in queue:
            queue.remove(filename)
            session['queue'] = queue
        
        if queue:
            return redirect(url_for('annotate_page', filename=queue[0]))
        else:
            return "All images annotated!"

    return render_template('annotate.html', filename=filename)


if __name__ == "__main__":
    app.run(debug=True)