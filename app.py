from flask import Flask, render_template, request, redirect, url_for
import os
import csv
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
CSV_FILE = 'annotations.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize CSV with new header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "file", "label", "coordinates",
            "box_width", "box_height",
            "image_width", "image_height"
        ])

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            filename = secure_filename(img.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            img.save(save_path)
            return redirect(url_for('annotate_image', filename=filename))
    return render_template('index.html')

@app.route('/annotate/<filename>', methods=['GET', 'POST'])
def annotate_image(filename):
    if request.method == 'POST':
        import json
        boxes = json.loads(request.form.get('boxes'))  # List of dicts {x1, y1, x2, y2}
        labels = json.loads(request.form.get('labels'))  # List of strings

        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Write to CSV with box width/height and image width/height
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                coords = f"({x1}, {y1}, {x2}, {y2})"
                box_width = x2 - x1
                box_height = y2 - y1
                writer.writerow([
                    filename, label, coords,
                    box_width, box_height,
                    img_width, img_height
                ])

        return "Saved!"

    return render_template('annotate.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
