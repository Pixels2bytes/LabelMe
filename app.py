from flask import Flask, render_template, request, redirect, url_for
import os
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
CSV_FILE = 'annotations.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize CSV with header if not exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "coordinates"])

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
        boxes = request.form.get('boxes')  # JSON string list of boxes
        labels = request.form.get('labels')  # JSON string list of labels

        import json
        boxes = json.loads(boxes)
        labels = json.loads(labels)

        # Write to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for box, label in zip(boxes, labels):
                coords = f"({box['x1']}, {box['y1']}, {box['x2']}, {box['y2']})"
                writer.writerow([filename, label, coords])

        return "Saved!"

    return render_template('annotate.html', filename=filename)
    
if __name__ == '__main__':
    app.run(debug=True)
