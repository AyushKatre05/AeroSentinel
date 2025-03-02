from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load YOLO model
model = YOLO("best.pt")

# Define object classes
classNames = ['Barrel', 'Barrels', 'Fuse', 'GUN', 'Gun', 'Missile', 'Missilr', 'TAnk', 'Tank', 'Tanks', 'tank']

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    return detect_objects(filename)

def detect_objects(file_path):
    image = cv2.imread(file_path)
    results = model(image)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            if cls < len(classNames):  # Prevent index errors
                current_class = classNames[cls]
                if conf > 0.5:
                    detections.append({
                        "class": current_class,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    # Draw bounding box on image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{current_class} {conf*100:.1f}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
    cv2.imwrite(output_path, image)  # Save output image with detections

    return jsonify({"detections": detections, "image_url": f"http://localhost:5000/output/{os.path.basename(output_path)}"})

@app.route('/output/<filename>')
def get_output_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
