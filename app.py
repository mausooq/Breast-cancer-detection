import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
from PIL import Image
import torch

app = Flask(__name__)

# Set up the paths for uploads and output images
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can replace with another model

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    # Get the uploaded file
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Perform detection
        results = model(filepath)
        
        # Process the first result
        result = results[0]  # YOLOv8 returns a list of results, take the first one
        
        # Generate the output filename
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Plot the result and save it
        fig = result.plot()
        Image.fromarray(fig).save(output_path)

        return render_template('result.html', input_image=file.filename, output_image=output_filename)

# Serve uploaded and output images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)