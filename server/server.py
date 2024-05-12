from flask import Flask, request, redirect, url_for,jsonify,render_template, Response, send_from_directory
from flask_cors import CORS
import test_deeplearning 
import os
from werkzeug.utils import secure_filename
import subprocess
current_file_path = os.path.abspath(__file__)
import sys
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory)
from video_utils.png2video import png_to_video
from video_utils.video2png import extract_frames
import zipfile
import cv2



app = Flask(__name__)
CORS(app)

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = os.path.join(project_root,UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

upload_dir = os.path.join(current_directory, "upload")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

result_dir = os.path.join(current_directory, "result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

app.config['UPLOAD_FOLDER'] = upload_dir # Ensure this path exists and is correct
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Max upload size of 500 MB

ALLOWED_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov'}  # Acceptable video formats

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(error="No file part in the request"), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify(message=f"File uploaded successfully at {file_path}"), 200
    else:
        return jsonify(error="File type not allowed"), 400
    

@app.route('/compression', methods=['POST'])
def compress_video():
    # Extract video file path from request
    fileName = request.json['fileName']
    video_path = os.path.join(upload_dir, fileName)
    orig_folder = os.path.join(upload_dir, 'orig_images')
    app.config['ORIG_FOLDER'] = orig_folder
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    app.config['fps'] = fps

    extract_frames(video_path, orig_folder)
    command = ['python','test_deeplearning.py', 'gray_data.py', orig_folder]
    subprocess.run(command, check=True, cwd=current_directory)   
    meas_dir = os.path.join(result_dir, "meas_images")
    meas_video_path = os.path.join(result_dir, "meas_video.mp4")
    png_to_video(meas_dir, meas_video_path)

    mask_dir = os.path.join(result_dir, "mask_pre_images")
    mask_zip_path = os.path.join(result_dir, "mask.zip")
    with zipfile.ZipFile(mask_zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(mask_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, mask_dir))

    return jsonify(compressed_meas_path=meas_video_path, mask_zip_path=mask_zip_path)

@app.route('/decompression', methods=['POST'])
def decompress_video():
    command = ['python','test_deeplearning.py', 'gray_data.py', app.config['ORIG_FOLDER']]
    subprocess.run(command, check=True, cwd=current_directory)   
    test_images_dir = os.path.join(result_dir, "test_images")
    test_video_path = os.path.join(result_dir, "test_video.mp4")

    png_to_video(test_images_dir, test_video_path, fps=25)
    return jsonify(test_video_path=test_video_path)

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory(result_dir, filename, as_attachment=True)

if __name__ == '__main__':
    print('Starting Python Flask Server...')
    test_deeplearning.load_model()
    app.run(debug=True)