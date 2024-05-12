import os
import cv2
import re

def png_to_video(png_folder, output_video_path, size=(512, 512), fps=1):
    # Get list of PNG files in the folder
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
    
    # Sort files based on filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None

    # Sort the filenames by the extracted number
    sorted_filenames = sorted(png_files, key=extract_number)

    # Define the codec and create VideoWriter object
    # Use an integer directly if cv2.VideoWriter_fourcc is unavailable
    fourcc = 0x7634706d  # Equivalent to 'mp4v' in hexadecimal
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # Loop through all PNG files and add them to the video
    for file in sorted_filenames:
        img = cv2.imread(os.path.join(png_folder, file))
        img_resized = cv2.resize(img, size)  # Resize the image if necessary
        out.write(img_resized)

    # Release the video writer
    out.release()

# Usage

png_to_video("C:\\Users\\Leon\\Desktop\\video_compressor_gray\\server\\result\\test_images", "C:\\Users\\Leon\\Desktop\\video_compressor_gray\\server\\result\\test_video.mp4", fps=25)


