import os
import cv2

def png_to_video(png_folder, output_video_path, size=(512, 512), fps=10):
    # Get list of PNG files in the folder
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
    
    # Sort files based on filename
    png_files.sort()

    # Define the codec and create VideoWriter object
    # Use an integer directly if cv2.VideoWriter_fourcc is unavailable
    fourcc = 0x7634706d  # Equivalent to 'mp4v' in hexadecimal
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    # Loop through all PNG files and add them to the video
    for file in png_files:
        img = cv2.imread(os.path.join(png_folder, file))
        img_resized = cv2.resize(img, size)  # Resize the image if necessary
        out.write(img_resized)

    # Release the video writer
    out.release()

# Usage
png_to_video('frames', 'output_video.mp4')

