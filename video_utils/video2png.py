import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables
    frame_count = 1
    size = (128, 128)

    # Read until the end of the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        frame_resized = cv2.resize(frame, size)
        
        # Break the loop if no frame is retrieved
        if not ret:
            break
        
        # Save the frame as a PNG file
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame_resized)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Frames extracted: {frame_count}")

# Example usage
video_path = "C:\\xampp\\htdocs\\video_compressor\\videos\\orig_kobe.mp4"
output_folder = "frames"
extract_frames(video_path, output_folder)
