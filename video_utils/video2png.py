import imageio
import os
from PIL import Image

def extract_frames(video_path, output_folder, frame_size=(128, 128)):
    # Check if the video path exists
    if not os.path.exists(video_path):
        print("Error: Video path does not exist.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use imageio to read the video
    reader = imageio.get_reader(video_path)
    frame_count = 0

    try:
        for i, frame in enumerate(reader):
            # Convert array to an image
            image = Image.fromarray(frame)
            # Resize the image
            image = image.resize(frame_size)

            # Save the frame as a PNG file
            frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
            image.save(frame_path)

            # Increment frame count
            frame_count += 1
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        reader.close()

    print(f"Frames extracted: {frame_count}")

# Example usage
video_path = "C:\\Users\\Leon\\Desktop\\video_compressor_gray\\videos\\cm_kobe.mp4"  # Adjust path as necessary
output_folder = "C:\\Users\\Leon\\Desktop\\video_compressor_gray\\frames\\cm_frames"
extract_frames(video_path, output_folder)
