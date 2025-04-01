import shutil
import cv2
import os

def extract_frames(video_path, output_folder):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return
    
    # if not os.path.exists(output_folder):
    if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        if not ret:
            break  # No more frames
        
        # Save the current frame as an image file
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    # Release the video capture object
    video.release()
    print(f"Frames extracted and saved to {output_folder}")

# Example usage
video_path = 'data/test.mp4'  # Replace with your video path
output_folder = 'data/video_frames'        # Folder to save frames
extract_frames(video_path, output_folder)
