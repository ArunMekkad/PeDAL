import cv2
import os
import glob
import argparse
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get list of all files in the folder without sorting
    image_files = glob.glob(os.path.join(image_folder, '*'))
    
    if not image_files:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the width and height
    frame = cv2.imread(image_files[0])
    if frame is None:
        print(f"Error: {image_files[0]} could not be read as an image.")
        return
    
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each image and write to the video with a progress bar
    for image_file in tqdm(image_files, desc="Creating Video"):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: {image_file} could not be read and will be skipped.")
            continue
        video.write(frame)
    
    # Release the video writer object
    video.release()
    print(f"Video created successfully and saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from a folder of images")
    parser.add_argument("image_folder", help="Path to the folder containing images")
    parser.add_argument("output_video", help="Path where the output video will be saved")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    args = parser.parse_args()
    
    create_video_from_images(args.image_folder, args.output_video, args.fps)
