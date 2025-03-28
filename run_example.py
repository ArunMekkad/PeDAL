#!/usr/bin/env python3
"""
Example script to demonstrate how to use the pedestrian detection tools.
This script shows the various ways to use the tools in this repository.
"""

import os
import argparse
import subprocess
import sys

def print_section(title):
    """Print a section title with decoration."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, capture=True):
    """Run a command and print its output."""
    print(f"Running: {' '.join(command)}")
    try:
        if capture:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout)
        else:
            # For interactive commands like image display
            result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture:
            print(e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run example pedestrian detection tools")
    parser.add_argument("--video", help="Path to a video file for pedestrian detection")
    parser.add_argument("--image", help="Path to an image file for pedestrian detection")
    parser.add_argument("--images_dir", help="Path to a directory of images to create a video")
    parser.add_argument("--point_cloud", help="Path to a point cloud file for LIDAR processing")
    parser.add_argument("--model", help="Path to YOLO model (automatically used for the right component)", default=None)
    parser.add_argument("--output_dir", help="Directory to save outputs", default="outputs")
    parser.add_argument("--interactive", action="store_true", help="Show results interactively when possible")
    parser.add_argument("--class_id", type=int, help="Class ID for pedestrian detection (0 for COCO, 1 for custom)", default=None)
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print welcome message
    print_section("Pedestrian Detection Examples")
    print("This script demonstrates how to use the various tools in this repository.")
    print("You can provide different inputs to test different components.")
    
    # Check if any arguments were provided
    if not (args.video or args.image or args.images_dir or args.point_cloud):
        print("No input files specified. Please provide at least one of: --video, --image, --images_dir, or --point_cloud")
        parser.print_help()
        return
    
    # Process video if provided
    if args.video:
        print_section("Video-based Pedestrian Detection")
        video_output_dir = os.path.join(args.output_dir, "video_detection")
        os.makedirs(video_output_dir, exist_ok=True)
        
        cmd = [
            sys.executable, 
            "PedestrianDetection/detector.py", 
            args.video, 
            "--output", video_output_dir,
            "--mode", "video"
        ]
        
        if args.model:
            cmd.extend(["--model", args.model])
        if args.class_id is not None:
            cmd.extend(["--class_id", str(args.class_id)])
        
        run_command(cmd)
    
    # Process image if provided
    if args.image:
        print_section("Image-based Pedestrian Detection")
        img_output_path = os.path.join(args.output_dir, "image_detection", os.path.basename(args.image))
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        
        cmd = [
            sys.executable, 
            "PedestrianDetection/detector.py", 
            args.image,
            "--output", img_output_path,
            "--mode", "image"
        ]
        
        if args.model:
            cmd.extend(["--model", args.model])
        if args.class_id is not None:
            cmd.extend(["--class_id", str(args.class_id)])
        if args.interactive:
            cmd.append("--show")
        
        run_command(cmd, not args.interactive)
    
    # Create video from images if provided
    if args.images_dir:
        print_section("Creating Video from Images")
        video_output_path = os.path.join(args.output_dir, "sequence_video.mp4")
        
        cmd = [
            sys.executable, 
            "PedestrianDetection/video_creator.py", 
            args.images_dir, 
            video_output_path
        ]
        run_command(cmd)
    
    # Process point cloud if provided
    if args.point_cloud:
        print_section("LIDAR Point Cloud Processing")
        
        # Run basic filtering
        lidar_output_file = os.path.join(args.output_dir, "filtered_pointcloud.bin")
        cmd = [
            sys.executable, 
            "LidarProcessing/filter.py", 
            args.point_cloud, 
            lidar_output_file
        ]
        run_command(cmd)
        
        # Run integrated pedestrian detection
        lidar_detection_output = os.path.join(args.output_dir, "pedestrian_detection.ply")
        cmd = [
            sys.executable, 
            "LidarProcessing/lidar_visualizer.py", 
            args.point_cloud,
            "--output", lidar_detection_output
        ]
        
        if args.model:
            cmd.extend(["--model", args.model])
        if not args.interactive:
            cmd.append("--no-visualize")
        
        run_command(cmd, not args.interactive)
    
    print_section("Example Run Completed")
    print(f"Check the output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 