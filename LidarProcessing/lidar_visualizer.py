import numpy as np
import open3d as o3d
import torch
from PIL import Image
import os
import argparse
import sys

# Add the project root to the path so we can import the utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_yolov5_model

def read_point_cloud(input_file):
    pcd = o3d.io.read_point_cloud(input_file)
    return pcd

def extract_image_from_point_cloud(pcd, width=640, height=480):
    points = np.asarray(pcd.points)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x = ((points[:, 0] - x_min) / (x_max - x_min) * width).astype(np.int32)
    y = ((points[:, 1] - y_min) / (y_max - y_min) * height).astype(np.int32)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[y, x] = 255
    return image, x_min, x_max, y_min, y_max

def filter_pedestrian_clusters(pcd, bounding_boxes, x_min, x_max, y_min, y_max):
    points = np.asarray(pcd.points)
    filtered_points = []

    for box in bounding_boxes:
        # Convert bounding box coordinates to match the point cloud scale
        x_min_box = box[0] * (x_max - x_min) / 640 + x_min
        y_min_box = box[1] * (y_max - y_min) / 480 + y_min
        x_max_box = box[2] * (x_max - x_min) / 640 + x_max
        y_max_box = box[3] * (y_max - y_min) / 480 + y_max

        mask = (points[:, 0] >= x_min_box) & (points[:, 0] <= x_max_box) & (points[:, 1] >= y_min_box) & (points[:, 1] <= y_max_box)
        filtered_points.append(points[mask])

    filtered_points = np.vstack(filtered_points) if filtered_points else np.empty((0, 3))
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd

def get_bounding_boxes(image, model):
    results = model(image)
    results = results.pandas().xyxy[0]
    pedestrian_boxes = results[results['name'] == 'person']
    return pedestrian_boxes[['xmin', 'ymin', 'xmax', 'ymax']].values

def main(input_file, model_path=None, visualize=True, output_file=None):
    """
    Process a point cloud file to detect and filter pedestrian clusters.
    
    Args:
        input_file: Path to input point cloud file
        model_path: Path to YOLOv5 model (optional)
        visualize: Whether to display the result in a visualization window
        output_file: Optional path to save the filtered point cloud
    
    Returns:
        The filtered point cloud containing only pedestrian clusters
    """
    # Load the model
    model, device = load_yolov5_model(model_path)
    
    # Process the point cloud
    pcd = read_point_cloud(input_file)
    image, x_min, x_max, y_min, y_max = extract_image_from_point_cloud(pcd)
    bounding_boxes = get_bounding_boxes(image, model)
    num_pedestrians = len(bounding_boxes)
    print(f"Number of pedestrians detected: {num_pedestrians}")
    
    if num_pedestrians > 0:
        filtered_pcd = filter_pedestrian_clusters(pcd, bounding_boxes, x_min, x_max, y_min, y_max)
        
        # Save the filtered point cloud if output path is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            o3d.io.write_point_cloud(output_file, filtered_pcd)
            print(f"Filtered point cloud saved to: {output_file}")
        
        # Visualize the result if requested
        if visualize:
            o3d.visualization.draw_geometries([filtered_pcd],
                                          window_name="Pedestrian Clusters",
                                          width=800,
                                          height=600,
                                          left=50,
                                          top=50,
                                          point_show_normal=False)
        
        return filtered_pcd
    else:
        print("No pedestrians detected in the point cloud.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter point cloud data to identify pedestrians")
    parser.add_argument("input_file", help="Path to the input point cloud file")
    parser.add_argument("--model", help="Path to the YOLOv5 model file", default=None)
    parser.add_argument("--output", help="Path to save the filtered point cloud", default=None)
    parser.add_argument("--no-visualize", help="Disable visualization", action="store_true")
    args = parser.parse_args()
    
    main(args.input_file, args.model, not args.no_visualize, args.output)
