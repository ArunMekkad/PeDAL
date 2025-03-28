"""
Common utility functions for the pedestrian detection project.
"""
import os
import torch
from ultralytics import YOLO

def find_model_path(model_name, search_dirs=None):
    """
    Find a model file by searching in common directories.
    
    Args:
        model_name: Name of the model file (e.g., 'yolov8n.pt')
        search_dirs: Optional list of directories to search in
        
    Returns:
        Full path to the model if found, None otherwise
    """
    if search_dirs is None:
        # Default search directories include the current directory, parent directory,
        # and component-specific directories
        import inspect
        caller_file = inspect.stack()[1].filename
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        project_root = os.path.dirname(caller_dir)
        
        search_dirs = [
            os.path.dirname(os.path.abspath(__file__)),  # Module directory
            caller_dir,  # Directory of the calling script
            project_root,  # Project root directory
            os.path.join(project_root, "PedestrianDetection"),
            os.path.join(project_root, "LidarProcessing")
        ]
    
    # Look for the model in each search directory
    for directory in search_dirs:
        model_path = os.path.join(directory, model_name)
        if os.path.exists(model_path):
            return model_path
    
    return None

def load_yolo_model(model_path=None, model_type="custom"):
    """
    Load a YOLO model from a file or use a default path.
    
    Args:
        model_path: Path to model file. If None, will search in default locations.
        model_type: "custom", "yolov8", or "yolov5" to determine default filenames
    
    Returns:
        YOLO model instance
    """
    if model_path is None:
        # Default model filename based on type
        if model_type == "custom":
            filename = "model.pt"
        elif model_type == "yolov8":
            filename = "yolov8n.pt"
        elif model_type == "yolov5":
            filename = "yolov5s.pt"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Search for the model
        model_path = find_model_path(filename)
        if model_path is None:
            raise FileNotFoundError(f"Could not find {filename} in search paths. Please specify model path.")
    
    print(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)

def load_yolov5_model(model_path=None):
    """
    Load a YOLOv5 model using torch hub.
    
    Args:
        model_path: Optional path to a local model file
        
    Returns:
        YOLOv5 model and device
    """
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Determine model path
    if model_path is None:
        model_path = find_model_path("yolov5s.pt")
    
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading YOLOv5 model from: {model_path}")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', path=model_path, trust_repo=True).to(device)
    else:
        print("Downloading YOLOv5 model from torch hub")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True).to(device)
    
    return model, device 