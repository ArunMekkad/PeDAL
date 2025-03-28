import cv2
import os
import sys
import argparse
from tqdm import tqdm
import sys

# Add the project root to the path so we can import the utils module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_yolo_model

def detect_people_image(image_path, model=None, output_path=None, class_id=0):
    """
    Detect pedestrians in a single image.
    
    Args:
        image_path: Path to input image
        model: YOLO model instance or path to model
        output_path: Optional path to save annotated image
        class_id: Class ID for "person" (0 for COCO, 1 for custom model)
    
    Returns:
        Annotated image with detections
    """
    # Load model if not provided
    if model is None or isinstance(model, str):
        model_type = "custom" if model and "model.pt" in model else "yolov8"
        model = load_yolo_model(model, model_type)
    
    # Run the model on the image
    results = model(image_path)

    # Filter results to include only the person class
    people = []
    for result in results:
        for box in result.boxes:
            if box.cls == class_id:
                people.append(box)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    height, width = image.shape[:2]

    # Draw bounding boxes for people
    for person in people:
        x_min, y_min, x_max, y_max = map(int, person.xyxy[0])
        conf = person.conf.item()  # Convert the tensor to a scalar
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = "Person" if class_id == 0 else "Pedestrian"
            cv2.putText(image, f"{label} {conf:.2f}", (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    # Save output if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"Detection result saved to {output_path}")
    
    return image

def detect_people_video(input_video, output_dir, class_id=1, class_name="Pedestrian", model_path=None):
    """
    Detect pedestrians in a video and save the annotated result.
    
    Args:
        input_video: Path to input video
        output_dir: Directory to save output video
        class_id: Class ID for pedestrian class (default 1 for custom model)
        class_name: Label to show in annotations
        model_path: Optional path to custom model
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_video.mp4')

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video at {input_video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load the model
    model_type = "custom" if model_path and "model.pt" in model_path else "yolov8"
    model = load_yolo_model(model_path, model_type)

    # Process each frame with a progress bar
    with tqdm(total=frame_count, desc='Processing', unit='frame', dynamic_ncols=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run the model on the frame
            results = model(frame)

            # Filter results to include only the specified class
            people = [box for result in results for box in result.boxes if box.cls == class_id]

            # Draw bounding boxes for people
            for person in people:
                x_min, y_min, x_max, y_max = map(int, person.xyxy[0])
                conf = person.conf.item()  # Convert the tensor to a scalar
                if conf > 0.4:  # Confidence threshold
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x_min, y_min - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

            # Write the frame with the detection boxes
            out.write(frame)
            pbar.update(1)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Pedestrian detection completed. Processed video saved to {output_path}")

def display_image(image, title="Detection Result"):
    """Display an image in a window"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect pedestrians in images or videos")
    parser.add_argument("input_path", help="Path to input image or video")
    parser.add_argument("--output", help="Path to output directory (for video) or file (for image)")
    parser.add_argument("--model", help="Path to YOLO model file", default=None)
    parser.add_argument("--class-id", type=int, help="Class ID for pedestrian (0 for COCO, 1 for custom)", default=None)
    parser.add_argument("--show", action="store_true", help="Display results (for images)")
    parser.add_argument("--mode", choices=["auto", "image", "video"], default="auto", 
                       help="Force processing mode (auto detects based on extension)")
    args = parser.parse_args()
    
    # Determine if input is image or video based on extension
    if args.mode == "auto":
        input_ext = os.path.splitext(args.input_path)[1].lower()
        if input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            mode = "image"
        elif input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            mode = "video"
        else:
            print(f"Could not determine input type from extension '{input_ext}'. Use --mode to specify.")
            sys.exit(1)
    else:
        mode = args.mode
    
    # Determine class ID based on model type (custom or standard YOLO)
    if args.class_id is None:
        # Default: class 1 for custom model, class 0 for standard YOLO
        class_id = 1 if args.model and "model.pt" in args.model else 0 
    else:
        class_id = args.class_id
    
    if mode == "image":
        # For image mode, process a single image
        result = detect_people_image(args.input_path, args.model, args.output, class_id)
        if args.show and result is not None:
            display_image(result)
    else:
        # For video mode, process a video
        if not args.output:
            args.output = "output"  # Default output directory
        detect_people_video(args.input_path, args.output, class_id, "Pedestrian", args.model)
