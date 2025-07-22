import json
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def get_image_Id(img_name):
    """
    Generate image ID from filename following the specified format
    Format: camera{N}_{scene}_{frame}.png
    """
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId

def scale_bbox_to_image(bbox, original_size, target_size):
    """
    Scale bounding box coordinates from original size to target size
    bbox: [x1, y1, width, height]
    original_size: (width, height) of the image used for detection
    target_size: (width, height) of the current image
    """
    x1, y1, width, height = bbox
    
    # Calculate scaling factors
    scale_x = target_size[0] / original_size[0]
    scale_y = target_size[1] / original_size[1]
    
    # Scale coordinates
    scaled_x1 = x1 * scale_x
    scaled_y1 = y1 * scale_y
    scaled_width = width * scale_x
    scaled_height = height * scale_y
    
    return [scaled_x1, scaled_y1, scaled_width, scaled_height]

def draw_detections_on_image(image, detections, detection_image_size=None):
    """
    Draw bounding boxes and labels on image
    detection_image_size: (width, height) of the image used for creating detections
                         If None, assumes detections are already in correct scale
    """
    if image is None:
        return None
    
    img_copy = image.copy()
    current_height, current_width = image.shape[:2]
    current_image_size = (current_width, current_height)
    
    for detection in detections:
        bbox = detection['bbox']
        
        # Scale bbox if detection was made on different image size
        
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, current_width))
        y1 = max(0, min(y1, current_height))
        x2 = max(0, min(x2, current_width))
        y2 = max(0, min(y2, current_height))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw rectangle
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label with category and score
        label = f"Cat:{detection['category_id']} Score:{detection['score']:.2f}"
        
        # Ensure label position is within image
        label_y = max(20, int(y1) - 10)
        cv2.putText(img_copy, label, (int(x1), label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_copy

def process_camera_data(json_file_path, images_folder_path, output_folder=None, target_fps=25, detection_image_size=None):
    """
    Process JSON detections and render videos for different cameras
    detection_image_size: (width, height) tuple specifying the image size used for detections
                         If None, will try to auto-detect or assume same as current images
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        detections_data = json.load(f)
    
    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Group detections by image_id
    detections_by_image = {}
    for detection in detections_data:
        image_id = detection['image_id']
        if image_id not in detections_by_image:
            detections_by_image[image_id] = []
        detections_by_image[image_id].append(detection)
    
    # Get all image files
    images_folder = Path(images_folder_path)
    image_files = sorted(list(images_folder.glob('*.png')))
    
    if not image_files:
        print(f"No PNG images found in {images_folder_path}")
        return
    
    # Group images by camera
    cameras_data = {}
    scene_list = ['M', 'A', 'E', 'N']
    
    for img_file in image_files:
        try:
            img_name = img_file.name
            # Parse filename: camera{N}_{scene}_{frame}.png
            parts = img_name.split('.png')[0].split('_')
            camera_num = int(parts[0].split('camera')[1])
            scene = parts[1]
            frame_num = int(parts[2])
            
            if camera_num not in cameras_data:
                cameras_data[camera_num] = {}
            if scene not in cameras_data[camera_num]:
                cameras_data[camera_num][scene] = []
            
            cameras_data[camera_num][scene].append({
                'file_path': img_file,
                'frame_num': frame_num,
                'image_id': get_image_Id(img_name)
            })
        except (IndexError, ValueError) as e:
            print(f"Skipping invalid filename: {img_name} - {e}")
            continue
    
    # Sort frames by frame number for each camera-scene combination
    for camera_num in cameras_data:
        for scene in cameras_data[camera_num]:
            cameras_data[camera_num][scene].sort(key=lambda x: x['frame_num'])
    
    # Process each camera
    for camera_num in sorted(cameras_data.keys()):
        print(f"\nProcessing Camera {camera_num}:")
        
        for scene in sorted(cameras_data[camera_num].keys()):
            frames_data = cameras_data[camera_num][scene]
            if not frames_data:
                continue
                
            print(f"  Scene {scene}: {len(frames_data)} frames")
            
            # Create video writer if output folder is specified
            video_writer = None
            if output_folder:
                output_video_path = os.path.join(output_folder, f"camera{camera_num}_{scene}.mp4")
                
                # Get first image to determine video dimensions
                first_img = cv2.imread(str(frames_data[0]['file_path']))
                if first_img is not None:
                    height, width = first_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))
            
            # Process each frame
            for frame_data in frames_data:
                img_path = frame_data['file_path']
                image_id = frame_data['image_id']
                frame_num = frame_data['frame_num']
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"    Warning: Could not load image {img_path}")
                    continue
                
                # Get detections for this image
                detections = detections_by_image.get(image_id, [])
                
                # Draw detections
                if detections:
                    rendered_image = draw_detections_on_image(image, detections, detection_image_size)
                    print(f"    Frame {frame_num} (ID: {image_id}): {len(detections)} detections")
                else:
                    rendered_image = image
                    print(f"    Frame {frame_num} (ID: {image_id}): No detections")
                
                # Add frame info text
                info_text = f"Camera{camera_num} Scene:{scene} Frame:{frame_num} ID:{image_id}"
                cv2.putText(rendered_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to video if video writer is available
                if video_writer:
                    video_writer.write(rendered_image)
            
            # Release video writer
            if video_writer:
                video_writer.release()
                print(f"  Video saved: camera{camera_num}_{scene}.mp4")

def main():
    parser = argparse.ArgumentParser(description='Render camera videos with detections')
    parser.add_argument('json_file', help='Path to JSON file with detections')
    parser.add_argument('images_folder', help='Path to folder containing PNG images')
    parser.add_argument('--output', '-o', help='Output folder for rendered videos')
    parser.add_argument('--fps', type=int, default=25, help='Target FPS for output videos (default: 25)')
    parser.add_argument('--detection-width', type=int, help='Width of images used for detection')
    parser.add_argument('--detection-height', type=int, help='Height of images used for detection')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder not found: {args.images_folder}")
        return
    
    # Set detection image size if provided
    detection_image_size = None
    if args.detection_width and args.detection_height:
        detection_image_size = (args.detection_width, args.detection_height)
        print(f"Using detection image size: {detection_image_size}")
    
    process_camera_data(args.json_file, args.images_folder, args.output, args.fps, detection_image_size)
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

# Example usage:
# python camera_renderer.py detections.json /path/to/images --output /path/to/output --fps 25
# 
# With detection image size scaling:
# python camera_renderer.py detections.json /path/to/images --output /path/to/output --detection-width 640 --detection-height 480