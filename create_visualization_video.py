#!/usr/bin/env python3
"""
Visualization Video Creator for MZSON Results

Creates MP4 videos from visualization and selection images with the following logic:
- Shows visualization images at 0.5s intervals
- When a selection image exists for a step, shows it alongside for 3s
- Replaces selection image when a new one appears

Usage:
    python create_visualization_video.py --scene_id 00848-ziup5kvtCCR --subtask 0
    python create_visualization_video.py --scene_id 00848-ziup5kvtCCR  # defaults to subtask 0
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
import re
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_step_number(filename: str) -> int:
    """Extract step number from filename."""
    match = re.search(r'_(\d{4})\.png$', filename)
    if match:
        return int(match.group(1))
    return 0


def get_visualization_files(visualization_dir: str) -> List[Tuple[int, str]]:
    """Get all visualization files sorted by step number."""
    if not os.path.exists(visualization_dir):
        logger.error(f"Visualization directory not found: {visualization_dir}")
        return []
    
    files = []
    for filename in os.listdir(visualization_dir):
        if filename.endswith('.png'):
            step_num = extract_step_number(filename)
            files.append((step_num, filename))
    
    # Sort by step number
    files.sort(key=lambda x: x[0])
    logger.info(f"Found {len(files)} visualization files")
    return files


def get_selection_files(selection_dir: str) -> dict:
    """Get all selection files mapped by step number."""
    if not os.path.exists(selection_dir):
        logger.warning(f"Selection directory not found: {selection_dir}")
        return {}
    
    selection_map = {}
    for filename in os.listdir(selection_dir):
        if filename.startswith('step_') and filename.endswith('_candidates.png'):
            # Extract step number from filename like "step_0011_00848-ziup5kvtCCR_0_0_candidates.png"
            match = re.search(r'step_(\d+)_', filename)
            if match:
                step_num = int(match.group(1))
                selection_map[step_num] = filename
    
    logger.info(f"Found {len(selection_map)} selection files")
    return selection_map


def resize_image_to_fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image to fit within target dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def create_combined_frame(visualization_img: np.ndarray, selection_img: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a combined frame with visualization and optional selection image."""
    vis_h, vis_w = visualization_img.shape[:2]
    
    if selection_img is not None:
        # Create side-by-side layout with original sizes
        # Both Selection and Visualization: 1x size (original)
        selection_w = vis_w  # Selection takes original width
        selection_h = vis_h  # Selection takes original height
        
        # Resize selection image to fit
        selection_resized = resize_image_to_fit(selection_img, selection_w, selection_h)
        
        # Resize visualization image to fit
        vis_w_new = vis_w  # Visualization takes original width
        vis_h_new = vis_h  # Visualization takes original height
        vis_resized = resize_image_to_fit(visualization_img, vis_w_new, vis_h_new)
        
        # Use the larger height for the combined frame
        combined_h = max(selection_h, vis_h_new)
        combined_w = selection_w + vis_w_new
        combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        
        # Center the images vertically
        sel_y_offset = (combined_h - selection_h) // 2
        vis_y_offset = (combined_h - vis_h_new) // 2
        
        # Place selection on the left
        combined_frame[sel_y_offset:sel_y_offset+selection_h, :selection_w] = selection_resized
        # Place visualization on the right
        combined_frame[vis_y_offset:vis_y_offset+vis_h_new, selection_w:] = vis_resized
        
        return combined_frame
    else:
        # Just return visualization image (1x size - original)
        vis_h_new = vis_h
        vis_w_new = vis_w
        vis_resized = resize_image_to_fit(visualization_img, vis_w_new, vis_h_new)
        return vis_resized


def create_video(scene_id: str, subtask: int = 0, base_dir: str = "results/mzson_cfg_3dmem/0831") -> str:
    """Create MP4 video from visualization and selection images."""
    
    # Construct paths
    subtask_dir = os.path.join(base_dir, scene_id, f"subtask_{scene_id}_0_{subtask}")
    visualization_dir = os.path.join(subtask_dir, "visualization")
    selection_dir = os.path.join(subtask_dir, "selection")
    output_path = os.path.join(subtask_dir, f"{scene_id}_0_{subtask}_visualization.mp4")
    
    logger.info(f"Creating video for scene: {scene_id}, subtask: {subtask}")
    logger.info(f"Visualization dir: {visualization_dir}")
    logger.info(f"Selection dir: {selection_dir}")
    logger.info(f"Output path: {output_path}")
    
    # Get file lists
    visualization_files = get_visualization_files(visualization_dir)
    selection_map = get_selection_files(selection_dir)
    
    # Update selection map to use correct step numbers
    # Selection files use format: step_XXXX_scene_id_0_subtask_candidates.png
    updated_selection_map = {}
    for step_num, filename in selection_map.items():
        # Extract the correct step number from the filename
        match = re.search(r'step_(\d+)_', filename)
        if match:
            actual_step = int(match.group(1))
            updated_selection_map[actual_step] = filename
    selection_map = updated_selection_map
    
    if not visualization_files:
        logger.error("No visualization files found!")
        return ""
    
    # Get first image to determine dimensions
    first_vis_path = os.path.join(visualization_dir, visualization_files[0][1])
    first_img = cv2.imread(first_vis_path)
    if first_img is None:
        logger.error(f"Could not read first image: {first_vis_path}")
        return ""
    
    # Determine output dimensions
    vis_h, vis_w = first_img.shape[:2]
    if selection_map:
        # If we have selections, make room for side-by-side layout
        # Both Selection and Visualization: 1x size (original)
        selection_w = vis_w  # Selection takes original width
        selection_h = vis_h  # Selection takes original height
        vis_w_new = vis_w    # Visualization takes original width
        vis_h_new = vis_h    # Visualization takes original height
        output_w = selection_w + vis_w_new
        output_h = max(selection_h, vis_h_new)
    else:
        # Just visualization at 1x size (original)
        output_w = vis_w
        output_h = vis_h
    
    # Set up video writer with different codec options
    # Try different codecs in order of preference
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('X264', cv2.VideoWriter_fourcc(*'X264')),
    ]
    
    out = None
    used_codec = None
    
    for codec_name, fourcc in codecs_to_try:
        try:
            test_out = cv2.VideoWriter(output_path, fourcc, 2.0, (output_w, output_h))
            if test_out.isOpened():
                out = test_out
                used_codec = codec_name
                logger.info(f"Using codec: {codec_name}")
                break
            else:
                test_out.release()
        except Exception as e:
            logger.warning(f"Failed to use codec {codec_name}: {e}")
            continue
    
    if out is None:
        logger.error("Could not initialize video writer with any codec!")
        return ""
    
    if not out.isOpened():
        logger.error("Could not open video writer!")
        return ""
    
    current_selection_img = None
    current_selection_step = None
    
    try:
        for step_num, vis_filename in visualization_files:
            vis_path = os.path.join(visualization_dir, vis_filename)
            vis_img = cv2.imread(vis_path)
            
            if vis_img is None:
                logger.warning(f"Could not read visualization image: {vis_path}")
                continue
            
            # Check if there's a selection for this step
            if step_num in selection_map:
                selection_filename = selection_map[step_num]
                selection_path = os.path.join(selection_dir, selection_filename)
                selection_img = cv2.imread(selection_path)
                
                if selection_img is not None:
                    current_selection_img = selection_img
                    current_selection_step = step_num
                    logger.info(f"Found selection for step {step_num}: {selection_filename}")
            
            # Create combined frame
            combined_frame = create_combined_frame(vis_img, current_selection_img)
            
            # Ensure frame is in correct format (BGR for OpenCV)
            if combined_frame.shape[2] == 3:
                # Already BGR
                pass
            elif combined_frame.shape[2] == 4:
                # RGBA to BGR
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGBA2BGR)
            else:
                # Convert to BGR
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
            
            # Ensure frame dimensions match video writer
            if combined_frame.shape[:2] != (output_h, output_w):
                combined_frame = cv2.resize(combined_frame, (output_w, output_h))
            
            # Determine frame duration
            if current_selection_img is not None and current_selection_step == step_num:
                # Show for 3 seconds (6 frames at 2 FPS)
                frame_duration = 6
                logger.info(f"Step {step_num}: Showing with selection for 3 seconds")
            else:
                # Show for 0.5 seconds (1 frame at 2 FPS)
                frame_duration = 1
                logger.info(f"Step {step_num}: Showing visualization for 0.5 seconds")
            
            # Write frames
            for _ in range(frame_duration):
                out.write(combined_frame)
        
        logger.info(f"Video created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        return ""
    
    finally:
        out.release()


def create_video_with_ffmpeg(scene_id: str, subtask: int = 0, base_dir: str = "results/mzson_cfg_3dmem/0831") -> str:
    """Alternative method using FFmpeg to create video."""
    import subprocess
    import tempfile
    import shutil
    
    # Construct paths
    subtask_dir = os.path.join(base_dir, scene_id, f"subtask_{scene_id}_0_{subtask}")
    visualization_dir = os.path.join(subtask_dir, "visualization")
    selection_dir = os.path.join(subtask_dir, "selection")
    output_path = os.path.join(subtask_dir, f"{scene_id}_0_{subtask}_visualization_ffmpeg.mp4")
    
    logger.info(f"Creating video with FFmpeg for scene: {scene_id}, subtask: {subtask}")
    
    # Get file lists
    visualization_files = get_visualization_files(visualization_dir)
    selection_map = get_selection_files(selection_dir)
    
    # Update selection map to use correct step numbers
    updated_selection_map = {}
    for step_num, filename in selection_map.items():
        match = re.search(r'step_(\d+)_', filename)
        if match:
            actual_step = int(match.group(1))
            updated_selection_map[actual_step] = filename
    selection_map = updated_selection_map
    
    if not visualization_files:
        logger.error("No visualization files found!")
        return ""
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        frame_count = 0
        current_selection_img = None
        current_selection_step = None
        
        for step_num, vis_filename in visualization_files:
            vis_path = os.path.join(visualization_dir, vis_filename)
            vis_img = cv2.imread(vis_path)
            
            if vis_img is None:
                continue
            
            # Check if there's a selection for this step
            if step_num in selection_map:
                selection_filename = selection_map[step_num]
                selection_path = os.path.join(selection_dir, selection_filename)
                selection_img = cv2.imread(selection_path)
                
                if selection_img is not None:
                    current_selection_img = selection_img
                    current_selection_step = step_num
                    logger.info(f"Found selection for step {step_num}: {selection_filename}")
            
            # Create combined frame
            combined_frame = create_combined_frame(vis_img, current_selection_img)
            
            # Determine frame duration
            if current_selection_img is not None and current_selection_step == step_num:
                # Show for 3 seconds (6 frames at 2 FPS)
                frame_duration = 6
                logger.info(f"Step {step_num}: Showing with selection for 3 seconds")
            else:
                # Show for 0.5 seconds (1 frame at 2 FPS)
                frame_duration = 1
                logger.info(f"Step {step_num}: Showing visualization for 0.5 seconds")
            
            # Create frames for this step
            for i in range(frame_duration):
                frame_filename = f"frame_{frame_count:06d}.png"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, combined_frame)
                frame_count += 1
        
        # Use FFmpeg to create video
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-framerate', '2',  # 2 FPS
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"FFmpeg video created successfully: {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return ""
            
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description="Create visualization video from MZSON results")
    parser.add_argument("--scene_id", type=str, required=True, 
                       help="Scene ID (e.g., 00848-ziup5kvtCCR)")
    parser.add_argument("--subtask", type=int, default=0, 
                       help="Subtask number (0-9, default: 0)")
    parser.add_argument("--base_dir", type=str, default="results/mzson_cfg_3dmem/0831",
                       help="Base directory containing results")
    parser.add_argument("--use_ffmpeg", action="store_true",
                       help="Use FFmpeg instead of OpenCV for video creation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.scene_id:
        logger.error("Scene ID is required!")
        return
    
    if args.subtask < 0 or args.subtask > 9:
        logger.error("Subtask must be between 0 and 9!")
        return
    
    # Create video
    if args.use_ffmpeg:
        output_path = create_video_with_ffmpeg(args.scene_id, args.subtask, args.base_dir)
    else:
        output_path = create_video(args.scene_id, args.subtask, args.base_dir)
    
    if output_path:
        logger.info(f"✅ Video created successfully: {output_path}")
    else:
        logger.error("❌ Failed to create video")


if __name__ == "__main__":
    main()
