#!/bin/bash

# Visualization Video Creator Script
# Creates MP4 videos from MZSON visualization and selection images

# Default values
# SCENE_ID="00848-ziup5kvtCCR"
SCENE_ID="00871-VBzV5z6i1WS"
SUBTASK=0
# BASE_DIR="results/mzson_cfg_3dmem/0831"
BASE_DIR="results/mzson_cfg_3dmem/0831_nomem"
USE_FFMPEG=false

# Function to display usage
usage() {
    echo "Usage: $0 --scene_id <SCENE_ID> [--subtask <SUBTASK>] [--base_dir <BASE_DIR>] [--use_ffmpeg]"
    echo ""
    echo "Arguments:"
    echo "  --scene_id    Scene ID (required, e.g., 00848-ziup5kvtCCR)"
    echo "  --subtask     Subtask number (0-9, default: 0)"
    echo "  --base_dir    Base directory containing results (default: results/mzson_cfg_3dmem/0831)"
    echo "  --use_ffmpeg  Use FFmpeg instead of OpenCV (recommended if video is corrupted)"
    echo ""
    echo "Examples:"
    echo "  $0 --scene_id 00848-ziup5kvtCCR"
    echo "  $0 --scene_id 00848-ziup5kvtCCR --subtask 2"
    echo "  $0 --scene_id 00871-VBzV5z6i1WS --base_dir results/mzson_cfg_3dmem/0831_nomem"
    echo "  $0 --scene_id 00848-ziup5kvtCCR --use_ffmpeg"
    echo ""
    echo "Layout: Selection (2/3) + Visualization (1/3)"
    echo "Timing: Visualization 0.5s, Selection 3.0s"
    echo ""
    echo "Note: If you get corrupted/green video, try --use_ffmpeg option"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene_id)
            SCENE_ID="$2"
            shift 2
            ;;
        --subtask)
            SUBTASK="$2"
            shift 2
            ;;
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --use_ffmpeg)
            USE_FFMPEG=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SCENE_ID" ]]; then
    echo "Error: --scene_id is required!"
    echo ""
    usage
    exit 1
fi

# Validate subtask range
if [[ $SUBTASK -lt 0 || $SUBTASK -gt 9 ]]; then
    echo "Error: Subtask must be between 0 and 9!"
    echo ""
    usage
    exit 1
fi

# Check if base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
fi

# Check if scene directory exists
SCENE_DIR="$BASE_DIR/$SCENE_ID"
if [[ ! -d "$SCENE_DIR" ]]; then
    echo "Error: Scene directory does not exist: $SCENE_DIR"
    exit 1
fi

# Check if subtask directory exists
SUBTASK_DIR="$SCENE_DIR/subtask_${SCENE_ID}_0_${SUBTASK}"
if [[ ! -d "$SUBTASK_DIR" ]]; then
    echo "Error: Subtask directory does not exist: $SUBTASK_DIR"
    exit 1
fi

# Check if visualization directory exists
VIS_DIR="$SUBTASK_DIR/visualization"
if [[ ! -d "$VIS_DIR" ]]; then
    echo "Error: Visualization directory does not exist: $VIS_DIR"
    exit 1
fi

# Display configuration
echo "=========================================="
echo "🎬 Visualization Video Creator"
echo "=========================================="
echo "Scene ID:     $SCENE_ID"
echo "Subtask:      $SUBTASK"
echo "Base Dir:     $BASE_DIR"
echo "Scene Dir:    $SCENE_DIR"
echo "Subtask Dir:  $SUBTASK_DIR"
echo "Vis Dir:      $VIS_DIR"
echo "Layout:       Selection (2/3) + Visualization (1/3)"
echo "Timing:       Visualization 0.5s, Selection 3.0s"
echo "Method:       $([ "$USE_FFMPEG" = true ] && echo "FFmpeg" || echo "OpenCV")"
echo "=========================================="

# Check if Python script exists
SCRIPT_PATH="create_visualization_video.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# Run the Python script
echo "🚀 Creating video..."
if [ "$USE_FFMPEG" = true ]; then
    python3 "$SCRIPT_PATH" \
        --scene_id "$SCENE_ID" \
        --subtask "$SUBTASK" \
        --base_dir "$BASE_DIR" \
        --use_ffmpeg
else
    python3 "$SCRIPT_PATH" \
        --scene_id "$SCENE_ID" \
        --subtask "$SUBTASK" \
        --base_dir "$BASE_DIR"
fi

# Check if video was created successfully
if [ "$USE_FFMPEG" = true ]; then
    OUTPUT_VIDEO="$SUBTASK_DIR/${SCENE_ID}_0_${SUBTASK}_visualization_ffmpeg.mp4"
else
    OUTPUT_VIDEO="$SUBTASK_DIR/${SCENE_ID}_0_${SUBTASK}_visualization.mp4"
fi
if [[ -f "$OUTPUT_VIDEO" ]]; then
    echo ""
    echo "✅ Video created successfully!"
    echo "📁 Output: $OUTPUT_VIDEO"
    
    # Get file size
    FILE_SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
    echo "📊 File size: $FILE_SIZE"
    
    # Count visualization and selection files
    VIS_COUNT=$(find "$VIS_DIR" -name "*.png" | wc -l)
    SEL_COUNT=$(find "$SUBTASK_DIR/selection" -name "*.png" 2>/dev/null | wc -l)
    echo "📈 Visualization files: $VIS_COUNT"
    echo "📈 Selection files: $SEL_COUNT"
    
    echo ""
    echo "🎥 You can now play the video with:"
    echo "   vlc \"$OUTPUT_VIDEO\""
    echo "   or any other video player"
else
    echo ""
    echo "❌ Failed to create video!"
    echo "Check the error messages above for details."
    exit 1
fi
