# Facial Vision

A backend system for detecting and recognizing faces in images and videos, outputting cropped faces with metadata.

## Features

- Face detection in images and videos
- Automatic face cropping and metadata extraction
- **GPS coordinate extraction from video overlays using OCR** - Reads GPS coordinates directly from camera overlay text (e.g., AXIS W120 body cameras)
- Frame-specific GPS extraction for videos - Captures GPS coordinates at the exact moment each face is detected
- GPS and timestamp extraction from media files (supports GPMF format for GoPro/action cameras and standard MP4 metadata)
- Face clustering to group similar faces across frames
- Batch processing capabilities
- **Blockchain provenance tracking** - Complete asset lineage via Maverix/FireFly
- **IPFS storage** - Distributed storage for face chip bundles
- **Video chip bundle generation** - Clean file paths with frame info in metadata
- Support for multiple face detection models

## Project Structure

```
facial-vision/
├── src/           # Core application code
├── tests/         # Test suite
├── config/        # Configuration files
├── data/          # Input/output directories
├── scripts/       # Utility scripts
└── docs/          # Documentation
```

## Setup

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment support

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yomi-adamo/ai-recognition-system-v2.git
cd ai-recognition-system-v2
```

2. Create a virtual environment:
```bash
# For systems with python3-venv installed:
python3 -m venv venv

# Or install python3-venv first:
sudo apt-get update
sudo apt-get install python3-venv

# Activate the virtual environment:
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPS extraction from videos:
```bash
# Install Tesseract OCR for reading GPS overlays
sudo apt-get install tesseract-ocr  # On Ubuntu/Debian
# or
brew install tesseract  # On macOS

# Install ffprobe for video metadata extraction
sudo apt-get install ffmpeg  # On Ubuntu/Debian
# or
brew install ffmpeg  # On macOS

# Install GPMF parser for GoPro videos (optional)
pip install gpmf
```

## Quick Start Commands

### Environment Setup
```bash
# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore"

# Activate virtual environment
source venv/bin/activate
```

### Test Commands
```bash
# Test image processing with blockchain
python test_full_pipeline_blockchain.py data/input/Photos/people-collage-design.jpg

# Test video processing with blockchain  
python test_full_pipeline_blockchain.py data/input/test1.mp4

# Test large video file
python test_full_pipeline_blockchain.py data/input/test_stroll.mp4

# Debug clustering on image directory
python debug_clustering.py data/input/Photos
```

### Face Clustering Tests
```bash
# Test clustering accuracy on people-collage-design.jpg
python test_clustering_accuracy.py

# Test clustering with additional images
python test_clustering_append.py --image data/input/Photos/group.jpg --test-dir threshold_test_0.95

# Test clustering with videos
python test_clustering_append_video.py --video data/input/Videos/test1.mp4 --test-dir threshold_test_0.95

# Test multiple threshold values for clustering optimization
python test_threshold_analysis.py

# Batch test multiple images
python test_clustering_batch.py

# Test glasses tolerance (same person with/without glasses)
python test_glasses_tolerance.py
```

### Download Chip Bundles
```bash
# Get asset ID from test output, then download:
curl "http://localhost:3000/provenance/assets/{ASSET_ID}/download" -o chips.zip

# Extract and verify
unzip chips.zip
```

### Useful Directories
```bash
# Clear previous test results  
rm -rf test_output/*

# Check generated chips
ls -la test_output/person_*/

# View clustering results
cat test_output/manifest.json
```

## Usage

### Process a single image:
```bash
python scripts/process_folder.py --input data/input/image.jpg
```

### Process a video with GPS extraction:
```bash
# Basic video processing
python scripts/process_video.py data/input/Videos/video.mp4 --output-dir data/output

# Process video with GPS overlay extraction (e.g., AXIS W120 body camera)
python scripts/process_video.py data/input/Videos/location_test.mp4 --frame-interval 30 --output-dir data/output/gps_test
```

### Process a folder of images:
```bash
python scripts/process_folder.py --input data/input/
```

## Configuration

Edit `config/default.yaml` to customize:
- Face detection models and parameters
- Video processing settings
- Output formats and quality
- IPFS and blockchain endpoints

## Output Format

The system outputs JSON files with the following structure:

### For Images:
```json
{
    "file": "base64_encoded_image_or_path",
    "type": "image",
    "name": "face_chip_timestamp",
    "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "gps": {"lat": 40.7128, "lon": -74.0060},
        "confidence": 0.95,
        "source_file": "original_filename.jpg",
        "face_bounds": {"x": 100, "y": 150, "w": 200, "h": 200}
    }
}
```

### For Videos (with GPS overlay extraction):
```json
{
    "file": "data/output/gps_test/video_processing_20250624_085532/person_4/chip_001.jpg",
    "type": "image",
    "name": "chip_001",
    "clusterId": "person_4",
    "timestamp": "2024-06-24T08:55:53.436Z",
    "videoTimestamp": "00:00:23.903",
    "frameNumber": 718,
    "gps": {
        "lat": 38.9549,
        "lon": -77.412
    },
    "confidence": 0.8945,
    "face_bounds": {"x": 808, "y": 928, "w": 107, "h": 107}
}
```

**Note**: GPS coordinates are extracted using OCR from camera overlay text. Supported formats include:
- Decimal degrees: `38.9549, -77.4120`
- Degree format: `38.9549° -77.4120°`
- N/S E/W format: `38.9549N 77.4120W`

## Development

See `docs/` for detailed documentation:
- `CLAUDE.md` - Development instructions
- `architecture.md` - System architecture
- `specs.md` - Technical specifications
- `tasks.md` - Task tracking

## License

[Add your license here]
