# Facial Vision

A backend system for detecting and recognizing faces in images and videos, outputting cropped faces with metadata.

## Features

- Face detection in images and videos
- Automatic face cropping and metadata extraction
- GPS and timestamp extraction from media files
- Batch processing capabilities
- IPFS integration for distributed storage
- Blockchain logging via FireFly
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
cd /home/adamoyomi/code/facial-vision
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

## Usage

### Process a single image:
```bash
python scripts/process_folder.py --input data/input/image.jpg
```

### Process a video:
```bash
python scripts/process_video.py --input data/input/video.mp4
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

## Development

See `docs/` for detailed documentation:
- `CLAUDE.md` - Development instructions
- `architecture.md` - System architecture
- `specs.md` - Technical specifications
- `tasks.md` - Task tracking

## License

[Add your license here]