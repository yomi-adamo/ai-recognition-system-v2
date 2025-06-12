# Claude Code Instructions

## Project Overview
Facial Vision is a backend system for detecting and recognizing faces in images/videos, outputting cropped faces with metadata.

## Project Structure

facial-vision/
├── docs/`
│   ├── CLAUDE.md           # Claude Code instructions and prompts
│   ├── specs.md            # Technical specifications
│   ├── architecture.md     # System architecture design
│   ├── tasks.md            # Task breakdown and progress tracking
│   └── api.md              # API documentation
├── src/
│   ├── core/
│   │   ├── face_detector.py      # Face detection logic
│   │   ├── face_recognizer.py    # Face recognition/matching
│   │   ├── metadata_extractor.py # Extract GPS, timestamp from files
│   │   └── chip_generator.py     # Face cropping logic
│   ├── processors/
│   │   ├── image_processor.py    # Single image processing
│   │   ├── video_processor.py    # Video chunking and processing
│   │   └── batch_processor.py    # Batch operations
│   ├── outputs/
│   │   ├── json_formatter.py     # JSON output generation
│   │   ├── ipfs_uploader.py      # IPFS integration
│   │   └── blockchain_logger.py  # FireFly integration
│   └── utils/
│       ├── file_handler.py       # File I/O operations
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging utilities
├── tests/
│   ├── test_face_detection.py
│   ├── test_video_processing.py
│   └── test_output_format.py
├── config/
│   ├── default.yaml              # Default configuration
│   └── models.yaml               # Model configurations
├── data/
│   ├── input/                    # Input images/videos
│   ├── output/                   # Output JSON and chips
│   └── models/                   # Pre-trained models
├── scripts/
│   ├── setup.py                  # Environment setup
│   ├── process_folder.py         # Process entire folders
│   └── process_video.py          # Process single video
├── bugs/
│   ├── known_issues.md           # Known bugs and limitations
│   └── bug_reports/              # Individual bug reports
├── requirements.txt
├── README.md
└── .gitignore

## Key Commands

### Initial Setup
"Set up the Facial Vision project with face detection capabilities using face_recognition library and OpenCV for video processing"

### Face Detection Tasks
"Implement face detection for the image at [path] and generate JSON output with cropped faces"

"Process all images in the input folder and generate face chips with metadata"

### Video Processing
"Implement intelligent video chunking that processes every 30th frame and detects scene changes"

"Process the video file and extract all unique faces with timestamps"

### Testing
"Test the face detection pipeline with sample images and verify JSON output format"

### Integration
"Implement IPFS upload for face chips and metadata"

"Add FireFly blockchain logging for processed faces"

## Code Patterns

### Standard JSON Output
~~~python
{
    "file": "base64_encoded_image_or_path",
    "type": "image",
    "name": f"face_chip_{timestamp}",
    "author": "facial-vision-system",
    "parentId": "original_file_id",
    "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "gps": {"lat": 40.7128, "lon": -74.0060},
        "confidence": 0.95,
        "identity": "unknown",
        "source_file": "original_filename.jpg",
        "face_bounds": {"x": 100, "y": 150, "w": 200, "h": 200}
    },
    "topics": ["face", "biometric", "person"]
}
~~~

## 4. Additional Components
Configuration File (config/default.yaml)

~~~yaml
face_detection:
  model: "hog"  # or "cnn" for better accuracy
  tolerance: 0.6
  min_face_size: 40
  
video_processing:
  frame_interval: 30
  scene_change_threshold: 30.0
  max_faces_per_frame: 20

output:
  chip_size: [224, 224]
  jpeg_quality: 85
  use_base64: true
  
ipfs:
  api_endpoint: "http://localhost:5001"
  timeout: 30

blockchain:
  firefly_endpoint: "http://localhost:5000"
  namespace: "facial-vision"
~~~

## Requirements.txt
face_recognition==1.3.0
opencv-python==4.8.1
Pillow==10.1.0
numpy==1.24.3
pyyaml==6.0.1
ipfshttpclient==0.8.0
requests==2.31.0
python-dateutil==2.8.2
exifread==3.0.0
deepface==0.0.79  # Alternative to face_recognition
mtcnn==0.1.1      # For better face detection

## 5. Scalability Considerations

Parallel Processing: Use multiprocessing for batch operations
Caching: Cache face encodings for known identities
Chunking: Process large videos in chunks to manage memory
Queue System: Consider adding Redis/RabbitMQ for distributed processing
Storage: Use object storage (S3-compatible) for chips
Database: PostgreSQL with pgvector for face embedding search

Next Steps

Start with basic face detection implementation
Test with sample images
Add video processing capabilities
Implement IPFS integration
Add blockchain logging
Optimize for performance
Add additional detection features

## Common Issues & Solutions

- If face detection is slow, reduce image resolution before processing
- For video processing, use keyframe extraction instead of every frame
- Cache face encodings for known identities to speed up recognition