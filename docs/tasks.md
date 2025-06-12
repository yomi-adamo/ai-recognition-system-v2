### tasks.md

# Task Breakdown

## Phase 1: Project Setup & Core Infrastructure

### 1.1 Initial Project Creation
```
Create a new Python project called 'facial-vision' with this folder structure:
- docs/ (CLAUDE.md, specs.md, architecture.md, tasks.md)
- src/core/, src/processors/, src/outputs/, src/utils/
- tests/, config/, data/input/, data/output/, scripts/, bugs/
- Create requirements.txt with: face_recognition, opencv-python, Pillow, numpy, pyyaml, ipfshttpclient, requests, python-dateutil, exifread
- Set up a virtual environment and install all dependencies
```

### 1.2 Configuration System
```
Create config/default.yaml with settings for:
- face_detection (model type, tolerance, min_face_size)
- video_processing (frame_interval, scene_change_threshold)
- output (chip_size, jpeg_quality, use_base64)
- ipfs and blockchain endpoints

Then create src/utils/config.py to load and manage these configurations using a singleton pattern
```

### 1.3 Logging System
```
Implement src/utils/logger.py with:
- Configurable log levels
- File and console output
- Structured logging for JSON output
- Performance timing decorators
Include a setup_logger() function that creates logs in a 'logs/' directory
```

## Phase 2: Core Detection Components

### 2.1 Face Detector
```
Implement src/core/face_detector.py with a FaceDetector class that:
- Uses face_recognition library for detection
- Has methods: detect_faces(image_path) returning list of face locations
- Includes get_face_encodings(image_path, face_locations) for recognition
- Handles multiple faces per image
- Returns confidence scores
- Has error handling for corrupted images
```

### 2.2 Metadata Extractor
```
Create src/core/metadata_extractor.py with MetadataExtractor class that:
- Extracts EXIF data from images (GPS, timestamp)
- Gets file creation/modification time as fallback
- Extracts video metadata using OpenCV
- Returns structured dict with timestamp, GPS coords, camera info
- Handles missing metadata gracefully
```

### 2.3 Face Chip Generator
```
Implement src/core/chip_generator.py with ChipGenerator class that:
- Takes image and face bounding box
- Adds padding around face (configurable)
- Resizes to standard size (224x224 default)
- Maintains aspect ratio
- Saves as JPEG or returns base64 encoded string
- Generates unique filenames using timestamp and UUID
```

## Phase 3: Processing Pipeline

### 3.1 Image Processor
```
Create src/processors/image_processor.py with ImageProcessor class that:
- Combines FaceDetector, MetadataExtractor, and ChipGenerator
- Process single images and return JSON objects
- Handles multiple faces in one image
- Generates the required JSON format with all fields
- Includes error handling and logging
- Add process_image(image_path) method returning list of face JSONs
```

### 3.2 Video Processor
```
Implement src/processors/video_processor.py with VideoProcessor class that:
- Opens videos with OpenCV
- Extracts frames at configurable intervals (every 30 frames default)
- Detects scene changes using frame differencing
- Tracks unique faces across frames using encodings
- Yields face detections with video timestamp
- Handles long videos efficiently with streaming
- Includes progress tracking
```

### 3.3 Batch Processor
```
Create src/processors/batch_processor.py with BatchProcessor class that:
- Recursively scans folders for images and videos
- Uses multiprocessing Pool for parallel processing
- Maintains processing queue and results
- Generates summary statistics
- Handles errors without stopping batch
- Saves results incrementally
- Includes resume capability for interrupted jobs
```

## Phase 4: Output Generation

### 4.1 JSON Formatter
```
Implement src/outputs/json_formatter.py with JSONFormatter class that:
- Creates JSON objects following the exact schema
- Generates unique IDs for faces and parent files
- Formats timestamps in ISO 8601
- Handles base64 encoding of chips
- Validates required fields
- Includes batch_to_json() for multiple faces
- Pretty prints with proper indentation
```

### 4.2 File Handler
```
Create src/utils/file_handler.py with FileHandler class that:
- Manages input/output directories
- Creates timestamped output folders
- Handles file naming conflicts
- Provides methods for saving chips and JSON
- Includes cleanup methods
- Tracks processed files to avoid duplicates
```

## Phase 5: Main Processing Scripts

### 5.1 Single Image Script
```
Create scripts/process_image.py that:
- Takes image path as command line argument
- Uses ImageProcessor to detect faces
- Saves face chips and JSON output
- Prints summary to console
- Includes --output-dir and --base64 options
```

### 5.2 Video Processing Script
```
Create scripts/process_video.py that:
- Takes video path as argument
- Uses VideoProcessor for extraction
- Deduplicates faces across frames
- Saves unique faces with best quality
- Generates timeline JSON with all detections
- Includes progress bar
- Has --frame-interval and --scene-detection options
```

### 5.3 Folder Processing Script
```
Create scripts/process_folder.py that:
- Takes folder path as argument
- Uses BatchProcessor for parallel processing
- Supports --recursive flag
- Filters by file extensions
- Generates summary report
- Has --workers option for parallelism
- Includes --resume option
```

## Phase 6: Integration Components

### 6.1 IPFS Uploader
```
Implement src/outputs/ipfs_uploader.py with IPFSUploader class that:
- Connects to IPFS daemon
- Uploads face chips and returns CIDs
- Uploads JSON metadata
- Handles connection errors with retry
- Includes batch upload method
- Pins important content
- Returns IPFS URLs
```

### 6.2 Blockchain Logger
```
Create src/outputs/blockchain_logger.py with BlockchainLogger class that:
- Connects to FireFly API
- Creates data messages with face detection events
- Includes face CID and metadata
- Handles batch submissions
- Implements retry logic
- Tracks transaction hashes
```

## Phase 7: Testing & Validation

### 7.1 Core Tests
```
Create comprehensive tests in tests/ directory:
- test_face_detection.py: Test detection accuracy
- test_metadata_extraction.py: Test EXIF parsing
- test_json_format.py: Validate JSON schema
- test_video_processing.py: Test frame extraction
- Add sample images and videos in tests/fixtures/
```

### 7.2 Integration Tests
```
Create tests/test_integration.py that:
- Tests full pipeline from image to JSON
- Validates IPFS uploads
- Tests batch processing
- Measures performance
- Checks memory usage
```

## Phase 8: Documentation & Final Setup

### 8.1 Documentation
```
Complete all documentation files:
- Update README.md with usage examples
- Fill in docs/api.md with module documentation
- Create docs/deployment.md with production setup
- Add docs/performance.md with benchmarks
```

### 8.2 Docker Setup (Optional)
```
Create Dockerfile and docker-compose.yml that:
- Sets up Python environment
- Installs dependencies
- Includes IPFS node
- Configures volumes for data
- Adds environment variables
```




## Phase 9: Run Complete System

### 9.1 Final Test Run
```
Test the complete system:
1. Process a folder of mixed images and videos
2. Verify JSON output format
3. Upload results to IPFS
4. Log to blockchain
5. Generate performance report
```

# Facial Vision - Extensions Timeline & Implementation Guide

## Phase 10: Extensions

### **After Phase 9 (Main Scripts) is Complete**
You should have:
- ✅ Working face detection pipeline
- ✅ JSON output format established
- ✅ Batch processing capability
- ✅ Video frame extraction working

This is the ideal time because:
1. Your core architecture is proven
2. You can reuse the same processing pipeline
3. The JSON schema can be extended easily
4. You know the system performs well

## Extension Implementation Order & Prompts

### 10.1 **License Plate OCR** (Easiest - Start Here)
**Why first**: Uses similar detection + extraction pattern as faces

**Claude Code Prompt**:
```
Create src/core/license_plate_detector.py that:
- Uses OpenCV and pytesseract for plate detection
- Implements LicensePlateDetector class similar to FaceDetector
- Detects plate regions using cascade classifiers or YOLO
- Extracts text using OCR
- Returns plate location, extracted text, and confidence
- Handles multiple plates per image
Add pytesseract and opencv-contrib-python to requirements.txt
```

**Follow-up Integration**:
```
Update src/processors/image_processor.py to:
- Add license plate detection alongside face detection
- Generate JSON with type: "license_plate"
- Include OCR text in metadata.extracted_text field
- Save cropped plate images as chips
```

### 10.2 **Logo Recognition** (Second - Builds on Detection)
**Why second**: Similar pattern but needs a logo database

**Claude Code Prompt**:
```
Create src/core/logo_detector.py with LogoDetector class that:
- Uses SIFT/ORB features for logo matching
- Loads reference logo database from data/logos/
- Implements detect_logos(image_path) method
- Matches against known logos (Nike, Adidas, etc.)
- Returns logo bounding box, brand name, confidence
- Handles multiple logos per image
Include a setup script to download common logo datasets
```

**Database Setup**:
```
Create scripts/setup_logos.py that:
- Downloads common brand logos
- Processes them into feature descriptors
- Saves as pickle file for fast loading
- Includes top 100 brand logos
```

### 10.3 **Vehicle Detection** (Third - More Complex)
**Why third**: Requires additional model for make/model classification

**Claude Code Prompt**:
```
Implement src/core/vehicle_detector.py that:
- Uses YOLOv5 or similar for vehicle detection
- Detects cars, trucks, motorcycles
- Extracts vehicle chips for classification
- Uses a pre-trained model to identify make/model
- Detects color using dominant color analysis
- Returns: type, make, model, color, confidence, bounding box
Add ultralytics and webcolors to requirements.txt
```

**Model Download**:
```
Create scripts/download_models.py that:
- Downloads YOLOv5 weights for vehicle detection
- Downloads vehicle make/model classifier
- Saves to data/models/ directory
- Verifies model integrity
```

### 10.4 **Tattoo Detection** (Fourth - Requires Custom Training)
**Why fourth**: Most challenging, needs specialized model

**Claude Code Prompt**:
```
Create src/core/tattoo_detector.py with TattooDetector class that:
- Uses a CNN-based detector (modify face detector approach)
- Detects exposed skin regions first
- Identifies tattoo regions within skin areas
- Extracts tattoo chips
- Optionally classifies tattoo style/content
- Returns tattoo locations and descriptions
Consider using detectron2 or similar framework
```

**Training Data Prep**:
```
Create scripts/prepare_tattoo_dataset.py that:
- Processes tattoo datasets
- Augments training data
- Generates annotations
- Splits into train/val/test
Note: You'll need to obtain a tattoo dataset separately
```

### 10.5 **Audio Extraction & Transcription** (Last - Different Pipeline)
**Why last**: Completely different processing pipeline

**Claude Code Prompt**:
```
Create src/processors/audio_processor.py with AudioProcessor class that:
- Extracts audio from video files using ffmpeg-python
- Splits audio into chunks for processing
- Uses OpenAI Whisper for transcription
- Detects speaker changes
- Generates timestamped transcripts
- Returns JSON with transcript segments
Add openai-whisper and ffmpeg-python to requirements.txt
```

**Integration Prompt**:
```
Update src/processors/video_processor.py to:
- Optionally process audio alongside video
- Sync transcripts with face detections
- Add speaker diarization
- Link voices to detected faces when possible
```

## Modified Project Structure for Extensions

```
src/
├── core/
│   ├── face_detector.py
│   ├── license_plate_detector.py    # NEW
│   ├── logo_detector.py              # NEW
│   ├── vehicle_detector.py           # NEW
│   ├── tattoo_detector.py            # NEW
│   └── base_detector.py              # NEW - Abstract base class
├── processors/
│   ├── image_processor.py            # UPDATED
│   ├── video_processor.py            # UPDATED
│   ├── audio_processor.py            # NEW
│   └── multi_detector_processor.py   # NEW - Combines all detectors
```

## Unified Processing Prompt

After implementing 2-3 extensions:

```
Create src/processors/multi_detector_processor.py that:
- Combines all available detectors
- Processes images/videos through all detectors
- Merges results into unified JSON output
- Handles overlapping detections
- Optimizes by sharing preprocessing steps
- Includes enable/disable flags for each detector
- Maintains performance with parallel processing
```

## JSON Schema Extension

Update your JSON output to handle multiple detection types:

```json
{
    "file": "base64_or_path",
    "type": "face|license_plate|logo|vehicle|tattoo|transcript",
    "name": "detection_<type>_<timestamp>",
    "author": "facial-vision-system",
    "parentId": "source_file_id",
    "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "gps": {"lat": 40.7128, "lon": -74.0060},
        "confidence": 0.95,
        
        // Type-specific fields:
        // For faces:
        "identity": "person_name_or_unknown",
        
        // For license plates:
        "extracted_text": "ABC-1234",
        "plate_region": "US-NY",
        
        // For vehicles:
        "make": "Toyota",
        "model": "Camry",
        "year_range": "2018-2020",
        "color": "silver",
        
        // For logos:
        "brand": "Nike",
        "product_category": "apparel",
        
        // For tattoos:
        "body_location": "left_arm",
        "style": "tribal",
        
        // For transcripts:
        "speaker_id": "speaker_1",
        "text": "transcribed text here",
        "start_time": 10.5,
        "end_time": 15.3
    },
    "topics": ["face", "biometric", "person"] // Varies by type
}
```

## Performance Optimization Prompt

After adding all extensions:

```
Optimize the multi-detector system:
- Implement shared image preprocessing
- Use GPU acceleration where available
- Add detector priority queues
- Cache detection results
- Implement batch processing per detector type
- Add performance profiling
- Create a detector scheduler for resource management
```

## Testing Extensions

```
Create tests/test_extensions.py that:
- Tests each detector individually
- Tests multi-detector integration
- Validates JSON output for each type
- Measures performance impact
- Tests error handling when models are missing
```

## Configuration Update

```
Update config/default.yaml to include:

detectors:
  face:
    enabled: true
    model: "hog"
    min_confidence: 0.6
  
  license_plate:
    enabled: true
    ocr_language: "eng"
    min_confidence: 0.7
    
  vehicle:
    enabled: true
    detect_make_model: true
    detect_color: true
    
  logo:
    enabled: true
    logo_db_path: "data/logos/database.pkl"
    match_threshold: 0.8
    
  tattoo:
    enabled: false  # Until model is trained
    model_path: "data/models/tattoo_detector.pth"
    
  audio:
    enabled: true
    model: "whisper-base"
    language: "en"
```

## Tips for Extension Development

1. **Test each extension in isolation first**
2. **Reuse the base detection pattern** - All visual detectors follow similar flow
3. **Start with pre-trained models** - Fine-tune later if needed
4. **Keep the same JSON structure** - Just add type-specific metadata
5. **Monitor performance** - Some detectors are much slower than others
6. **Make detectors optional** - Not every use case needs all detectors

## Common Integration Pattern

For each new detector:
1. Create detector class in `src/core/`
2. Update `image_processor.py` to call new detector
3. Extend JSON metadata fields
4. Add configuration options
5. Create tests
6. Update batch processor

## Execution Order Summary

Run these prompts in order:
1. Phase 1: All setup prompts (1.1 - 1.3)
2. Phase 2: Core components (2.1 - 2.3)
3. Phase 3: Processors (3.1 - 3.3)
4. Phase 4: Output generation (4.1 - 4.2)
5. Phase 5: Main scripts (5.1 - 5.3)
6. Phase 6: Integration (6.1 - 6.2)
7. Phase 7: Testing (7.1 - 7.2)
8. Phase 8: Documentation (8.1 - 8.2)
9. Phase 9: Final test (9.1)
10. Phase 10: Extentions (10.1 - 10.5)


## Quick Test After Each Phase

After implementing each phase, test with:
```
# After Phase 2:
python -c "from src.core.face_detector import FaceDetector; fd = FaceDetector(); print(fd.detect_faces('test_image.jpg'))"

# After Phase 3:
python scripts/process_image.py test_image.jpg --output-dir ./output

# After Phase 5:
python scripts/process_folder.py ./data/input --recursive --workers 4
```

## Troubleshooting Prompts

If you encounter issues:

```
"Debug the face detection error in [file]. Add detailed logging and check if the image is being loaded correctly"

"The JSON output is missing required fields. Update the JSONFormatter to ensure all fields from the schema are included"

"Video processing is running out of memory. Implement frame buffering and garbage collection in VideoProcessor"

"IPFS upload is failing. Add connection testing and implement exponential backoff retry logic"
```