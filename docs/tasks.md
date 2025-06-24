### tasks.md

⚠️ NOTE: This file is from the previous implementation and may conflict with CLAUDE.md. Refer to CLAUDE.md for the current design goals.

# TASKS — Facial Vision System

> Use this phased task list to guide development of the facial recognition and provenance tracking system.

---

## ✅ Phase 1: Core Infrastructure Setup

- [ ] Set up project structure and dependencies
- [ ] Implement configuration management (`config.py`)
- [ ] Create logging framework (`logger.py`)
- [ ] Set up file handling utilities
- [ ] Create base processor classes

---

## 🔍 Phase 2: Face Detection & Clustering

- [ ] Implement `face_detector.py` with MTCNN / `face_recognition`
- [ ] Create `FaceEmbeddingExtractor` in `face_clusterer.py`
- [ ] Implement `ClusterManager` with HDBSCAN
- [ ] Build `ClusterRegistry` for tracking clusters
- [ ] Enhance `chip_generator.py` with:
  - [ ] Clustered folder structure
  - [ ] Quality checks (blur detection, resolution)
- [ ] Write `test_clustering.py` test suite

---

## 🧠 Phase 3: Metadata & Processing

- [ ] Enhance `metadata_extractor.py` for:
  - [ ] Video metadata
  - [ ] EXIF extraction
  - [ ] GPS and device ID parsing
- [ ] Implement `image_processor.py`
- [ ] Implement `video_processor.py` with frame extraction
- [ ] Create `batch_processor.py` for bulk operations
- [ ] Implement JSON metadata output formatter

---

## 🔗 Phase 4: Blockchain Integration

- [ ] Create `blockchain_integration.py` module
- [ ] Implement asset existence checking (SHA-256 & filename/author)
- [ ] Build root asset upload functionality
- [ ] Implement derived asset creation logic
- [ ] Create IPFS chip bundle uploader
- [ ] Implement `blockchain_logger.py` (optional auditing)
- [ ] Write `test_blockchain_integration.py` suite

---

## 🧩 Phase 5: Extended Detection Modules

### 5.1 Architecture Overview

**Goal**: Extend the facial recognition system to support multi-modal detection (tattoos, license plates, vehicles, audio transcription) with unified blockchain integration.

**Design Philosophy**:
- Each detection type creates a separate derived asset linked to the original video/image
- Consistent metadata structure across all detection types
- Unified processing pipeline with modular detector components
- CPU and GPU implementation paths for different deployment scenarios

### 5.2 Core Infrastructure

- [ ] **Create `multi_detector_manager.py`** - Central coordinator for all detection modules
  - Manages detector registration and lifecycle
  - Handles unified configuration and resource allocation
  - Provides common interfaces for all detection types
  - Implements parallel processing for multiple detectors

- [ ] **Enhance `blockchain_integration.py`** - Multi-modal asset management
  - Add `upload_multi_detection_results()` method
  - Implement detection-specific asset creation
  - Create unified query interface for multi-modal results
  - Add batch processing for multiple detection types

- [ ] **Create `detection_base.py`** - Abstract base class for all detectors
  - Standardized interface for detection implementations
  - Common confidence scoring and result formatting
  - Shared preprocessing and postprocessing utilities
  - Error handling and retry mechanisms

### 5.3 Detection Modules Implementation

#### 5.3.1 Tattoo Detection (`tattoo_detector.py`)

**CPU Implementation**:
- Use OpenCV + traditional computer vision techniques
- Implement skin tone detection with HSV color space
- Apply edge detection and contour analysis for tattoo patterns
- Use SIFT/ORB for pattern matching against known tattoo designs
- Classify tattoo types: tribal, text, portrait, geometric, etc.

**GPU Implementation**:
- Use YOLOv8 or Detectron2 for tattoo detection
- Custom training on tattoo datasets (TattooNet, ink detection datasets)
- Implement tattoo style classification with CNN
- Use semantic segmentation for precise tattoo boundaries

**Metadata Structure**:
```json
{
  "detectionType": "tattoo",
  "tattoos": [
    {
      "boundingBox": {"x": 150, "y": 200, "w": 80, "h": 120},
      "bodyLocation": "right_arm",
      "style": "tribal",
      "confidence": 0.85,
      "chipFile": "tattoos/tattoo_001.jpg",
      "timestamp": "2024-01-15T13:24:05Z",
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "frameNumber": 725,
      "videoTimestamp": "00:00:24.200"
    }
  ]
}
```

#### 5.3.2 License Plate Detection (`license_plate_detector.py`)

**CPU Implementation**:
- OpenCV for license plate detection with Haar cascades
- Tesseract OCR for text extraction
- Regex validation for license plate formats
- Geometric correction for skewed plates

**GPU Implementation**:
- YOLOv8 for license plate detection
- EasyOCR or PaddleOCR for GPU-accelerated text recognition
- Deep learning-based text correction models
- Multi-language support for international plates

**Metadata Structure**:
```json
{
  "detectionType": "license_plate",
  "plates": [
    {
      "boundingBox": {"x": 300, "y": 450, "w": 120, "h": 40},
      "plateNumber": "ABC-1234",
      "state": "MD",
      "confidence": 0.92,
      "chipFile": "plates/plate_001.jpg",
      "timestamp": "2024-01-15T13:24:08Z",
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "frameNumber": 795,
      "videoTimestamp": "00:00:26.500"
    }
  ]
}
```

#### 5.3.3 Vehicle Detection (`vehicle_detector.py`)

**CPU Implementation**:
- Haar cascade classifiers for vehicle detection
- Color histogram analysis for dominant vehicle colors
- Template matching for vehicle type classification
- Geometric analysis for vehicle orientation

**GPU Implementation**:
- YOLOv8 for vehicle detection and classification
- ResNet-based make/model classifier
- Semantic segmentation for precise vehicle boundaries
- Advanced color analysis with deep learning

**Metadata Structure**:
```json
{
  "detectionType": "vehicle",
  "vehicles": [
    {
      "boundingBox": {"x": 200, "y": 300, "w": 250, "h": 150},
      "vehicleType": "sedan",
      "make": "Toyota",
      "model": "Camry",
      "color": "blue",
      "confidence": 0.78,
      "chipFile": "vehicles/vehicle_001.jpg",
      "timestamp": "2024-01-15T13:24:10Z",
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "frameNumber": 825,
      "videoTimestamp": "00:00:27.500"
    }
  ]
}
```

#### 5.3.4 Audio Transcription (`audio_transcriber.py`)

**CPU Implementation**:
- Use SpeechRecognition library with Google/Sphinx backends
- Basic speaker separation with voice activity detection
- Simple noise reduction with librosa
- Timestamp alignment with video frames

**GPU Implementation**:
- OpenAI Whisper for high-quality transcription
- pyannote.audio for speaker diarization
- Advanced noise reduction with RNNoise
- Real-time processing with CUDA acceleration

**Metadata Structure**:
```json
{
  "detectionType": "audio_transcription",
  "segments": [
    {
      "startTime": "00:01:23.450",
      "endTime": "00:01:27.200",
      "text": "Suspect is heading north on Main Street",
      "speaker": "speaker_1",
      "confidence": 0.95,
      "language": "en-US",
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "startFrameNumber": 2503,
      "endFrameNumber": 2616
    }
  ]
}
```

### 5.4 Person Correlation and Profile System

- [ ] **Create `person_correlator.py`** - Cross-modal person association system
  - Associate detections across different modalities (face + tattoo + voice)
  - Spatial-temporal correlation analysis
  - Confidence-based association scoring
  - Person profile generation and management

- [ ] **Implement `correlation_manager.py`** - Unified person profile management
  - Generate comprehensive person profiles (person_1, person_2, etc.)
  - Track all associated detections per person across video timeline
  - Create person-specific IPFS bundles with all related evidence
  - Support queryable person profiles through blockchain interface

- [ ] **Person Profile Structure**:
```json
{
  "personId": "person_1",
  "profileSummary": {
    "totalAppearances": 15,
    "timeRange": {"start": "00:00:05.000", "end": "00:03:45.000"},
    "associatedDetections": {
      "faces": 10,
      "tattoos": 2,
      "vehicles": 1,
      "audioSegments": 8
    },
    "representativeImage": "faces/person_1/chip_001.jpg"
  },
  "detectionHistory": [
    {
      "timestamp": "2024-01-15T13:24:00Z",
      "videoTimestamp": "00:00:24.000",
      "frameNumber": 720,
      "detectionTypes": ["face"],
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "files": ["faces/person_1/chip_001.jpg"]
    },
    {
      "timestamp": "2024-01-15T13:24:05Z",
      "videoTimestamp": "00:00:24.200",
      "frameNumber": 725,
      "detectionTypes": ["face", "tattoo"],
      "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112},
      "files": ["faces/person_1/chip_002.jpg", "tattoos/tattoo_1/tattoo_001.jpg"],
      "correlations": {"tattoo_1": "right_arm_tribal"}
    }
  ],
  "audioProfile": {
    "speakerId": "speaker_1",
    "totalSpeechDuration": "00:02:15.000",
    "keyPhrases": ["suspect", "heading north", "Main Street"],
    "voiceSegments": [
      {
        "startTime": "00:01:23.450",
        "endTime": "00:01:27.200",
        "text": "Suspect is heading north on Main Street",
        "gpsCoordinates": {"lat": 39.2557, "lon": -76.7112}
      }
    ]
  },
  "vehicleAssociations": [
    {
      "vehicleId": "vehicle_1",
      "plateNumber": "ABC-1234",
      "associationConfidence": 0.85,
      "timeRange": {"start": "00:02:30.000", "end": "00:02:45.000"}
    }
  ]
}
```

### 5.5 Unified Processing Pipeline

- [ ] **Create `unified_processor.py`** - Orchestrates all detection modules
  - Configurable detector selection (enable/disable specific modules)
  - Parallel processing with resource management
  - Progress tracking and status reporting
  - Error handling and recovery mechanisms

- [ ] **Implement `batch_multi_detector.py`** - Batch processing for multiple files
  - Queue management for large video sets
  - Priority-based processing
  - Resume capability for interrupted processing
  - Resource monitoring and optimization

- [ ] **Create `detection_config.py`** - Configuration management
  - Detector-specific configuration parameters
  - CPU/GPU resource allocation settings
  - Confidence threshold tuning
  - Output format customization

### 5.6 Integration and Optimization

- [ ] **Enhance `video_processor.py`** - Multi-modal video processing
  - Single-pass video analysis for all detectors
  - Shared frame extraction and preprocessing
  - Memory-efficient processing for large videos
  - Temporal coherence tracking across frames
  - GPS coordinate extraction from video overlay

- [ ] **Create `detection_merger.py`** - Combine results from multiple detectors
  - Spatial correlation between detections
  - Temporal tracking across video frames
  - Cross-modal validation (e.g., face + tattoo association)
  - Confidence score aggregation
  - Person profile correlation and updating

- [ ] **Implement `resource_manager.py`** - GPU/CPU resource optimization
  - Dynamic resource allocation based on available hardware
  - Batch size optimization for GPU processing
  - Memory management for large video files
  - Fallback mechanisms for resource constraints

### 5.7 Person Profile IPFS Integration

- [ ] **Create `profile_bundler.py`** - Person-specific evidence packaging
  - Generate individual IPFS bundles per person (person_1_bundle.zip)
  - Include all associated detection chips and metadata
  - Create person profile summary files
  - Support incremental updates to person bundles

- [ ] **Implement `profile_query_interface.py`** - Person profile access system
  - Blockchain-queryable person profile endpoints
  - Support queries like "show me all evidence for person_1"
  - Timeline-based evidence browsing
  - Cross-video person tracking and correlation

### 5.8 Blockchain Integration Enhancements

- [ ] **Multi-Modal Asset Strategy**
  - Each detection type creates separate derived asset
  - Consistent parent-child relationships
  - Unified topic structure for efficient querying
  - Batch upload optimization for multiple detection results

- [ ] **Enhanced Metadata Standards**
  - Standardized confidence scoring across detection types
  - Consistent timestamp and coordinate formats
  - Unified file naming and organization patterns
  - Cross-detection correlation metadata

- [ ] **Query Interface Improvements**
  - Multi-modal search capabilities
  - Detection-type filtering
  - Temporal range queries
  - Confidence threshold filtering

### 5.9 Testing and Validation

- [ ] **Create `test_multi_detection.py`** - Comprehensive test suite
  - Unit tests for each detection module
  - Integration tests for unified pipeline
  - Performance benchmarking
  - Blockchain integration validation
  - Person correlation accuracy testing

- [ ] **Implement `validation_dataset.py`** - Test data management
  - Curated test datasets for each detection type
  - Ground truth labeling for accuracy validation
  - Performance metrics collection
  - Regression testing automation
  - Person profile validation testing

- [ ] **Create `test_person_correlation.py`** - Person profile testing
  - Cross-modal association accuracy tests
  - Person profile completeness validation
  - IPFS bundle integrity testing
  - Query interface performance testing

### 5.10 Performance Optimization

- [ ] **GPU Optimization**
  - CUDA kernel optimization for batch processing
  - Memory pool management for large datasets
  - Multi-GPU support for parallel processing
  - Mixed precision training for faster inference

- [ ] **CPU Optimization**
  - Multi-threading for parallel detection
  - Memory-mapped file processing
  - Cache optimization for repeated operations
  - Vectorized operations with NumPy

- [ ] **Person Profile Optimization**
  - Efficient correlation algorithm implementation
  - Incremental profile updates
  - Optimized IPFS bundle generation
  - Fast person profile queries

### 5.11 Deployment and Monitoring

- [ ] **Create `deployment_config.py`** - Environment-specific configuration
  - CPU-only deployment for resource-constrained environments
  - GPU-enabled deployment for high-performance scenarios
  - Cloud deployment configurations
  - Edge device optimization

- [ ] **Implement `monitoring_dashboard.py`** - Real-time monitoring
  - Processing speed metrics
  - Detection accuracy tracking
  - Resource utilization monitoring
  - Error rate analysis
  - Person correlation accuracy monitoring

### 5.12 Documentation and Examples

- [ ] **Multi-Detection API Documentation**
  - Detailed API reference for each detection module
  - Configuration examples for different scenarios
  - Performance tuning guidelines
  - Troubleshooting guide
  - Person profile system documentation

- [ ] **Example Implementations**
  - Sample code for each detection type
  - End-to-end processing examples
  - Blockchain integration demonstrations
  - Performance optimization examples
  - Person profile query examples

---

## 🚀 Phase 6: Integration & Testing

- [ ] Create end-to-end integration tests
- [ ] Implement performance benchmarking scripts
- [ ] Build CLI interface for batch processing
- [ ] Write comprehensive documentation
- [ ] Create deployment scripts (Docker, bash) and containerization

---

## Notes

- Each phase builds upon the previous.
- Integration points with FireFly/Maverix blockchain system are covered in Phases 4 & 6.
- Adjust phases and tasks as new modules or priorities emerge.


## Facial Vision System Extensions
Use this list to track the development of all detection extensions beyond face recognition.

✅ Pre-Requisite Milestones (Complete)
✅ Working face detection pipeline

✅ JSON output format established

✅ Batch processing capability

✅ Video frame extraction working

## Extension Modules
5.1 License Plate OCR
  - Implement src/core/license_plate_detector.py with OpenCV + Tesseract OCR
  - Integrate into image_processor.py to extract text and save plate chips
  - Update JSON to include type: "license_plate" and metadata.extracted_text

5.2 Logo Recognition
  - Implement src/core/logo_detector.py with SIFT/ORB and logo DB
  - Create scripts/setup_logos.py to fetch and prepare reference logos
  - Return bounding boxes, brand names, confidence scores in JSON

5.3 Vehicle Detection
  - Implement vehicle_detector.py using YOLOv5 + make/model classifier
  - Add dominant color detection with webcolors
  - Extend JSON with make, model, color, type, and confidence

5.4 Tattoo Detection
  - Implement tattoo_detector.py using CNN or Detectron2
  - Create scripts/prepare_tattoo_dataset.py for dataset and augmentation
  - Add tattoo-specific metadata to JSON (e.g., style, body location)

5.5 Audio Transcription
  - Create audio_processor.py using ffmpeg-python and OpenAI Whisper
  - Extract speaker segments and timestamps
  - Sync transcripts with video + add to final JSON output

## Unified Processing & Optimization
  - Implement multi_detector_processor.py to unify all detectors
  - Add config flags to enable/disable each detector
  - Optimize shared preprocessing, batching, and GPU usage
  - Write integration tests for each detection module

## Testing, Docs, and Deployment
  - Create tests/test_extensions.py for all detectors
  - Validate JSON output formats across detection types
  - Update config/default.yaml to support detector-specific configs
  - Create CLI interface for multi-detector batch runs
  - Add performance profiling and scheduler logic
  - Update docs and deploy via Docker
