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

Look at Facial Vision System Extensions for specifics on the implementations.

- [ ] Implement `tattoo_detector.py`
- [ ] Create `license_plate_reader.py` with OCR
- [ ] Build `vehicle_detector.py` classifier
- [ ] Implement `voice_transcriber.py` (audio-to-text)
- [ ] Create unified multi-modal detection pipeline

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
