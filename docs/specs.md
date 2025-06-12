# Facial Vision Technical Specifications

## Overview
Backend facial detection and recognition system for images and videos.

## Core Requirements
- Face detection with tight cropping
- Metadata extraction (timestamp, GPS, identity)
- JSON output format
- IPFS upload capability
- Blockchain logging via FireFly

## Performance Targets
- Process 1000 images/minute
- Handle 4K video at 30fps
- Batch processing for folders
- Intelligent video chunking

## Detection Capabilities
### Current:
- Faces (multiple per frame)
- Face recognition (known identities)

### Future:
- Tattoos
- Cars (make, model, color)
- License plates
- Logos (clothing, accessories)
- Voice-to-text extraction