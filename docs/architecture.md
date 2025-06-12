# System Architecture

## Processing Pipeline
1. **Input Stage**
   - File validation
   - Metadata extraction
   - Format normalization

2. **Detection Stage**
   - Face detection (MTCNN/RetinaFace)
   - Face alignment
   - Quality assessment

3. **Recognition Stage**
   - Feature extraction
   - Identity matching
   - Confidence scoring

4. **Output Stage**
   - Chip generation
   - JSON formatting
   - IPFS upload
   - Blockchain logging

## Technology Stack
- Python 3.9+
- OpenCV for image/video processing
- face_recognition or DeepFace for detection
- IPFS Python client
- FireFly SDK