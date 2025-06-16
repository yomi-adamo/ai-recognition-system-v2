# Phase 2 & Phase 3 Testing Commands

## Prerequisites - Virtual Environment

First, activate your virtual environment:
```bash
source venv/bin/activate
```

Or use the automated test script:
```bash
./test_with_venv.sh
```

## Quick Component Tests

### 1. Run Comprehensive Test Suite
```bash
# With virtual environment activated:
python test_phase2_phase3.py

# Or use automated script:
./test_with_venv.sh
```

### 2. View Implementation Demos
```bash
python demo_examples.py
```

### 3. Test Individual Components

#### Test Configuration Loading
```bash
python -c "
from src.utils.config import get_config
config = get_config()
print('✅ Config loaded:', config.get_face_detection_config())
"
```

#### Test Face Detection
```bash
python -c "
from src.core.face_detector import FaceDetector
detector = FaceDetector(backend='face_recognition')
print('✅ FaceDetector ready:', detector.get_available_backends())
"
```

#### Test Clustering
```bash
python -c "
from src.core.face_clusterer import FaceClusterer
clusterer = FaceClusterer.create_from_config()
stats = clusterer.get_cluster_statistics()
print('✅ Clustering ready:', stats)
"
```

#### Test Metadata Extraction
```bash
python -c "
from src.core.metadata_extractor import MetadataExtractor
extractor = MetadataExtractor()
print('✅ Metadata extractor ready')
"
```

### 4. Test with Sample Data

#### Create Test Image and Process
```bash
python -c "
import numpy as np
import cv2
from pathlib import Path
from src.processors.image_processor import ImageProcessor

# Create test image
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
cv2.imwrite('test_image.jpg', img)

# Process image  
processor = ImageProcessor(enable_clustering=True)
result = processor.process_image('test_image.jpg', 'output', save_chips=True)
print('✅ Image processed:', result['metadata']['processing_stats'])
"
```

#### Test JSON Formatting
```bash
python -c "
from src.outputs.json_formatter import JSONFormatter

formatter = JSONFormatter()
test_result = {
    'file': 'test.jpg',
    'type': 'image', 
    'name': 'test',
    'author': 'test-system',
    'timestamp': '2024-01-15T10:00:00Z',
    'metadata': {'processing_stats': {'faces_detected': 2}, 'chips': []}
}

blockchain_json = formatter.create_blockchain_asset_json(test_result)
print('✅ Blockchain JSON created:', list(blockchain_json.keys()))
"
```

### 5. Test Error Handling

#### Test Missing Dependencies
```bash
python -c "
try:
    from src.core.face_detector import FaceDetector
    detector = FaceDetector(backend='mtcnn')
    print('✅ MTCNN available')
except ImportError as e:
    print('ℹ️  MTCNN not available:', e)
"
```

#### Test File Not Found
```bash
python -c "
from src.processors.image_processor import ImageProcessor
processor = ImageProcessor()

try:
    result = processor.process_image('nonexistent.jpg', 'output')
except FileNotFoundError:
    print('✅ Error handling works correctly')
"
```

## Performance Tests

### Test Batch Processing Performance
```bash
python -c "
import time
from src.processors.batch_processor import BatchProcessor

# Test batch processor initialization time
start = time.time()
processor = BatchProcessor(enable_clustering=True, max_workers=4)
init_time = time.time() - start
print(f'✅ Batch processor init time: {init_time:.3f}s')
"
```

## Cluster Registry Tests

### Test Cluster Persistence
```bash
python -c "
from src.core.face_clusterer import ClusterRegistry
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temp_dir:
    registry_path = Path(temp_dir) / 'test_registry.json'
    
    # Create registry
    registry = ClusterRegistry(registry_path)
    print('✅ Registry created')
    
    # Test persistence
    registry.save_registry()
    registry2 = ClusterRegistry(registry_path)
    print('✅ Registry persistence works')
"
```

## Integration with Configuration

### Test Different Backends
```bash
python -c "
from src.utils.config import get_config
from src.core.face_detector import FaceDetector

config = get_config()
face_config = config.get_face_detection_config()

# Test with different backends
for backend in ['face_recognition', 'opencv']:
    try:
        detector = FaceDetector(backend=backend)
        print(f'✅ {backend} backend works')
    except Exception as e:
        print(f'⚠️  {backend} backend issue: {e}')
"
```

## Testing with Real Images and Videos

### Test with Real Image
```bash
# Test single image
python test_real_media.py --image /path/to/your/image.jpg

# Example with sample images
python test_real_media.py --image ~/Pictures/photo.jpg --output real_test_output
```

### Test with Real Video
```bash
# Test single video
python test_real_media.py --video /path/to/your/video.mp4

# Example with sample video
python test_real_media.py --video ~/Videos/sample.mp4 --output real_test_output
```

### Test Batch Processing
```bash
# Test entire directory
python test_real_media.py --batch /path/to/media/directory

# Example with Pictures folder
python test_real_media.py --batch ~/Pictures --output batch_test_output
```

### Combined Real Media Test
```bash
# Test image, video, and batch together
python test_real_media.py \
  --image ~/Pictures/photo.jpg \
  --video ~/Videos/sample.mp4 \
  --batch ~/Pictures/test_folder \
  --output comprehensive_test
```

## Cleanup Test Files
```bash
rm -f test_image.jpg
rm -rf output/ test_output/ real_test_output/ batch_test_output/ comprehensive_test/
echo '🧹 Test files cleaned up'
```

## Summary Commands

### Quick Health Check
```bash
python -c "
print('🔍 Facial Vision Health Check')
print('-' * 30)

try:
    from src.core.face_detector import FaceDetector
    from src.core.face_clusterer import FaceClusterer  
    from src.processors.image_processor import ImageProcessor
    from src.processors.batch_processor import BatchProcessor
    from src.outputs.json_formatter import JSONFormatter
    
    print('✅ All imports successful')
    print('✅ Phase 2 & Phase 3 ready!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### Check Available Features
```bash
python -c "
from src.core.face_detector import FaceDetector
detector = FaceDetector(backend='face_recognition')
backends = detector.get_available_backends()
print('Available backends:', backends)

from src.core.face_clusterer import FaceClusterer
clusterer = FaceClusterer.create_from_config()
stats = clusterer.get_cluster_statistics()
print('Cluster stats:', stats)
"
```