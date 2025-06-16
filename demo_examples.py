#!/usr/bin/env python3
"""
Demo examples for Phase 2 & Phase 3 functionality
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.processors.batch_processor import BatchProcessor
from src.outputs.json_formatter import JSONFormatter

def demo_image_processing():
    """Demo image processing with clustering"""
    print("📸 Demo: Image Processing with Clustering")
    print("-" * 40)
    
    # Initialize processor
    processor = ImageProcessor(enable_clustering=True)
    
    print("Example usage:")
    print("""
# Process a single image
result = processor.process_image(
    image_path="/path/to/image.jpg",
    output_dir="/path/to/output",
    save_chips=True
)

# Results include:
- Detected faces with bounding boxes
- Cluster assignments (person_1, person_2, etc.)
- Face chips saved to cluster directories
- Comprehensive metadata with GPS, device info
    """)
    
    print("✅ Image processor ready!")

def demo_video_processing():
    """Demo video processing with frame extraction"""
    print("🎥 Demo: Video Processing with Frame Extraction")
    print("-" * 40)
    
    # Initialize processor
    processor = VideoProcessor(enable_clustering=True)
    
    print("Example usage:")
    print("""
# Process a video file
result = processor.process_video(
    video_path="/path/to/video.mp4",
    output_dir="/path/to/output",
    save_chips=True
)

# Results include:
- Faces detected across video frames
- Cluster assignments for all detected faces
- Frame-by-frame metadata with timestamps
- Video-specific info (FPS, duration, etc.)
    """)
    
    print("✅ Video processor ready!")

def demo_batch_processing():
    """Demo batch processing"""
    print("📦 Demo: Batch Processing")
    print("-" * 40)
    
    # Initialize processor
    processor = BatchProcessor(enable_clustering=True, max_workers=4)
    
    print("Example usage:")
    print("""
# Process entire directory
result = processor.process_directory(
    input_dir="/path/to/media/files",
    output_dir="/path/to/output",
    recursive=True,
    save_chips=True
)

# Process specific files
result = processor.process_files(
    file_paths=["/path/file1.jpg", "/path/video1.mp4"],
    output_dir="/path/to/output",
    save_chips=True
)

# Results include:
- Statistics for all processed files
- Combined clustering across all media
- Error handling for failed files
- Progress tracking
    """)
    
    print("✅ Batch processor ready!")

def demo_json_formatting():
    """Demo JSON formatting for blockchain"""
    print("📄 Demo: JSON Formatting for Blockchain")
    print("-" * 40)
    
    # Initialize formatter
    formatter = JSONFormatter()
    
    print("Example usage:")
    print("""
# Format processing result for blockchain
blockchain_json = formatter.create_blockchain_asset_json(
    processing_result=result,
    include_chips_in_metadata=True
)

# Save blockchain-ready JSON
formatter.save_blockchain_json(
    processing_result=result,
    output_path="/path/to/blockchain.json"
)

# Create cluster summary
cluster_summary = formatter.create_cluster_summary(
    processing_results=[result1, result2, result3]
)
    """)
    
    print("✅ JSON formatter ready!")

def demo_cluster_workflow():
    """Demo complete clustering workflow"""
    print("🧬 Demo: Complete Clustering Workflow")
    print("-" * 40)
    
    print("Complete workflow example:")
    print("""
from src.processors.image_processor import ImageProcessor
from src.outputs.json_formatter import JSONFormatter

# 1. Initialize components
processor = ImageProcessor(enable_clustering=True)
formatter = JSONFormatter()

# 2. Process images/videos
results = []
for file_path in ["/img1.jpg", "/img2.jpg", "/video1.mp4"]:
    result = processor.process_image(file_path, "/output", save_chips=True)
    results.append(result)

# 3. Create comprehensive summary
cluster_summary = formatter.create_cluster_summary(results)

# 4. Format for blockchain submission
for result in results:
    blockchain_json = formatter.create_blockchain_asset_json(result)
    # Submit to blockchain...

# Expected output structure:
{
  "file": "/path/to/image.jpg",
  "type": "image", 
  "name": "image_name",
  "author": "facial-vision-system",
  "timestamp": "2024-01-15T13:24:00Z",
  "parentId": "blockchain-asset-id",
  "metadata": {
    "processing_stats": {
      "faces_detected": 3,
      "clusters_assigned": 2,
      "clustering_enabled": true
    },
    "chips": [
      {
        "file": "person_1/chip_001.jpg",
        "clusterId": "person_1",
        "face_bounds": {"x": 84, "y": 122, "w": 64, "h": 64},
        "confidence": 0.95,
        "timestamp": "2024-01-15T13:24:00Z",
        "topics": ["face_detected", "clustered"]
      }
    ],
    "GPS": {"lat": 39.2557, "lon": -76.7112},
    "device": {"id": "AXIS-W120", "manufacturer": "AXIS"}
  },
  "topics": ["face_detected", "image_analysis"]
}
    """)
    
    print("✅ Complete workflow demonstrated!")

def main():
    """Run all demos"""
    print("🎬 Facial Vision Phase 2 & Phase 3 Demos")
    print("=" * 50)
    
    demo_image_processing()
    print()
    
    demo_video_processing()
    print()
    
    demo_batch_processing()
    print()
    
    demo_json_formatting()
    print()
    
    demo_cluster_workflow()
    print()
    
    print("=" * 50)
    print("🎯 Key Features Implemented:")
    print("  ✅ Face detection with multiple backends")
    print("  ✅ Unsupervised face clustering (person_1, person_2, etc.)")
    print("  ✅ Enhanced metadata extraction (GPS, device info, timestamps)")
    print("  ✅ Video frame extraction and processing")
    print("  ✅ Batch processing with multi-threading")
    print("  ✅ Blockchain-ready JSON formatting")
    print("  ✅ Comprehensive cluster management")
    print()
    print("Ready for Phase 4: Blockchain Integration! 🚀")

if __name__ == "__main__":
    main()