#!/usr/bin/env python3
"""
Test script for Phase 2 & Phase 3 with real images and videos
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.processors.batch_processor import BatchProcessor
from src.outputs.json_formatter import JSONFormatter


def test_with_real_image(image_path: str, output_dir: str = "test_output"):
    """Test image processing with real image"""
    print(f"📸 Testing with real image: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Initialize processor
        processor = ImageProcessor(enable_clustering=True)
        formatter = JSONFormatter()
        
        # Process image
        result = processor.process_image(
            image_path=image_path,
            output_dir=f"{output_dir}/images",
            save_chips=True
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON
        with open(f"{output_dir}/image_result.json", "w") as f:
            import json
            json.dump(result, f, indent=2)
        
        # Save blockchain-ready JSON
        blockchain_json = formatter.create_blockchain_asset_json(result)
        with open(f"{output_dir}/blockchain_ready.json", "w") as f:
            import json
            json.dump(blockchain_json, f, indent=2)
        
        print(f"✅ Image processed successfully!")
        
        # Handle different possible stats structure
        stats = result.get('metadata', {}).get('processing_stats', {})
        faces_detected = stats.get('faces_detected', stats.get('total_faces_detected', 0))
        clusters_assigned = stats.get('clusters_assigned', stats.get('unique_clusters', 0))
        
        print(f"   Faces detected: {faces_detected}")
        print(f"   Clusters: {clusters_assigned}")
        print(f"   Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False


def test_with_real_video(video_path: str, output_dir: str = "test_output"):
    """Test video processing with real video"""
    print(f"🎥 Testing with real video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return False
    
    try:
        # Initialize processor
        processor = VideoProcessor(enable_clustering=True)
        formatter = JSONFormatter()
        
        # Process video
        result = processor.process_video(
            video_path=video_path,
            output_dir=f"{output_dir}/videos",
            save_chips=True
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed JSON
        with open(f"{output_dir}/video_result.json", "w") as f:
            import json
            json.dump(result, f, indent=2)
        
        # Save blockchain-ready JSON
        blockchain_json = formatter.create_blockchain_asset_json(result)
        with open(f"{output_dir}/video_blockchain_ready.json", "w") as f:
            import json
            json.dump(blockchain_json, f, indent=2)
        
        print(f"✅ Video processed successfully!")
        
        # Handle different possible stats structure
        stats = result.get('metadata', {}).get('processing_stats', {})
        faces_detected = stats.get('faces_detected', stats.get('total_faces_detected', 0))
        clusters_assigned = stats.get('clusters_assigned', stats.get('unique_clusters', 0))
        
        print(f"   Faces detected: {faces_detected}")
        print(f"   Clusters: {clusters_assigned}")
        print(f"   Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return False


def test_batch_processing(input_dir: str, output_dir: str = "test_output"):
    """Test batch processing with directory of files"""
    print(f"📦 Testing batch processing: {input_dir}")
    
    if not Path(input_dir).exists():
        print(f"❌ Directory not found: {input_dir}")
        return False
    
    try:
        # Initialize processor with single worker to avoid segfault
        processor = BatchProcessor(enable_clustering=True, max_workers=1)
        formatter = JSONFormatter()
        
        # Process directory
        result = processor.process_directory(
            input_dir=input_dir,
            output_dir=f"{output_dir}/batch",
            recursive=True,
            save_chips=True
        )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save batch results
        with open(f"{output_dir}/batch_result.json", "w") as f:
            import json
            json.dump(result, f, indent=2)
        
        # Format for blockchain
        batch_formatted = formatter.format_batch_result(result)
        with open(f"{output_dir}/batch_blockchain_ready.json", "w") as f:
            import json
            json.dump(batch_formatted, f, indent=2)
        
        stats = result['processing_stats']
        print(f"✅ Batch processing completed!")
        print(f"   Files processed: {stats['total_files_processed']}")
        print(f"   Images: {stats['images_processed']}")
        print(f"   Videos: {stats['videos_processed']}")
        print(f"   Total faces: {stats['total_faces_detected']}")
        print(f"   Unique clusters: {stats['unique_clusters']}")
        print(f"   Output saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Phase 2 & 3 with real media")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--video", help="Path to test video") 
    parser.add_argument("--batch", help="Path to directory with images/videos")
    parser.add_argument("--output", default="test_output", help="Output directory")
    
    args = parser.parse_args()
    
    if not any([args.image, args.video, args.batch]):
        print("Usage examples:")
        print("  # Test with image:")
        print("  python test_real_media.py --image /path/to/image.jpg")
        print()
        print("  # Test with video:")
        print("  python test_real_media.py --video /path/to/video.mp4")
        print()
        print("  # Test batch processing:")
        print("  python test_real_media.py --batch /path/to/media/directory")
        print()
        print("  # All tests with custom output:")
        print("  python test_real_media.py --image test.jpg --output my_output")
        return
    
    print("🚀 Real Media Testing for Phase 2 & Phase 3")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    if args.image:
        total_tests += 1
        if test_with_real_image(args.image, args.output):
            success_count += 1
        print()
    
    if args.video:
        total_tests += 1
        if test_with_real_video(args.video, args.output):
            success_count += 1
        print()
    
    if args.batch:
        total_tests += 1
        if test_batch_processing(args.batch, args.output):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} successful")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")


if __name__ == "__main__":
    main()