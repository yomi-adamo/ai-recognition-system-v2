#!/usr/bin/env python3
"""
Test script for Phase 2 and Phase 3 functionality
Run this to verify all components are working correctly
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.face_detector import FaceDetector
from src.core.face_clusterer import FaceClusterer
from src.core.metadata_extractor import MetadataExtractor
from src.core.chip_generator import ChipGenerator
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.processors.batch_processor import BatchProcessor
from src.outputs.json_formatter import JSONFormatter

def test_phase2_components():
    """Test Phase 2 face detection and clustering"""
    print("🔍 Testing Phase 2 Components...")
    
    try:
        # Test face detector
        print("  - Testing FaceDetector...")
        detector = FaceDetector(backend="face_recognition")
        print(f"    ✅ FaceDetector initialized with backend: {detector.backend_name}")
        
        # Test face clusterer
        print("  - Testing FaceClusterer...")
        clusterer = FaceClusterer.create_from_config()
        stats = clusterer.get_cluster_statistics()
        print(f"    ✅ FaceClusterer initialized: {stats['total_clusters']} clusters")
        
        # Test chip generator
        print("  - Testing ChipGenerator...")
        chip_gen = ChipGenerator()
        print(f"    ✅ ChipGenerator initialized: {chip_gen.chip_size}")
        
        print("✅ Phase 2 components working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 test failed: {e}\n")
        return False

def test_phase3_components():
    """Test Phase 3 metadata and processing"""
    print("🧠 Testing Phase 3 Components...")
    
    try:
        # Test metadata extractor
        print("  - Testing MetadataExtractor...")
        extractor = MetadataExtractor()
        print(f"    ✅ MetadataExtractor initialized")
        
        # Test image processor
        print("  - Testing ImageProcessor...")
        img_processor = ImageProcessor(enable_clustering=True)
        print(f"    ✅ ImageProcessor initialized: clustering={img_processor.enable_clustering}")
        
        # Test video processor
        print("  - Testing VideoProcessor...")
        vid_processor = VideoProcessor(enable_clustering=True)
        print(f"    ✅ VideoProcessor initialized: clustering={vid_processor.enable_clustering}")
        
        # Test batch processor
        print("  - Testing BatchProcessor...")
        batch_processor = BatchProcessor(enable_clustering=True)
        print(f"    ✅ BatchProcessor initialized: workers={batch_processor.max_workers}")
        
        # Test JSON formatter
        print("  - Testing JSONFormatter...")
        formatter = JSONFormatter()
        print(f"    ✅ JSONFormatter initialized: schema v{formatter.schema_version}")
        
        print("✅ Phase 3 components working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 test failed: {e}\n")
        return False

def test_integration():
    """Test end-to-end integration"""
    print("🔗 Testing Integration...")
    
    try:
        # Create test image (simple colored rectangle)
        import numpy as np
        import cv2
        
        # Create a simple test image with face-like regions
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_img_path = temp_path / "test_image.jpg"
            
            # Save test image
            cv2.imwrite(str(test_img_path), test_image)
            
            # Test image processing
            print(f"  - Processing test image: {test_img_path}")
            processor = ImageProcessor(enable_clustering=True)
            
            # This might not detect faces in random image, but should not crash
            result = processor.process_image(
                image_path=test_img_path,
                output_dir=temp_path / "output",
                save_chips=True
            )
            
            print(f"    ✅ Image processed: {result['metadata']['processing_stats']['faces_detected']} faces")
            
            # Test JSON formatting
            formatter = JSONFormatter()
            blockchain_json = formatter.create_blockchain_asset_json(result)
            
            print(f"    ✅ Blockchain JSON created: {len(blockchain_json)} fields")
            
        print("✅ Integration test completed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("🚀 Facial Vision Phase 2 & Phase 3 Tests\n")
    print("=" * 50)
    
    phase2_ok = test_phase2_components()
    phase3_ok = test_phase3_components()
    integration_ok = test_integration()
    
    print("=" * 50)
    print("📊 Test Results:")
    print(f"  Phase 2: {'✅ PASS' if phase2_ok else '❌ FAIL'}")
    print(f"  Phase 3: {'✅ PASS' if phase3_ok else '❌ FAIL'}")
    print(f"  Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if all([phase2_ok, phase3_ok, integration_ok]):
        print("\n🎉 All tests passed! Ready for Phase 4.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())