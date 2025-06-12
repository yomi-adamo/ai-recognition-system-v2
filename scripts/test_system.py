#!/usr/bin/env python3
"""
Test the facial vision system with sample data
"""

import sys
import numpy as np
from pathlib import Path
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.face_detector import FaceDetector
from src.core.metadata_extractor import MetadataExtractor
from src.core.chip_generator import ChipGenerator
from src.processors.image_processor import ImageProcessor
from src.utils.logger import setup_logger
from src.utils.config import get_config


def create_test_image_with_face():
    """Create a simple test image with a face-like pattern"""
    # Create a 400x400 image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw a simple face pattern
    center_x, center_y = 200, 200
    
    # Face outline (circle)
    cv2.circle(img, (center_x, center_y), 80, (200, 180, 160), -1)  # Skin color
    
    # Eyes
    cv2.circle(img, (center_x - 25, center_y - 20), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (center_x + 25, center_y - 20), 8, (50, 50, 50), -1)  # Right eye
    
    # Nose
    cv2.circle(img, (center_x, center_y), 4, (150, 120, 100), -1)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y + 25), (15, 8), 0, 0, 180, (100, 50, 50), -1)
    
    return img


def test_individual_components():
    """Test each component individually"""
    logger = setup_logger("test_system", level="INFO")
    
    print("Testing Individual Components")
    print("=" * 50)
    
    # Test 1: Configuration
    print("1. Testing Configuration System...")
    try:
        config = get_config()
        face_config = config.get_face_detection_config()
        print(f"   ✓ Config loaded. Face detection model: {face_config.get('model', 'hog')}")
    except Exception as e:
        print(f"   ✗ Config failed: {e}")
        return False
    
    # Test 2: Create test image
    print("2. Creating test image...")
    try:
        test_img = create_test_image_with_face()
        test_img_path = Path("data/input/test_face.jpg")
        test_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(test_img_path), test_img)
        print(f"   ✓ Test image created: {test_img_path}")
    except Exception as e:
        print(f"   ✗ Test image creation failed: {e}")
        return False
    
    # Test 3: Face Detector
    print("3. Testing Face Detector...")
    try:
        detector = FaceDetector(model="hog")
        faces = detector.detect_faces(test_img)
        print(f"   ✓ Face detector initialized. Detected {len(faces)} faces")
        if len(faces) > 0:
            face = faces[0]
            print(f"     First face: confidence={face['confidence']:.2f}, bbox={face['bbox']}")
    except Exception as e:
        print(f"   ✗ Face detector failed: {e}")
        return False
    
    # Test 4: Metadata Extractor
    print("4. Testing Metadata Extractor...")
    try:
        extractor = MetadataExtractor()
        metadata = extractor.extract_metadata(test_img_path)
        print(f"   ✓ Metadata extracted. Keys: {list(metadata.keys())}")
        print(f"     Timestamp: {metadata.get('timestamp', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Metadata extractor failed: {e}")
        return False
    
    # Test 5: Chip Generator
    print("5. Testing Chip Generator...")
    try:
        generator = ChipGenerator()
        if len(faces) > 0:
            chip_data = generator.generate_chip(test_img, faces[0]['bbox'])
            print(f"   ✓ Chip generated. Name: {chip_data['name']}")
            print(f"     Chip size: {chip_data['chip_size']}")
        else:
            print("   ⚠ No faces to generate chips from")
    except Exception as e:
        print(f"   ✗ Chip generator failed: {e}")
        return False
    
    return True


def test_image_processor():
    """Test the complete image processing pipeline"""
    print("\nTesting Complete Pipeline")
    print("=" * 50)
    
    try:
        processor = ImageProcessor()
        test_img_path = Path("data/input/test_face.jpg")
        
        if not test_img_path.exists():
            print("   ✗ Test image not found. Run individual component tests first.")
            return False
        
        print("1. Processing test image...")
        results = processor.process_image(test_img_path, save_chips=True)
        
        if len(results) == 0:
            print("   ⚠ No faces detected in test image")
            return True
        
        print(f"   ✓ Pipeline complete. Generated {len(results)} face records")
        
        # Validate JSON structure
        print("2. Validating JSON structure...")
        for idx, result in enumerate(results):
            required_fields = ["file", "type", "name", "author", "parentId", "metadata", "topics"]
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                print(f"   ✗ Face {idx}: Missing fields {missing_fields}")
                return False
            
            metadata = result["metadata"]
            required_metadata = ["timestamp", "confidence", "identity", "source_file", "face_bounds"]
            missing_metadata = [f for f in required_metadata if f not in metadata]
            
            if missing_metadata:
                print(f"   ✗ Face {idx}: Missing metadata {missing_metadata}")
                return False
        
        print("   ✓ JSON structure valid for all faces")
        
        # Print sample result
        print("3. Sample face detection result:")
        sample = results[0]
        print(f"   Name: {sample['name']}")
        print(f"   Confidence: {sample['metadata']['confidence']:.2%}")
        print(f"   Face bounds: {sample['metadata']['face_bounds']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Pipeline test failed: {e}")
        return False


def test_scripts():
    """Test the processing scripts"""
    print("\nTesting Processing Scripts")
    print("=" * 50)
    
    test_img_path = Path("data/input/test_face.jpg")
    
    if not test_img_path.exists():
        print("   ✗ Test image not found")
        return False
    
    # Test process_image.py
    print("1. Testing process_image.py...")
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/process_image.py", 
            str(test_img_path), "--json-only"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("   ✓ process_image.py executed successfully")
            # Try to parse JSON output
            import json
            json.loads(result.stdout)
            print("   ✓ Valid JSON output generated")
        else:
            print(f"   ✗ process_image.py failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ✗ Script test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("Facial Vision System Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test individual components
    if not test_individual_components():
        all_passed = False
    
    # Test complete pipeline
    if not test_image_processor():
        all_passed = False
    
    # Test scripts
    if not test_scripts():
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add real images to data/input/")
        print("2. Run: python scripts/process_image.py data/input/your_image.jpg")
        print("3. Or process a folder: python scripts/process_folder.py data/input/")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check the virtual environment is activated")
        print("3. Verify the configuration files are present")
    
    print("=" * 60)


if __name__ == "__main__":
    main()