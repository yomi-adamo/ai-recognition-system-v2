import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.face_detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector_hog = FaceDetector(model="hog")
        self.detector_cnn = FaceDetector(model="cnn")
        
    def test_initialization(self):
        """Test detector initialization"""
        # Test default initialization
        detector = FaceDetector()
        self.assertEqual(detector.model, "hog")
        self.assertEqual(detector.tolerance, 0.6)
        self.assertEqual(detector.min_face_size, 40)
        
        # Test custom initialization
        detector = FaceDetector(model="cnn", tolerance=0.5, min_face_size=30)
        self.assertEqual(detector.model, "cnn")
        self.assertEqual(detector.tolerance, 0.5)
        self.assertEqual(detector.min_face_size, 30)
        
        # Test invalid model
        with self.assertRaises(ValueError):
            FaceDetector(model="invalid")
    
    def test_detect_faces_with_synthetic_image(self):
        """Test face detection with a synthetic image"""
        # Create a dummy image (would need actual face for real test)
        dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
        dummy_image.fill(255)  # White image
        
        # This will likely return no faces since it's a blank image
        faces = self.detector_hog.detect_faces(dummy_image)
        self.assertIsInstance(faces, list)
        
    def test_face_info_structure(self):
        """Test that face info has correct structure"""
        # Create dummy face info
        face_info = {
            "bbox": (100, 200, 300, 100),
            "confidence": 0.95,
            "area": 20000,
            "center": (150, 200),
            "dimensions": (100, 200)
        }
        
        # Check all required keys
        required_keys = ["bbox", "confidence", "area", "center", "dimensions"]
        for key in required_keys:
            self.assertIn(key, face_info)
            
        # Check bbox format (top, right, bottom, left)
        top, right, bottom, left = face_info["bbox"]
        self.assertLess(top, bottom)
        self.assertLess(left, right)
        
    def test_confidence_estimation(self):
        """Test confidence estimation logic"""
        detector = FaceDetector()
        
        # Test different face sizes
        image_shape = (1000, 1000, 3)
        
        # Large face (10% of image)
        large_face_area = 100000
        confidence = detector._estimate_confidence(large_face_area, image_shape)
        self.assertGreater(confidence, 0.9)
        
        # Medium face (5% of image)
        medium_face_area = 50000
        confidence = detector._estimate_confidence(medium_face_area, image_shape)
        self.assertGreater(confidence, 0.85)
        
        # Small face (1% of image)
        small_face_area = 10000
        confidence = detector._estimate_confidence(small_face_area, image_shape)
        self.assertGreater(confidence, 0.8)
        
    def test_batch_detect(self):
        """Test batch detection"""
        # Create dummy images
        dummy_paths = ["image1.jpg", "image2.jpg"]
        
        # This would need actual implementation with real images
        # For now, just test the return structure
        results = self.detector_hog.batch_detect([])
        self.assertIsInstance(results, dict)


class TestFaceDetectorIntegration(unittest.TestCase):
    """Integration tests that require actual face images"""
    
    @unittest.skip("Requires actual face images in data/input/")
    def test_real_face_detection(self):
        """Test with real face images"""
        detector = FaceDetector(model="hog")
        
        # This would load an actual test image
        test_image_path = Path("data/input/test_face.jpg")
        if test_image_path.exists():
            faces = detector.detect_faces(test_image_path)
            
            # Should detect at least one face
            self.assertGreater(len(faces), 0)
            
            # Check face structure
            for face in faces:
                self.assertIn("bbox", face)
                self.assertIn("confidence", face)
                self.assertIn("area", face)
                self.assertIn("center", face)
                
                # Validate bbox
                top, right, bottom, left = face["bbox"]
                self.assertLess(top, bottom)
                self.assertLess(left, right)
                
                # Validate confidence
                self.assertGreaterEqual(face["confidence"], 0.0)
                self.assertLessEqual(face["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()