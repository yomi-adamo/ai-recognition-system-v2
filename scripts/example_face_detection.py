#!/usr/bin/env python3
"""
Example script demonstrating FaceDetector usage
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.face_detector import FaceDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    """Example usage of FaceDetector"""
    
    # Initialize detector with HOG model (faster)
    detector_hog = FaceDetector(model="hog", tolerance=0.6, min_face_size=40)
    
    # For more accurate detection, use CNN model
    # detector_cnn = FaceDetector(model="cnn", tolerance=0.5)
    
    # Example 1: Detect faces in a single image
    print("Example 1: Single Image Face Detection")
    print("-" * 40)
    
    # You would replace this with an actual image path
    image_path = Path("data/input/sample_face.jpg")
    
    if image_path.exists():
        faces = detector_hog.detect_faces(image_path)
        
        print(f"Found {len(faces)} faces in {image_path.name}")
        
        for idx, face in enumerate(faces):
            print(f"\nFace {idx + 1}:")
            print(f"  Bounding box: {face['bbox']}")
            print(f"  Confidence: {face['confidence']:.2%}")
            print(f"  Area: {face['area']} pixels")
            print(f"  Center: {face['center']}")
            print(f"  Dimensions: {face['dimensions'][0]}x{face['dimensions'][1]}")
            
        # Draw bounding boxes and save
        output_path = Path("data/output/sample_face_detected.jpg")
        detector_hog.draw_faces(image_path, faces, output_path)
        print(f"\nSaved annotated image to: {output_path}")
        
    else:
        print(f"Sample image not found at {image_path}")
        print("Please add a test image to data/input/sample_face.jpg")
    
    # Example 2: Batch detection
    print("\n\nExample 2: Batch Face Detection")
    print("-" * 40)
    
    # Get all jpg images in input directory
    input_dir = Path("data/input")
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    if image_files:
        print(f"Processing {len(image_files)} images...")
        results = detector_hog.batch_detect(image_files)
        
        for image_path, faces in results.items():
            print(f"\n{Path(image_path).name}: {len(faces)} faces detected")
    else:
        print("No images found in data/input/")
    
    # Example 3: Using OpenCV as alternative
    print("\n\nExample 3: OpenCV Face Detection (Alternative)")
    print("-" * 40)
    
    if image_path.exists():
        faces_cv = detector_hog.detect_faces_opencv(image_path)
        print(f"OpenCV detected {len(faces_cv)} faces")
    
    # Example 4: Get face encodings for recognition
    print("\n\nExample 4: Face Encodings")
    print("-" * 40)
    
    if image_path.exists() and faces:
        encodings = detector_hog.get_face_encodings(image_path, faces)
        print(f"Generated {len(encodings)} face encodings")
        print(f"Each encoding is a {len(encodings[0]) if encodings else 0}-dimensional vector")


if __name__ == "__main__":
    main()