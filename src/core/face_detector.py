try:
    import face_recognition
except ImportError:
    # Create a mock face_recognition module for type checking
    import types
    face_recognition = types.ModuleType('face_recognition')
    face_recognition.load_image_file = lambda x: cv2.imread(str(x))
    face_recognition.face_locations = lambda *args, **kwargs: []
    face_recognition.face_encodings = lambda *args, **kwargs: []
    print("Warning: face_recognition module not found. Some functionality will be limited.")
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection class that supports multiple detection models.
    Returns bounding boxes for detected faces.
    """
    
    def __init__(self, model: str = "cnn", tolerance: float = 0.8, 
                 min_face_size: int = 40, upsampling: int = 1):
        """
        Initialize the face detector.
        
        Args:
            model: Detection model - "hog" (faster) or "cnn" (more accurate)
            tolerance: Face detection tolerance (0.0-1.0, lower is stricter)
            min_face_size: Minimum face size in pixels to detect
            upsampling: Number of times to upsample the image for better detection
        """
        self.model = model.lower()
        if self.model not in ["hog", "cnn"]:
            raise ValueError(f"Model must be 'hog' or 'cnn', got '{model}'")
            
        self.tolerance = tolerance
        self.min_face_size = min_face_size
        self.upsampling = upsampling
        
        logger.info(f"Initialized FaceDetector with model='{self.model}', "
                   f"tolerance={self.tolerance}, min_face_size={self.min_face_size}")
    
    def detect_faces(self, image_path: Union[str, Path, np.ndarray]) -> List[Dict]:
        """
        Detect faces in an image and return bounding boxes.
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            List of dictionaries containing face information:
            [{
                "bbox": (top, right, bottom, left),
                "confidence": float,
                "area": int,
                "center": (x, y)
            }]
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = face_recognition.load_image_file(str(image_path))
            logger.debug(f"Loaded image from {image_path}")
        else:
            image = image_path
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Detect face locations
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=self.upsampling,
            model=self.model
        )
        
        # Process detected faces
        faces = []
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # Calculate face dimensions
            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height
            
            # Filter by minimum size
            if face_width < self.min_face_size or face_height < self.min_face_size:
                logger.debug(f"Skipping face {idx}: size {face_width}x{face_height} "
                           f"below minimum {self.min_face_size}")
                continue
            
            # Calculate center point
            center_x = left + face_width // 2
            center_y = top + face_height // 2
            
            face_info = {
                "bbox": (top, right, bottom, left),
                "confidence": self._estimate_confidence(face_area, image.shape),
                "area": face_area,
                "center": (center_x, center_y),
                "dimensions": (face_width, face_height)
            }
            faces.append(face_info)
            
        logger.info(f"Detected {len(faces)} faces in image")
        return faces
    
    def detect_faces_opencv(self, image_path: Union[str, Path, np.ndarray]) -> List[Dict]:
        """
        Alternative face detection using OpenCV's Haar Cascades.
        Useful as a fallback or for comparison.
        
        Args:
            image_path: Path to image file or numpy array
            
        Returns:
            List of face dictionaries with bounding boxes
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            if len(image_path.shape) == 3:
                gray = cv2.cvtColor(image_path, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_path
                
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces
        faces_cv = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        # Convert to our format
        faces = []
        for (x, y, w, h) in faces_cv:
            # Convert from (x, y, w, h) to (top, right, bottom, left)
            top = y
            left = x
            bottom = y + h
            right = x + w
            
            face_info = {
                "bbox": (top, right, bottom, left),
                "confidence": 0.8,  # OpenCV doesn't provide confidence
                "area": w * h,
                "center": (x + w // 2, y + h // 2),
                "dimensions": (w, h)
            }
            faces.append(face_info)
            
        return faces
    
    def draw_faces(self, image_path: Union[str, Path, np.ndarray], 
                   faces: List[Dict], output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image_path: Path to image or numpy array
            faces: List of face dictionaries from detect_faces
            output_path: Optional path to save the annotated image
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
        else:
            image = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
            
        # Draw each face
        for idx, face in enumerate(faces):
            top, right, bottom, left = face["bbox"]
            
            # Draw rectangle
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"Face {idx + 1}: {face['confidence']:.2f}"
            cv2.putText(image, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), image)
            logger.info(f"Saved annotated image to {output_path}")
            
        return image
    
    def get_face_encodings(self, image_path: Union[str, Path, np.ndarray], 
                          faces: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """
        Get face encodings for detected faces (used for recognition).
        
        Args:
            image_path: Path to image or numpy array
            faces: Optional pre-detected faces, otherwise will detect
            
        Returns:
            List of face encodings (128-dimensional vectors)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = face_recognition.load_image_file(str(image_path))
        else:
            image = image_path
            
        # Get face locations
        if faces is None:
            faces = self.detect_faces(image)
            
        # Convert our bbox format to face_recognition format
        face_locations = [face["bbox"] for face in faces]
        
        # Get encodings
        encodings = face_recognition.face_encodings(image, face_locations)
        
        return encodings
    
    def _estimate_confidence(self, face_area: int, image_shape: Tuple) -> float:
        """
        Estimate confidence based on face size relative to image.
        
        Args:
            face_area: Area of detected face in pixels
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        image_area = image_shape[0] * image_shape[1]
        face_ratio = face_area / image_area
        
        # Larger faces relative to image size get higher confidence
        # Typical face is 1-10% of image area
        if face_ratio > 0.1:
            confidence = 0.95
        elif face_ratio > 0.05:
            confidence = 0.90
        elif face_ratio > 0.01:
            confidence = 0.85
        else:
            confidence = 0.80
            
        return min(confidence * (1.0 if self.model == "cnn" else 0.95), 1.0)
    
    def batch_detect(self, image_paths: List[Union[str, Path]]) -> Dict[str, List[Dict]]:
        """
        Detect faces in multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image paths to face lists
        """
        results = {}
        
        for image_path in image_paths:
            try:
                faces = self.detect_faces(image_path)
                results[str(image_path)] = faces
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[str(image_path)] = []
                
        return results
