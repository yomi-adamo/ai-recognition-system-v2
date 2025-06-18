from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Handle optional imports
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

try:
    from mtcnn import MTCNN

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    MTCNN = None

from src.utils.config import get_config
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


@dataclass
class FaceDetection:
    """Data class for face detection results"""

    bbox: Tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    encoding: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        """Calculate face area"""
        top, right, bottom, left = self.bbox
        return (right - left) * (bottom - top)

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate face center"""
        top, right, bottom, left = self.bbox
        return (left + (right - left) // 2, top + (bottom - top) // 2)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get face dimensions (width, height)"""
        top, right, bottom, left = self.bbox
        return (right - left, bottom - top)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "area": self.area,
            "center": self.center,
            "dimensions": self.dimensions,
            "landmarks": self.landmarks,
            "has_encoding": self.encoding is not None,
        }


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image"""
        pass

    @abstractmethod
    def extract_encodings(
        self, image: np.ndarray, detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract face encodings for detected faces"""
        pass


class FaceRecognitionDetector(BaseFaceDetector):
    """Face detector using face_recognition library"""

    def __init__(self, model: str = "hog", min_face_size: int = 40, upsampling: int = 1, confidence_threshold: float = 0.5):
        if not FACE_RECOGNITION_AVAILABLE:
            raise ImportError("face_recognition is not installed")

        self.model = model.lower()
        if self.model not in ["hog", "cnn"]:
            raise ValueError(f"Model must be 'hog' or 'cnn', got '{model}'")

        self.min_face_size = min_face_size
        self.upsampling = upsampling
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"Initialized FaceRecognitionDetector",
            model=self.model,
            min_face_size=self.min_face_size,
        )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using face_recognition"""
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            # Assume BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            image, number_of_times_to_upsample=self.upsampling, model=self.model
        )

        detections = []
        for bbox in face_locations:
            top, right, bottom, left = bbox
            width = right - left
            height = bottom - top

            # Filter by size
            if width < self.min_face_size or height < self.min_face_size:
                continue

            # Estimate confidence based on size
            confidence = self._estimate_confidence(width * height, image.shape)
            
            # Filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue

            detections.append(FaceDetection(bbox=bbox, confidence=confidence))

        return detections

    def extract_encodings(
        self, image: np.ndarray, detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract face encodings"""
        face_locations = [d.bbox for d in detections]
        encodings = face_recognition.face_encodings(image, face_locations)

        # Update detections with encodings
        for detection, encoding in zip(detections, encodings):
            detection.encoding = encoding

        return encodings

    def _estimate_confidence(self, face_area: int, image_shape: Tuple) -> float:
        """Estimate confidence based on face size"""
        image_area = image_shape[0] * image_shape[1]
        face_ratio = face_area / image_area

        if face_ratio > 0.1:
            confidence = 0.95
        elif face_ratio > 0.05:
            confidence = 0.90
        elif face_ratio > 0.01:
            confidence = 0.85
        else:
            confidence = 0.80

        return min(confidence * (1.0 if self.model == "cnn" else 0.95), 1.0)


class MTCNNDetector(BaseFaceDetector):
    """Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)"""

    def __init__(
        self,
        min_face_size: int = 40,
        thresholds: Optional[List[float]] = None,
        scale_factor: float = 0.709,
    ):
        if not MTCNN_AVAILABLE:
            raise ImportError("mtcnn is not installed")

        self.min_face_size = min_face_size
        self.detector = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds or [0.6, 0.7, 0.7],
            scale_factor=scale_factor,
        )

        logger.info(f"Initialized MTCNNDetector", min_face_size=min_face_size)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MTCNN"""
        # MTCNN expects RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Detect faces
        results = self.detector.detect_faces(image)

        detections = []
        for result in results:
            x, y, w, h = result["box"]
            confidence = result["confidence"]

            # Convert to our bbox format (top, right, bottom, left)
            bbox = (y, x + w, y + h, x)

            # Extract landmarks
            landmarks = {}
            if "keypoints" in result:
                landmarks = {
                    "left_eye": result["keypoints"]["left_eye"],
                    "right_eye": result["keypoints"]["right_eye"],
                    "nose": result["keypoints"]["nose"],
                    "mouth_left": result["keypoints"]["mouth_left"],
                    "mouth_right": result["keypoints"]["mouth_right"],
                }

            detections.append(FaceDetection(bbox=bbox, confidence=confidence, landmarks=landmarks))

        return detections

    def extract_encodings(
        self, image: np.ndarray, detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract face encodings using face_recognition if available"""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition not available for encoding extraction")
            return []

        face_locations = [d.bbox for d in detections]
        encodings = face_recognition.face_encodings(image, face_locations)

        # Update detections with encodings
        for detection, encoding in zip(detections, encodings):
            detection.encoding = encoding

        return encodings


class OpenCVDetector(BaseFaceDetector):
    """Face detector using OpenCV Haar Cascades"""

    def __init__(self, min_face_size: int = 40, scale_factor: float = 1.1, min_neighbors: int = 3):
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

        logger.info(f"Initialized OpenCVDetector", min_face_size=min_face_size)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )

        detections = []
        for x, y, w, h in faces:
            # Convert to our bbox format (top, right, bottom, left)
            bbox = (y, x + w, y + h, x)

            detections.append(
                FaceDetection(bbox=bbox, confidence=0.8)  # OpenCV doesn't provide confidence scores
            )

        return detections

    def extract_encodings(
        self, image: np.ndarray, detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract face encodings using face_recognition if available"""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.warning("face_recognition not available for encoding extraction")
            return []

        face_locations = [d.bbox for d in detections]
        encodings = face_recognition.face_encodings(image, face_locations)

        # Update detections with encodings
        for detection, encoding in zip(detections, encodings):
            detection.encoding = encoding

        return encodings


class FaceDetector:
    """
    Unified face detection class that supports multiple backends.
    """

    def __init__(self, backend: str = "face_recognition", **kwargs):
        """
        Initialize the face detector.

        Args:
            backend: Detection backend - "face_recognition", "mtcnn", or "opencv"
            **kwargs: Backend-specific arguments
        """
        self.config = get_config()
        self.backend_name = backend.lower()

        # Initialize the appropriate detector
        if self.backend_name == "face_recognition":
            self.detector = FaceRecognitionDetector(
                model=kwargs.get("model", "hog"),
                min_face_size=kwargs.get("min_face_size", 40),
                upsampling=kwargs.get("upsampling", 1),
                confidence_threshold=kwargs.get("confidence_threshold", 0.5),
            )
        elif self.backend_name == "mtcnn":
            self.detector = MTCNNDetector(
                min_face_size=kwargs.get("min_face_size", 40),
                thresholds=kwargs.get("thresholds", None),
                scale_factor=kwargs.get("scale_factor", 0.709),
            )
        elif self.backend_name == "opencv":
            self.detector = OpenCVDetector(
                min_face_size=kwargs.get("min_face_size", 40),
                scale_factor=kwargs.get("scale_factor", 1.1),
                min_neighbors=kwargs.get("min_neighbors", 3),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"Initialized FaceDetector", backend=self.backend_name)

    def detect_faces(self, image_path: Union[str, Path, np.ndarray]) -> List[Dict]:
        """
        Detect faces in an image and return bounding boxes.

        Args:
            image_path: Path to image file or numpy array

        Returns:
            List of dictionaries containing face information
        """
        # Load image
        image = self._load_image(image_path)

        # Detect faces using the selected backend
        detections = self.detector.detect(image)

        # Convert to dictionary format for backward compatibility
        faces = [detection.to_dict() for detection in detections]

        logger.info(f"Detected {len(faces)} faces using {self.backend_name}")
        return faces

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image array and return FaceDetection objects.

        Args:
            image: Image as numpy array

        Returns:
            List of FaceDetection objects
        """
        return self.detector.detect(image)

    def extract_encodings(
        self, image: np.ndarray, detections: Optional[List[FaceDetection]] = None
    ) -> List[np.ndarray]:
        """
        Extract face encodings for detected faces.

        Args:
            image: Image as numpy array
            detections: Optional pre-detected faces, otherwise will detect

        Returns:
            List of face encodings
        """
        if detections is None:
            detections = self.detector.detect(image)

        return self.detector.extract_encodings(image, detections)

    def _load_image(self, image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from path or return array"""
        if isinstance(image_path, (str, Path)):
            if self.backend_name == "face_recognition" and FACE_RECOGNITION_AVAILABLE:
                image = face_recognition.load_image_file(str(image_path))
            else:
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Could not load image from {image_path}")

            logger.debug(f"Loaded image", path=str(image_path), shape=image.shape)
        else:
            image = image_path

        return image

    def get_available_backends(self) -> List[str]:
        """Get list of available detection backends"""
        backends = ["opencv"]  # Always available

        if FACE_RECOGNITION_AVAILABLE:
            backends.append("face_recognition")

        if MTCNN_AVAILABLE:
            backends.append("mtcnn")

        return backends

    def draw_faces(
        self,
        image_path: Union[str, Path, np.ndarray],
        faces: Union[List[Dict], List[FaceDetection]],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.

        Args:
            image_path: Path to image or numpy array
            faces: List of face dictionaries or FaceDetection objects
            output_path: Optional path to save the annotated image

        Returns:
            Annotated image as numpy array
        """
        # Load image
        image = self._load_image(image_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw each face
        for idx, face in enumerate(faces):
            if isinstance(face, FaceDetection):
                bbox = face.bbox
                confidence = face.confidence
                landmarks = face.landmarks
            else:
                bbox = face["bbox"]
                confidence = face["confidence"]
                landmarks = face.get("landmarks")

            top, right, bottom, left = bbox

            # Draw rectangle
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add label with confidence
            label = f"Face {idx + 1}: {confidence:.2f}"
            cv2.putText(
                image_bgr, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            # Draw landmarks if available
            if landmarks:
                for landmark_name, (x, y) in landmarks.items():
                    cv2.circle(image_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)

        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), image_bgr)
            logger.info(f"Saved annotated image", path=str(output_path))

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def get_face_encodings(
        self, image_path: Union[str, Path, np.ndarray], faces: Optional[List[Dict]] = None
    ) -> List[np.ndarray]:
        """
        Get face encodings for detected faces (used for recognition).

        Args:
            image_path: Path to image or numpy array
            faces: Optional pre-detected faces, otherwise will detect

        Returns:
            List of face encodings (128-dimensional vectors)
        """
        # Load image
        image = self._load_image(image_path)

        # Get detections if not provided
        if faces is None:
            detections = self.detector.detect(image)
        else:
            # Convert dict format to FaceDetection objects
            detections = []
            for face in faces:
                detections.append(FaceDetection(bbox=face["bbox"], confidence=face["confidence"]))

        # Extract encodings
        return self.detector.extract_encodings(image, detections)

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
                logger.error(f"Error processing image", path=str(image_path), error=str(e))
                results[str(image_path)] = []

        logger.info(
            f"Batch processed {len(image_paths)} images",
            total_faces=sum(len(faces) for faces in results.values()),
        )
        return results

    @classmethod
    def create_from_config(cls, config: Optional[Dict[str, Any]] = None) -> "FaceDetector":
        """
        Create FaceDetector from configuration

        Args:
            config: Optional configuration dictionary

        Returns:
            Configured FaceDetector instance
        """
        if config is None:
            config_manager = get_config()
            config = config_manager.get_face_detection_config()

        backend = config.get("backend", "face_recognition")
        model = config.get("model", "hog")
        min_face_size = config.get("min_face_size", 40)

        return cls(backend=backend, model=model, min_face_size=min_face_size)
