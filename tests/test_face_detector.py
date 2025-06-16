"""
Tests for face detection functionality
"""
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from src.core.face_detector import (
    BaseFaceDetector,
    FaceDetection,
    FaceDetector,
    FaceRecognitionDetector,
    MTCNNDetector,
    OpenCVDetector,
)


class TestFaceDetection:
    """Test FaceDetection dataclass"""

    def test_face_detection_creation(self):
        """Test creating FaceDetection object"""
        bbox = (100, 200, 200, 100)  # top, right, bottom, left
        detection = FaceDetection(
            bbox=bbox, confidence=0.95, landmarks={"left_eye": (120, 130), "right_eye": (180, 130)}
        )

        assert detection.bbox == bbox
        assert detection.confidence == 0.95
        assert detection.landmarks is not None

    def test_face_detection_properties(self):
        """Test calculated properties"""
        detection = FaceDetection(bbox=(100, 200, 200, 100), confidence=0.95)  # 100x100 face

        assert detection.area == 10000  # 100 * 100
        assert detection.center == (150, 150)  # Center point
        assert detection.dimensions == (100, 100)  # width, height

    def test_face_detection_to_dict(self):
        """Test converting to dictionary"""
        detection = FaceDetection(
            bbox=(100, 200, 200, 100), confidence=0.95, landmarks={"left_eye": (120, 130)}
        )

        data = detection.to_dict()

        assert data["bbox"] == (100, 200, 200, 100)
        assert data["confidence"] == 0.95
        assert data["area"] == 10000
        assert data["center"] == (150, 150)
        assert data["landmarks"] == {"left_eye": (120, 130)}
        assert data["has_encoding"] is False

    def test_face_detection_with_encoding(self):
        """Test FaceDetection with encoding"""
        encoding = np.random.rand(128)
        detection = FaceDetection(bbox=(100, 200, 200, 100), confidence=0.95, encoding=encoding)

        assert np.array_equal(detection.encoding, encoding)
        assert detection.to_dict()["has_encoding"] is True


class TestFaceRecognitionDetector:
    """Test FaceRecognitionDetector"""

    def test_initialization(self):
        """Test detector initialization"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            detector = FaceRecognitionDetector(model="hog", min_face_size=40)

            assert detector.model == "hog"
            assert detector.min_face_size == 40

    def test_initialization_unavailable(self):
        """Test initialization when face_recognition is unavailable"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", False):
            with pytest.raises(ImportError, match="face_recognition is not installed"):
                FaceRecognitionDetector()

    def test_initialization_invalid_model(self):
        """Test initialization with invalid model"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with pytest.raises(ValueError, match="Model must be 'hog' or 'cnn'"):
                FaceRecognitionDetector(model="invalid")

    def test_detect_faces(self):
        """Test face detection"""
        mock_fr = Mock()
        mock_fr.face_locations.return_value = [
            (100, 200, 200, 100),  # top, right, bottom, left
            (300, 400, 400, 300),
        ]

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceRecognitionDetector()

                # Create test image
                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = detector.detect(image)

                assert len(detections) == 2
                assert all(isinstance(d, FaceDetection) for d in detections)
                assert detections[0].bbox == (100, 200, 200, 100)
                assert detections[1].bbox == (300, 400, 400, 300)

    def test_detect_no_faces(self):
        """Test detection with no faces"""
        mock_fr = Mock()
        mock_fr.face_locations.return_value = []

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceRecognitionDetector()

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = detector.detect(image)

                assert len(detections) == 0

    def test_detect_small_faces_filtered(self):
        """Test that small faces are filtered out"""
        mock_fr = Mock()
        mock_fr.face_locations.return_value = [
            (100, 200, 200, 100),  # 100x100 face (valid)
            (310, 330, 330, 310),  # 20x20 face (too small)
        ]

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceRecognitionDetector(min_face_size=40)

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = detector.detect(image)

                assert len(detections) == 1  # Only the large face
                assert detections[0].bbox == (100, 200, 200, 100)

    def test_extract_encodings(self):
        """Test face encoding extraction"""
        mock_fr = Mock()
        mock_fr.face_encodings.return_value = [np.random.rand(128), np.random.rand(128)]

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceRecognitionDetector()

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = [
                    FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9),
                    FaceDetection(bbox=(300, 400, 400, 300), confidence=0.8),
                ]

                encodings = detector.extract_encodings(image, detections)

                assert len(encodings) == 2
                assert all(enc.shape == (128,) for enc in encodings)
                # Check that encodings were added to detections
                assert detections[0].encoding is not None
                assert detections[1].encoding is not None


class TestMTCNNDetector:
    """Test MTCNNDetector"""

    def test_initialization(self):
        """Test MTCNN detector initialization"""
        mock_mtcnn = Mock()

        with patch("src.core.face_detector.MTCNN_AVAILABLE", True):
            with patch("src.core.face_detector.MTCNN", mock_mtcnn):
                detector = MTCNNDetector(min_face_size=40)

                assert detector.min_face_size == 40
                mock_mtcnn.assert_called_once()

    def test_initialization_unavailable(self):
        """Test initialization when MTCNN is unavailable"""
        with patch("src.core.face_detector.MTCNN_AVAILABLE", False):
            with pytest.raises(ImportError, match="mtcnn is not installed"):
                MTCNNDetector()

    def test_detect_faces(self):
        """Test MTCNN face detection"""
        mock_detector = Mock()
        mock_detector.detect_faces.return_value = [
            {
                "box": [100, 100, 100, 100],  # x, y, w, h
                "confidence": 0.95,
                "keypoints": {
                    "left_eye": (120, 130),
                    "right_eye": (180, 130),
                    "nose": (150, 160),
                    "mouth_left": (130, 180),
                    "mouth_right": (170, 180),
                },
            }
        ]

        with patch("src.core.face_detector.MTCNN_AVAILABLE", True):
            with patch("src.core.face_detector.MTCNN") as mock_mtcnn:
                mock_mtcnn.return_value = mock_detector

                detector = MTCNNDetector()

                image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = detector.detect(image)

                assert len(detections) == 1
                detection = detections[0]
                assert isinstance(detection, FaceDetection)
                assert detection.confidence == 0.95
                assert detection.landmarks is not None
                assert "left_eye" in detection.landmarks

    def test_extract_encodings_with_fallback(self):
        """Test encoding extraction with face_recognition fallback"""
        mock_fr = Mock()
        mock_fr.face_encodings.return_value = [np.random.rand(128)]

        with patch("src.core.face_detector.MTCNN_AVAILABLE", True):
            with patch("src.core.face_detector.MTCNN"):
                with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
                    with patch("src.core.face_detector.face_recognition", mock_fr):
                        detector = MTCNNDetector()

                        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        detections = [FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)]

                        encodings = detector.extract_encodings(image, detections)

                        assert len(encodings) == 1
                        assert encodings[0].shape == (128,)

    def test_extract_encodings_unavailable(self):
        """Test encoding extraction when face_recognition is unavailable"""
        with patch("src.core.face_detector.MTCNN_AVAILABLE", True):
            with patch("src.core.face_detector.MTCNN"):
                with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", False):
                    detector = MTCNNDetector()

                    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    detections = [FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)]

                    encodings = detector.extract_encodings(image, detections)

                    assert encodings == []


class TestOpenCVDetector:
    """Test OpenCVDetector"""

    def test_initialization(self):
        """Test OpenCV detector initialization"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False

        with patch("cv2.CascadeClassifier", return_value=mock_cascade):
            detector = OpenCVDetector(min_face_size=40)

            assert detector.min_face_size == 40
            assert detector.scale_factor == 1.1
            assert detector.min_neighbors == 3

    def test_initialization_cascade_failure(self):
        """Test initialization when cascade loading fails"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = True

        with patch("cv2.CascadeClassifier", return_value=mock_cascade):
            with pytest.raises(RuntimeError, match="Failed to load Haar Cascade classifier"):
                OpenCVDetector()

    def test_detect_faces(self):
        """Test OpenCV face detection"""
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = [
            (100, 100, 100, 100),  # x, y, w, h
            (300, 300, 80, 80),
        ]

        with patch("cv2.CascadeClassifier", return_value=mock_cascade):
            detector = OpenCVDetector()

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect(image)

            assert len(detections) == 2

            # Check bbox conversion (x,y,w,h) -> (top,right,bottom,left)
            assert detections[0].bbox == (100, 200, 200, 100)
            assert detections[1].bbox == (300, 380, 380, 300)
            assert all(d.confidence == 0.8 for d in detections)  # OpenCV default


class TestFaceDetector:
    """Test main FaceDetector class"""

    def test_initialization_face_recognition_backend(self):
        """Test initialization with face_recognition backend"""
        with patch("src.core.face_detector.FaceRecognitionDetector") as mock_detector:
            detector = FaceDetector(backend="face_recognition", model="hog")

            mock_detector.assert_called_once_with(model="hog", min_face_size=40, upsampling=1)
            assert detector.backend_name == "face_recognition"

    def test_initialization_mtcnn_backend(self):
        """Test initialization with MTCNN backend"""
        with patch("src.core.face_detector.MTCNNDetector") as mock_detector:
            detector = FaceDetector(backend="mtcnn", min_face_size=50)

            mock_detector.assert_called_once_with(
                min_face_size=50, thresholds=None, scale_factor=0.709
            )
            assert detector.backend_name == "mtcnn"

    def test_initialization_opencv_backend(self):
        """Test initialization with OpenCV backend"""
        with patch("src.core.face_detector.OpenCVDetector") as mock_detector:
            detector = FaceDetector(backend="opencv")

            mock_detector.assert_called_once()
            assert detector.backend_name == "opencv"

    def test_initialization_invalid_backend(self):
        """Test initialization with invalid backend"""
        with pytest.raises(ValueError, match="Unknown backend"):
            FaceDetector(backend="invalid_backend")

    def test_detect_faces_with_image_path(self, sample_image):
        """Test face detection with image path"""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)
        ]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            faces = detector.detect_faces(sample_image)

            assert len(faces) == 1
            assert isinstance(faces[0], dict)
            assert faces[0]["bbox"] == (100, 200, 200, 100)
            assert faces[0]["confidence"] == 0.9

    def test_detect_faces_with_array(self):
        """Test face detection with numpy array"""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)
        ]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            faces = detector.detect_faces(image)

            assert len(faces) == 1
            mock_detector.detect.assert_called_once()

    def test_detect_method(self):
        """Test detect method returning FaceDetection objects"""
        mock_detector = Mock()
        expected_detections = [FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)]
        mock_detector.detect.return_value = expected_detections

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = detector.detect(image)

            assert detections == expected_detections

    def test_extract_encodings(self):
        """Test encoding extraction"""
        mock_detector = Mock()
        mock_detector.extract_encodings.return_value = [np.random.rand(128)]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = [FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)]

            encodings = detector.extract_encodings(image, detections)

            assert len(encodings) == 1
            mock_detector.extract_encodings.assert_called_once_with(image, detections)

    def test_extract_encodings_auto_detect(self):
        """Test encoding extraction with automatic detection"""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)
        ]
        mock_detector.extract_encodings.return_value = [np.random.rand(128)]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Don't provide detections, should auto-detect
            encodings = detector.extract_encodings(image)

            assert len(encodings) == 1
            mock_detector.detect.assert_called_once()
            mock_detector.extract_encodings.assert_called_once()

    def test_get_available_backends(self):
        """Test getting available backends"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.MTCNN_AVAILABLE", True):
                detector = FaceDetector(backend="face_recognition")
                backends = detector.get_available_backends()

                assert "opencv" in backends
                assert "face_recognition" in backends
                assert "mtcnn" in backends

    def test_get_available_backends_limited(self):
        """Test getting available backends with limited libraries"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", False):
            with patch("src.core.face_detector.MTCNN_AVAILABLE", False):
                detector = FaceDetector(backend="opencv")
                backends = detector.get_available_backends()

                assert backends == ["opencv"]

    def test_draw_faces_with_face_detection_objects(self, sample_image_array):
        """Test drawing faces with FaceDetection objects"""
        with patch("cv2.imwrite") as mock_imwrite:
            detector = FaceDetector(backend="opencv")

            faces = [
                FaceDetection(
                    bbox=(100, 200, 200, 100),
                    confidence=0.9,
                    landmarks={"left_eye": (120, 130), "right_eye": (180, 130)},
                )
            ]

            result_image = detector.draw_faces(sample_image_array, faces)

            assert result_image.shape == sample_image_array.shape

    def test_draw_faces_with_dict_format(self, sample_image_array):
        """Test drawing faces with dictionary format"""
        detector = FaceDetector(backend="opencv")

        faces = [
            {"bbox": (100, 200, 200, 100), "confidence": 0.9, "landmarks": {"left_eye": (120, 130)}}
        ]

        result_image = detector.draw_faces(sample_image_array, faces)

        assert result_image.shape == sample_image_array.shape

    def test_batch_detect(self, sample_image):
        """Test batch detection on multiple images"""
        mock_detector = Mock()
        mock_detector.detect.return_value = [
            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)
        ]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            # Mock image loading
            with patch.object(detector, "_load_image") as mock_load:
                mock_load.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                image_paths = [sample_image, sample_image]
                results = detector.batch_detect(image_paths)

                assert len(results) == 2
                assert all(len(faces) == 1 for faces in results.values())

    def test_get_face_encodings_backward_compatibility(self, sample_image_array):
        """Test backward compatibility method for getting encodings"""
        mock_detector = Mock()
        mock_detector.extract_encodings.return_value = [np.random.rand(128)]

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector(backend="face_recognition")
            detector.detector = mock_detector

            faces = [{"bbox": (100, 200, 200, 100), "confidence": 0.9}]
            encodings = detector.get_face_encodings(sample_image_array, faces)

            assert len(encodings) == 1
            assert encodings[0].shape == (128,)

    def test_create_from_config(self):
        """Test creating detector from configuration"""
        config = {"backend": "face_recognition", "model": "cnn", "min_face_size": 50}

        with patch("src.core.face_detector.FaceRecognitionDetector"):
            detector = FaceDetector.create_from_config(config)

            assert detector.backend_name == "face_recognition"


class TestImageProcessing:
    """Test image processing utilities"""

    def test_load_image_from_path(self, sample_image):
        """Test loading image from file path"""
        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            mock_fr = Mock()
            mock_fr.load_image_file.return_value = np.random.randint(
                0, 255, (480, 640, 3), dtype=np.uint8
            )

            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceDetector(backend="face_recognition")

                image = detector._load_image(sample_image)

                assert isinstance(image, np.ndarray)
                assert len(image.shape) == 3
                mock_fr.load_image_file.assert_called_once()

    def test_load_image_opencv_fallback(self, sample_image):
        """Test loading image with OpenCV fallback"""
        with patch("cv2.imread") as mock_imread:
            with patch("cv2.cvtColor") as mock_cvtcolor:
                mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                mock_cvtcolor.return_value = np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8
                )

                detector = FaceDetector(backend="opencv")

                image = detector._load_image(sample_image)

                assert isinstance(image, np.ndarray)
                mock_imread.assert_called_once()
                mock_cvtcolor.assert_called_once()

    def test_load_image_failure(self):
        """Test handling of image loading failure"""
        with patch("cv2.imread", return_value=None):
            detector = FaceDetector(backend="opencv")

            with pytest.raises(ValueError, match="Could not load image"):
                detector._load_image("/nonexistent/path.jpg")

    def test_load_image_array_passthrough(self):
        """Test that numpy arrays are passed through unchanged"""
        detector = FaceDetector(backend="opencv")

        image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector._load_image(image_array)

        assert np.array_equal(result, image_array)


@pytest.mark.integration
class TestFaceDetectorIntegration:
    """Integration tests for face detection"""

    def test_end_to_end_detection_workflow(self, sample_image_array):
        """Test complete detection workflow"""
        # Mock all dependencies
        mock_fr = Mock()
        mock_fr.face_locations.return_value = [(100, 200, 200, 100)]
        mock_fr.face_encodings.return_value = [np.random.rand(128)]

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceDetector(backend="face_recognition")

                # Detect faces
                detections = detector.detect(sample_image_array)
                assert len(detections) == 1

                # Extract encodings
                encodings = detector.extract_encodings(sample_image_array, detections)
                assert len(encodings) == 1

                # Draw faces
                annotated = detector.draw_faces(sample_image_array, detections)
                assert annotated.shape == sample_image_array.shape

    def test_multiple_backend_comparison(self, sample_image_array):
        """Test comparing results from different backends"""
        # Mock all backends to return the same detection
        expected_bbox = (100, 200, 200, 100)

        # Test face_recognition backend
        mock_fr = Mock()
        mock_fr.face_locations.return_value = [expected_bbox]

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector_fr = FaceDetector(backend="face_recognition")
                detections_fr = detector_fr.detect(sample_image_array)

        # Test OpenCV backend
        mock_cascade = Mock()
        mock_cascade.empty.return_value = False
        mock_cascade.detectMultiScale.return_value = [(100, 100, 100, 100)]  # x,y,w,h

        with patch("cv2.CascadeClassifier", return_value=mock_cascade):
            detector_cv = FaceDetector(backend="opencv")
            detections_cv = detector_cv.detect(sample_image_array)

        # Both should detect faces
        assert len(detections_fr) > 0
        assert len(detections_cv) > 0

        # Both should return FaceDetection objects
        assert all(isinstance(d, FaceDetection) for d in detections_fr)
        assert all(isinstance(d, FaceDetection) for d in detections_cv)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_image(self):
        """Test detection on empty image"""
        with patch("src.core.face_detector.FaceRecognitionDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.detect.return_value = []
            mock_detector_class.return_value = mock_detector

            detector = FaceDetector(backend="face_recognition")

            empty_image = np.zeros((10, 10, 3), dtype=np.uint8)
            detections = detector.detect(empty_image)

            assert detections == []

    def test_grayscale_image_handling(self):
        """Test handling of grayscale images"""
        with patch("src.core.face_detector.FaceRecognitionDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.detect.return_value = []
            mock_detector_class.return_value = mock_detector

            detector = FaceDetector(backend="face_recognition")

            # Should handle grayscale images
            gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
            detections = detector.detect(gray_image)

            # Should still work (returns empty list in this mock case)
            assert isinstance(detections, list)

    def test_very_large_image(self):
        """Test detection on very large image"""
        with patch("src.core.face_detector.FaceRecognitionDetector") as mock_detector_class:
            mock_detector = Mock()
            mock_detector.detect.return_value = []
            mock_detector_class.return_value = mock_detector

            detector = FaceDetector(backend="face_recognition")

            # Large image
            large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
            detections = detector.detect(large_image)

            assert isinstance(detections, list)

    def test_detection_with_corrupted_bbox(self):
        """Test handling of corrupted bounding box data"""
        mock_fr = Mock()
        # Return invalid bbox that would cause issues
        mock_fr.face_locations.return_value = [(-10, -5, 5, 10)]  # Negative coordinates

        with patch("src.core.face_detector.FACE_RECOGNITION_AVAILABLE", True):
            with patch("src.core.face_detector.face_recognition", mock_fr):
                detector = FaceDetector(backend="face_recognition")

                image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

                # Should handle gracefully
                detections = detector.detect(image)

                # May return empty list or corrected detections
                assert isinstance(detections, list)
