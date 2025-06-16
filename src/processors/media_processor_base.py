from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.processors.base_processor import BaseProcessor
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


class MediaProcessorBase(BaseProcessor):
    """
    Base class for media processors (image and video)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize media processor

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)

        # Media-specific configuration
        self.output_format = self.config.get("output_format", "json")
        self.save_chips = self.config.get("save_chips", True)
        self.chip_quality = self.config.get("chip_quality", 85)

        # Processing components (to be initialized by subclasses)
        self.face_detector = None
        self.face_clusterer = None
        self.metadata_extractor = None
        self.chip_generator = None

    @abstractmethod
    def load_media(self, file_path: Path) -> Any:
        """
        Load media file (image or video)

        Args:
            file_path: Path to media file

        Returns:
            Loaded media object
        """
        pass

    @abstractmethod
    def extract_frames(self, media: Any) -> List[np.ndarray]:
        """
        Extract frames from media

        Args:
            media: Loaded media object

        Returns:
            List of frames as numpy arrays
        """
        pass

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame

        Args:
            frame: Frame as numpy array

        Returns:
            List of face detections with bounding boxes
        """
        if self.face_detector is None:
            raise RuntimeError("Face detector not initialized")

        return self.face_detector.detect(frame)

    def extract_face_embeddings(
        self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """
        Extract face embeddings for clustering

        Args:
            frame: Frame containing faces
            face_locations: List of face bounding boxes

        Returns:
            List of face embeddings
        """
        if self.face_clusterer is None:
            raise RuntimeError("Face clusterer not initialized")

        return self.face_clusterer.extract_embeddings(frame, face_locations)

    def generate_chips(
        self, frame: np.ndarray, face_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate face chips from detections

        Args:
            frame: Source frame
            face_data: List of face detection data

        Returns:
            List of chip data with arrays and metadata
        """
        if self.chip_generator is None:
            raise RuntimeError("Chip generator not initialized")

        return self.chip_generator.generate(frame, face_data)

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from media file

        Args:
            file_path: Path to media file

        Returns:
            Extracted metadata
        """
        if self.metadata_extractor is None:
            raise RuntimeError("Metadata extractor not initialized")

        return self.metadata_extractor.extract(file_path)

    def process_frame(self, frame: np.ndarray, frame_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single frame

        Args:
            frame: Frame to process
            frame_info: Frame metadata (timestamp, index, etc.)

        Returns:
            Processing results for the frame
        """
        results = {"frame_info": frame_info, "faces": [], "processing_time": 0.0}

        start_time = datetime.utcnow()

        try:
            # Detect faces
            face_detections = self.detect_faces(frame)

            if face_detections:
                # Extract embeddings for clustering
                face_locations = [f["bounds"] for f in face_detections]
                embeddings = self.extract_face_embeddings(frame, face_locations)

                # Add embeddings to face data
                for face, embedding in zip(face_detections, embeddings):
                    face["embedding"] = embedding

                # Generate chips if enabled
                if self.save_chips:
                    chips = self.generate_chips(frame, face_detections)
                    for face, chip in zip(face_detections, chips):
                        face["chip_data"] = chip

                results["faces"] = face_detections

        except Exception as e:
            logger.error(
                f"Error processing frame", frame_index=frame_info.get("index"), error=str(e)
            )
            self.handle_error(e, {"frame_info": frame_info})

        results["processing_time"] = (datetime.utcnow() - start_time).total_seconds()

        return results

    def aggregate_frame_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple frames

        Args:
            frame_results: List of individual frame results

        Returns:
            Aggregated results
        """
        aggregated = {
            "total_frames": len(frame_results),
            "frames_with_faces": sum(1 for r in frame_results if r["faces"]),
            "total_faces": sum(len(r["faces"]) for r in frame_results),
            "processing_time": sum(r["processing_time"] for r in frame_results),
            "frame_results": frame_results,
        }

        return aggregated

    def prepare_output(
        self, results: Dict[str, Any], file_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare final output structure

        Args:
            results: Processing results
            file_metadata: File metadata

        Returns:
            Final output dictionary
        """
        output = {
            "file": file_metadata["name"],
            "type": self._get_media_type(),
            "name": file_metadata["name"],
            "author": file_metadata.get("author", "facial-vision-system"),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {"file_info": file_metadata, "processing_stats": results, "chips": []},
            "topics": ["face_detected", f"{self._get_media_type()}_analysis"],
        }

        return output

    def _get_media_type(self) -> str:
        """Get media type for output"""
        return "media"  # Override in subclasses

    def validate_frame_quality(self, frame: np.ndarray) -> bool:
        """
        Validate frame quality for processing

        Args:
            frame: Frame to validate

        Returns:
            True if frame quality is acceptable
        """
        # Check minimum dimensions
        min_dimension = min(frame.shape[:2])
        if min_dimension < 100:
            return False

        # Check if frame is not too dark
        mean_brightness = np.mean(frame)
        if mean_brightness < 10:
            return False

        # Check if frame is not too bright
        if mean_brightness > 245:
            return False

        return True

    def calculate_frame_hash(self, frame: np.ndarray) -> str:
        """
        Calculate hash of a frame for deduplication

        Args:
            frame: Frame to hash

        Returns:
            Hash string
        """
        import hashlib

        # Resize frame to standard size for consistent hashing
        import cv2

        small_frame = cv2.resize(frame, (64, 64))

        # Convert to bytes and hash
        frame_bytes = small_frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()


class MediaBatchProcessor(MediaProcessorBase):
    """
    Base class for batch processing of media files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.batch_results = []

    def process_media_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process a batch of media files

        Args:
            file_paths: List of media file paths

        Returns:
            List of processing results
        """
        results = []

        for i, file_path in enumerate(file_paths):
            logger.info(f"Processing file {i+1}/{len(file_paths)}", file=str(file_path))

            try:
                result = self.process(file_path)
                result["success"] = True
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process file", file=str(file_path), error=str(e))
                results.append({"file": str(file_path), "success": False, "error": str(e)})

        self.batch_results = results
        return results

    def save_batch_results(self, output_dir: Path) -> Path:
        """
        Save batch processing results

        Args:
            output_dir: Output directory

        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"batch_results_{timestamp}.json"

        summary = {
            "processing_timestamp": datetime.utcnow().isoformat(),
            "total_files": len(self.batch_results),
            "successful": sum(1 for r in self.batch_results if r.get("success")),
            "failed": sum(1 for r in self.batch_results if not r.get("success")),
            "results": self.batch_results,
        }

        return self.file_handler.save_json_output(summary, results_file)
