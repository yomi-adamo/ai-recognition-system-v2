import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.chip_generator import ChipGenerator
from src.core.face_detector import FaceDetector
from src.core.face_clusterer import FaceClusterer
from src.core.metadata_extractor import MetadataExtractor
from src.processors.media_processor_base import MediaProcessorBase
from src.utils.config import get_config
from src.utils.logger import get_facial_vision_logger, timing_decorator

logger = get_facial_vision_logger(__name__)


class ImageProcessor(MediaProcessorBase):
    """Image processor with face detection, clustering, and metadata extraction"""

    def __init__(self, enable_clustering: bool = True):
        """
        Initialize processor with core components

        Args:
            enable_clustering: Whether to enable face clustering (default: True)
        """
        super().__init__()
        
        config = get_config()

        # Initialize components
        face_config = config.get_face_detection_config()
        self.face_detector = FaceDetector(
            backend=face_config.get("backend", "face_recognition"),
            model=face_config.get("model", "hog"),
            min_face_size=face_config.get("min_face_size", 40),
            confidence_threshold=face_config.get("confidence_threshold", 0.5),
        )

        self.metadata_extractor = MetadataExtractor()
        self.chip_generator = ChipGenerator()

        # Initialize face clusterer if enabled
        self.enable_clustering = enable_clustering
        self.face_clusterer = None

        if enable_clustering:
            try:
                self.face_clusterer = FaceClusterer.create_from_config()
                cluster_stats = self.face_clusterer.get_cluster_statistics()
                logger.info(
                    f"Face clustering enabled: {cluster_stats['total_clusters']} existing clusters"
                )
            except Exception as e:
                logger.warning(f"Could not initialize face clusterer: {e}")
                self.enable_clustering = False

        logger.info(f"ImageProcessor initialized (clustering: {self.enable_clustering})")

    @timing_decorator
    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_chips: bool = True,
        enable_clustering: Optional[bool] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image with face detection, clustering, and metadata extraction

        Args:
            image_path: Path to the image file
            output_dir: Directory to save face chips
            save_chips: Whether to save face chips to disk
            enable_clustering: Override clustering setting for this call
            parent_id: Blockchain parent asset ID

        Returns:
            Dictionary containing complete processing results with metadata and chips
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path}")

        # Determine if clustering should be used
        use_clustering = (
            enable_clustering if enable_clustering is not None else self.enable_clustering
        )
        use_clustering = use_clustering and self.face_clusterer is not None

        # Extract source file metadata
        source_metadata = self.metadata_extractor.extract_metadata(image_path)

        # Load image and detect faces
        import cv2
        image_array = cv2.imread(str(image_path))
        if image_array is None:
            raise ValueError(f"Could not load image: {image_path}")

        face_detections = self.face_detector.detect(image_array)
        logger.info(f"Detected {len(face_detections)} faces in {image_path.name}")

        if len(face_detections) == 0:
            logger.info(f"No faces detected in {image_path}")
            return self._create_empty_result(source_metadata, image_path, parent_id)

        # Perform clustering if enabled
        cluster_ids = []
        if use_clustering:
            try:
                cluster_ids = self.face_clusterer.process_faces(image_array, face_detections)
                logger.info(f"Assigned faces to clusters: {cluster_ids}")
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                use_clustering = False

        # If clustering failed, assign sequential IDs
        if not cluster_ids:
            cluster_ids = [f"person_{i+1}" for i in range(len(face_detections))]

        # Create output directory if needed
        if save_chips and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Generate chips with clustering
        chip_results = []
        if save_chips and output_dir:
            try:
                chip_results = self.chip_generator.generate_clustered_chips(
                    image=image_array,
                    face_detections=face_detections,
                    cluster_ids=cluster_ids,
                    output_dir=output_dir
                )
            except Exception as e:
                logger.error(f"Failed to generate clustered chips: {e}")
                chip_results = {}

        # Create comprehensive metadata for each chip
        chips_metadata = []
        for i, (detection, cluster_id) in enumerate(zip(face_detections, cluster_ids)):
            try:
                # Get chip path if generated
                chip_path = None
                if cluster_id in chip_results and chip_results[cluster_id]:
                    chip_path = chip_results[cluster_id][0].get("file_path")

                # Create comprehensive chip metadata
                chip_metadata = self.metadata_extractor.create_chip_metadata(
                    source_file=image_path,
                    chip_path=chip_path or f"chip_{i:03d}.jpg",
                    face_bbox=detection.bbox,
                    cluster_id=cluster_id,
                    confidence=detection.confidence,
                    parent_id=parent_id
                )

                # Add detection-specific data
                chip_metadata.update({
                    "landmarks": detection.landmarks,
                    "face_area": detection.area,
                    "face_center": detection.center,
                })

                chips_metadata.append(chip_metadata)

            except Exception as e:
                logger.error(f"Error creating metadata for face {i}: {e}")
                continue

        # Create final result structure
        result = {
            "file": str(image_path),
            "type": "image",
            "name": image_path.stem,
            "author": "facial-vision-system",
            "timestamp": self.metadata_extractor.get_timestamp(source_metadata),
            "parentId": parent_id,
            "metadata": {
                "source_metadata": source_metadata,
                "processing_stats": {
                    "faces_detected": len(face_detections),
                    "clusters_assigned": len(set(cluster_ids)) if cluster_ids else 0,
                    "clustering_enabled": use_clustering,
                    "chips_generated": len(chips_metadata),
                },
                "chips": chips_metadata,
            },
            "topics": ["face_detected", "image_analysis"],
        }

        # Add GPS if available
        gps_coords = self.metadata_extractor.get_gps_coordinates(source_metadata)
        if gps_coords:
            result["metadata"]["GPS"] = gps_coords

        # Add device info if available
        device_info = source_metadata.get("device")
        if device_info:
            result["metadata"]["device"] = device_info

        logger.info(
            f"Successfully processed {image_path.name}: "
            f"{len(face_detections)} faces, {len(set(cluster_ids))} clusters"
        )

        return result

    def _create_empty_result(
        self, source_metadata: Dict[str, Any], image_path: Path, parent_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create result structure for images with no faces detected"""
        return {
            "file": str(image_path),
            "type": "image", 
            "name": image_path.stem,
            "author": "facial-vision-system",
            "timestamp": self.metadata_extractor.get_timestamp(source_metadata),
            "parentId": parent_id,
            "metadata": {
                "source_metadata": source_metadata,
                "processing_stats": {
                    "faces_detected": 0,
                    "clusters_assigned": 0,
                    "clustering_enabled": self.enable_clustering,
                    "chips_generated": 0,
                },
                "chips": [],
            },
            "topics": ["image_analysis"],
        }

    def load_media(self, file_path: Path) -> Any:
        """Load image file"""
        import cv2
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Could not load image: {file_path}")
        return image

    def extract_frames(self, media: Any) -> List[Any]:
        """Extract frames - for images, return single frame"""
        return [media]

    def process(self, input_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process method required by BaseProcessor
        
        Args:
            input_path: Path to input file
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results dictionary
        """
        return self.process_image(input_path, **kwargs)

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        if not self.face_clusterer:
            return {"error": "Clustering not available"}
        return self.face_clusterer.get_cluster_statistics()

    def save_results_to_json(
        self, results: Dict[str, Any], output_path: Union[str, Path]
    ) -> None:
        """Save processing results to JSON file"""
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved processing results to {output_path}")

    def batch_process_images(
        self, image_paths: List[Union[str, Path]], output_dir: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save all outputs

        Returns:
            List of processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for image_path in image_paths:
            try:
                logger.info(f"Processing {image_path}")
                result = self.process_image(
                    image_path=image_path,
                    output_dir=output_dir,
                    save_chips=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
                
        logger.info(f"Batch processed {len(results)} images")
        return results
