import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.core.chip_generator import ChipGenerator
from src.core.face_detector import FaceDetector
from src.core.face_clusterer import FaceClusterer
from src.core.metadata_extractor import MetadataExtractor
from src.processors.media_processor_base import MediaProcessorBase
from src.utils.config import get_config
from src.utils.logger import get_facial_vision_logger, performance_logger, timing_decorator

logger = get_facial_vision_logger(__name__)


class VideoProcessor(MediaProcessorBase):
    """Process videos for face detection with frame sampling and scene change detection"""

    def __init__(self, enable_clustering: bool = True):
        """
        Initialize video processor with components and configuration

        Args:
            enable_clustering: Whether to enable face clustering
        """
        super().__init__()
        
        config = get_config()
        video_config = config.get_video_processing_config()
        face_config = config.get_face_detection_config()

        # Initialize components
        self.face_detector = FaceDetector(
            backend=face_config.get("backend", "face_recognition"),
            min_face_size=face_config.get("min_face_size", 40),
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
                    f"Face clustering enabled for video: {cluster_stats['total_clusters']} existing clusters"
                )
            except Exception as e:
                logger.warning(f"Could not initialize face clusterer: {e}")
                self.enable_clustering = False

        # Video processing configuration
        self.frame_interval = video_config.get("frame_interval", 30)
        self.scene_change_threshold = video_config.get("scene_change_threshold", 30.0)
        self.max_faces_per_frame = video_config.get("max_faces_per_frame", 20)

        logger.info(
            f"VideoProcessor initialized: frame_interval={self.frame_interval}, "
            f"scene_threshold={self.scene_change_threshold}, "
            f"clustering={self.enable_clustering}"
        )

    @timing_decorator
    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_chips: bool = True,
        enable_clustering: Optional[bool] = None,
        parent_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process video with face detection, clustering, and frame extraction

        Args:
            video_path: Path to video file
            output_dir: Directory to save face chips
            save_chips: Whether to save face chips to disk
            enable_clustering: Override clustering setting for this call
            parent_id: Blockchain parent asset ID
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with comprehensive processing results and metadata
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Processing video: {video_path}")

        # Determine if clustering should be used
        use_clustering = (
            enable_clustering if enable_clustering is not None else self.enable_clustering
        )
        use_clustering = use_clustering and self.face_clusterer is not None

        # Extract video metadata
        source_metadata = self.metadata_extractor.extract_metadata(video_path)

        # Generate parent ID if not provided
        if parent_id is None:
            parent_id = str(uuid.uuid4())

        # Collect all frame data for batch processing
        frame_detections = []
        all_faces = []
        all_frames = []

        with performance_logger(f"video_processing_{video_path.name}"):
            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            try:
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0

                logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.1f}s")

                frame_count = 0
                processed_frames = 0

                # Process frames and collect detections
                for frame_data in self._extract_frames(cap, total_frames):
                    frame_number = frame_data["frame_number"]
                    frame = frame_data["frame"]
                    timestamp = frame_data["timestamp"]

                    frame_count += 1

                    # Detect faces in frame
                    face_detections = self.face_detector.detect(frame)

                    if len(face_detections) > 0:
                        processed_frames += 1

                        # Limit faces per frame
                        if len(face_detections) > self.max_faces_per_frame:
                            face_detections = sorted(
                                face_detections, key=lambda x: x.confidence, reverse=True
                            )
                            face_detections = face_detections[: self.max_faces_per_frame]
                            logger.warning(
                                f"Limited to {self.max_faces_per_frame} faces in frame {frame_number}"
                            )

                        # Store frame data for clustering
                        for detection in face_detections:
                            all_faces.append(detection)
                            all_frames.append(frame)
                            frame_detections.append({
                                "frame_number": frame_number,
                                "timestamp": timestamp,
                                "video_timestamp": frame_data["video_timestamp"],
                                "detection": detection,
                                "frame": frame,
                            })

                    # Progress callback
                    if progress_callback and frame_count % 10 == 0:
                        progress = frame_count / (total_frames // self.frame_interval)
                        progress_callback(min(progress, 1.0))

            finally:
                cap.release()

        logger.info(f"Detected {len(all_faces)} total faces across {processed_frames} frames")

        # Perform clustering on all faces if enabled
        cluster_ids = []
        if use_clustering and all_faces:
            try:
                # Process faces frame by frame to get correct embeddings
                cluster_ids = []
                
                # Group faces by frame for efficient processing
                from collections import defaultdict
                frame_face_map = defaultdict(list)
                face_indices = []
                
                for idx, (frame_data, face) in enumerate(zip(frame_detections, all_faces)):
                    frame_num = frame_data['frame_number']
                    frame_face_map[frame_num].append((idx, face, frame_data['frame']))
                    face_indices.append(idx)
                
                # Process each frame's faces
                temp_cluster_ids = [None] * len(all_faces)
                
                for frame_num, face_data_list in frame_face_map.items():
                    if not face_data_list:
                        continue
                        
                    # Get the frame and faces for this frame
                    indices = [data[0] for data in face_data_list]
                    faces = [data[1] for data in face_data_list]
                    frame = face_data_list[0][2]  # All entries have the same frame
                    
                    # Process this frame's faces
                    frame_cluster_ids = self.face_clusterer.process_faces(frame, faces)
                    
                    # Map back to original indices
                    for idx, cluster_id in zip(indices, frame_cluster_ids):
                        temp_cluster_ids[idx] = cluster_id
                
                cluster_ids = temp_cluster_ids
                logger.info(f"Clustered faces into {len(set(cluster_ids))} clusters")
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                use_clustering = False

        # If clustering failed, assign sequential IDs
        if not cluster_ids:
            cluster_ids = [f"person_{i+1}" for i in range(len(all_faces))]

        # Create output directory if needed
        if save_chips and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Generate chips and metadata for each detection
        chips_metadata = []
        for i, (frame_data, cluster_id) in enumerate(zip(frame_detections, cluster_ids)):
            try:
                detection = frame_data["detection"]
                frame = frame_data["frame"]

                # Generate chip with cluster organization
                chip_path = None
                if save_chips and output_dir:
                    cluster_dir = output_dir / cluster_id
                    cluster_dir.mkdir(exist_ok=True)
                    chip_path = cluster_dir / f"frame_{frame_data['frame_number']:06d}_chip_{i:03d}.jpg"

                    chip_data = self.chip_generator.generate_chip(
                        image=frame,
                        face_bbox=detection.bbox,
                        output_path=chip_path,
                        cluster_id=cluster_id
                    )
                else:
                    chip_data = self.chip_generator.generate_chip(
                        image=frame,
                        face_bbox=detection.bbox,
                        cluster_id=cluster_id
                    )

                # Create comprehensive chip metadata with frame-specific GPS
                chip_metadata = self.metadata_extractor.create_chip_metadata(
                    source_file=video_path,
                    chip_path=chip_path or f"frame_{frame_data['frame_number']:06d}_chip_{i:03d}.jpg",
                    face_bbox=detection.bbox,
                    cluster_id=cluster_id,
                    confidence=detection.confidence,
                    frame_number=frame_data["frame_number"],
                    video_timestamp=frame_data["video_timestamp"],
                    parent_id=parent_id,
                    frame_specific_gps=True  # Enable frame-specific GPS extraction
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

        # Create comprehensive result structure
        result = {
            "file": str(video_path),
            "type": "video",
            "name": video_path.stem,
            "author": "facial-vision-system",
            "timestamp": self.metadata_extractor.get_timestamp(source_metadata),
            "parentId": parent_id,
            "metadata": {
                "source_metadata": source_metadata,
                "processing_stats": {
                    "total_frames_in_video": total_frames,
                    "frames_processed": frame_count,
                    "frames_with_faces": processed_frames,
                    "total_face_detections": len(all_faces),
                    "clusters_assigned": len(set(cluster_ids)) if cluster_ids else 0,
                    "clustering_enabled": use_clustering,
                    "chips_generated": len(chips_metadata),
                    "fps": fps,
                    "duration_seconds": duration,
                    "frame_interval": self.frame_interval,
                },
                "video_info": {
                    "fps": fps,
                    "duration_seconds": duration,
                    "total_frames": total_frames,
                    "resolution": f"{source_metadata.get('video', {}).get('width', 0)}x{source_metadata.get('video', {}).get('height', 0)}"
                },
                "chips": chips_metadata,
            },
            "topics": ["face_detected", "video_analysis"],
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
            f"Video processing complete: {len(all_faces)} faces, "
            f"{len(set(cluster_ids))} clusters, {len(chips_metadata)} chips generated"
        )

        return result

    def _extract_frames(
        self, cap: cv2.VideoCapture, total_frames: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Extract frames from video at specified intervals

        Args:
            cap: OpenCV video capture object
            total_frames: Total number of frames in video

        Yields:
            Dictionary with frame data
        """
        frame_number = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp_seconds = frame_number / fps if fps > 0 else 0
            timestamp = datetime.now() + timedelta(seconds=timestamp_seconds)

            yield {
                "frame_number": frame_number,
                "frame": frame,
                "timestamp": timestamp.isoformat(),
                "video_timestamp": timestamp_seconds,
            }

            # Move to next frame interval
            frame_number += self.frame_interval

            if frame_number >= total_frames:
                break

    def _detect_scene_change(
        self, current_frame: np.ndarray, previous_frame: Optional[np.ndarray]
    ) -> bool:
        """
        Detect scene changes using frame differencing

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            True if scene change detected
        """
        if previous_frame is None:
            return True

        # Convert to grayscale
        gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate histogram difference
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Compare histograms
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Scene change if correlation is below threshold
        scene_change = correlation < (self.scene_change_threshold / 100.0)

        if scene_change:
            logger.debug(f"Scene change detected: correlation={correlation:.3f}")

        return scene_change

    def load_media(self, file_path: Path) -> Any:
        """Load video file"""
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")
        return cap

    def extract_frames(self, media: Any) -> List[Any]:
        """Extract frames from video"""
        frames = []
        cap = media
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            
            # Limit frames to prevent memory issues
            if frame_count > 1000:
                break
                
        cap.release()
        return frames

    def process(self, input_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process method required by BaseProcessor
        
        Args:
            input_path: Path to input file
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results dictionary
        """
        return self.process_video(input_path, **kwargs)

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

        logger.info(f"Saved video processing results to {output_path}")

    def extract_frame_at_timestamp(
        self, video_path: Union[str, Path], timestamp_seconds: float
    ) -> Optional[np.ndarray]:
        """
        Extract a specific frame at given timestamp

        Args:
            video_path: Path to video file
            timestamp_seconds: Timestamp in seconds

        Returns:
            Frame as numpy array or None if failed
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            return frame if ret else None
            
        finally:
            cap.release()
