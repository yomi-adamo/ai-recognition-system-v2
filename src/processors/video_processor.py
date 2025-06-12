import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Generator, Tuple
import uuid
from datetime import datetime, timedelta
import time

from src.core.face_detector import FaceDetector
from src.core.face_recognizer import FaceRecognizer
from src.core.metadata_extractor import MetadataExtractor
from src.core.chip_generator import ChipGenerator
from src.utils.logger import get_logger, timing_decorator, performance_logger
from src.utils.config import get_config

logger = get_logger(__name__)


class VideoProcessor:
    """Process videos for face detection with frame sampling and scene change detection"""
    
    def __init__(self, enable_recognition: bool = True):
        """
        Initialize video processor with components and configuration
        
        Args:
            enable_recognition: Whether to enable face recognition
        """
        config = get_config()
        video_config = config.get_video_processing_config()
        face_config = config.get_face_detection_config()
        
        # Initialize components
        self.face_detector = FaceDetector(
            model=face_config.get('model', 'hog'),
            tolerance=face_config.get('tolerance', 0.6),
            min_face_size=face_config.get('min_face_size', 40)
        )
        
        self.metadata_extractor = MetadataExtractor()
        self.chip_generator = ChipGenerator()
        
        # Initialize face recognizer if enabled
        self.enable_recognition = enable_recognition
        self.face_recognizer = None
        
        if enable_recognition:
            try:
                self.face_recognizer = FaceRecognizer()
                logger.info(f"Face recognition enabled for video: {len(self.face_recognizer)} known faces")
            except Exception as e:
                logger.warning(f"Could not initialize face recognizer: {e}")
                self.enable_recognition = False
        
        # Video processing configuration
        self.frame_interval = video_config.get('frame_interval', 30)
        self.scene_change_threshold = video_config.get('scene_change_threshold', 30.0)
        self.max_faces_per_frame = video_config.get('max_faces_per_frame', 20)
        
        # Face tracking for deduplication
        self.known_video_faces = {}  # Store face encodings for deduplication within video
        self.face_similarity_threshold = 0.6
        
        logger.info(f"VideoProcessor initialized: frame_interval={self.frame_interval}, "
                   f"scene_threshold={self.scene_change_threshold}, "
                   f"recognition={self.enable_recognition}")
    
    @timing_decorator
    def process_video(self, video_path: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     save_chips: bool = True,
                     recognize_faces: Optional[bool] = None,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Process video and extract unique faces with timestamps
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save face chips
            save_chips: Whether to save face chips to disk
            recognize_faces: Override recognition setting for this call
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results and metadata
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Determine if recognition should be used
        use_recognition = recognize_faces if recognize_faces is not None else self.enable_recognition
        use_recognition = use_recognition and self.face_recognizer is not None
        
        # Extract video metadata
        metadata = self.metadata_extractor.extract_video_metadata(video_path)
        
        # Generate parent ID for the video
        parent_id = str(uuid.uuid4())
        
        # Clear video-specific face tracking
        self.known_video_faces = {}
        
        # Process video frames
        all_detections = []
        unique_faces = []
        frame_count = 0
        processed_frames = 0
        
        # Track recognized identities
        recognized_identities = set()
        identity_first_seen = {}
        
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
                
                prev_frame = None
                
                # Process frames
                for frame_data in self._extract_frames(cap, total_frames):
                    frame_number = frame_data['frame_number']
                    frame = frame_data['frame']
                    timestamp = frame_data['timestamp']
                    
                    frame_count += 1
                    
                    # Scene change detection
                    is_scene_change = self._detect_scene_change(frame, prev_frame)
                    prev_frame = frame.copy()
                    
                    # Detect faces in frame
                    faces = self.face_detector.detect_faces(frame)
                    
                    if len(faces) > 0:
                        processed_frames += 1
                        
                        # Limit faces per frame
                        if len(faces) > self.max_faces_per_frame:
                            faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
                            faces = faces[:self.max_faces_per_frame]
                            logger.warning(f"Limited to {self.max_faces_per_frame} faces in frame {frame_number}")
                        
                        # Get face encodings for recognition if enabled
                        face_encodings = []
                        if use_recognition:
                            try:
                                face_encodings = self.face_detector.get_face_encodings(frame, faces)
                            except Exception as e:
                                logger.warning(f"Could not generate face encodings in frame {frame_number}: {e}")
                        
                        # Process each face
                        for face_idx, face in enumerate(faces):
                            # Perform face recognition if enabled
                            identity = "unknown"
                            recognition_confidence = 0.0
                            recognition_data = {}
                            
                            if use_recognition and face_idx < len(face_encodings):
                                try:
                                    recognition_result = self.face_recognizer.recognize_face(face_encodings[face_idx])
                                    identity = recognition_result.get('name', 'unknown')
                                    recognition_confidence = recognition_result.get('confidence', 0.0)
                                    
                                    # Track first appearance of recognized faces
                                    if identity != "unknown" and identity not in identity_first_seen:
                                        identity_first_seen[identity] = {
                                            'frame_number': frame_number,
                                            'timestamp': timestamp,
                                            'video_timestamp': frame_data['video_timestamp']
                                        }
                                        recognized_identities.add(identity)
                                    
                                    # Build recognition data
                                    recognition_data = {
                                        'identity': identity,
                                        'recognition_confidence': recognition_confidence,
                                        'recognition_distance': recognition_result.get('distance', 1.0),
                                        'recognition_method': 'face_recognition_lib'
                                    }
                                    
                                    logger.debug(f"Frame {frame_number}, Face {face_idx}: "
                                               f"recognized as '{identity}' (confidence: {recognition_confidence:.2%})")
                                    
                                except Exception as e:
                                    logger.warning(f"Recognition failed for face {face_idx} in frame {frame_number}: {e}")
                            
                            # Generate chip
                            chip_data = None
                            if save_chips and output_dir:
                                chip_output = Path(output_dir) / f"frame_{frame_number:06d}_face_{face_idx:02d}.jpg"
                                chip_data = self.chip_generator.generate_chip(
                                    frame, face['bbox'], output_path=chip_output
                                )
                            else:
                                chip_data = self.chip_generator.generate_chip(
                                    frame, face['bbox']
                                )
                            
                            # Create detection record
                            detection = {
                                'frame_number': frame_number,
                                'timestamp': timestamp,
                                'video_timestamp': frame_data['video_timestamp'],
                                'face_info': face,
                                'chip_data': chip_data,
                                'is_scene_change': is_scene_change,
                                'source_file': video_path.name,
                                'parent_id': parent_id,
                                'identity': identity,
                                'recognition_data': recognition_data
                            }
                            
                            all_detections.append(detection)
                            
                            # Check if this is a unique face (considering identity)
                            is_unique = self._is_unique_face(face, frame, identity)
                            if is_unique:
                                unique_faces.append(detection)
                                logger.debug(f"Found unique face in frame {frame_number}: {identity}")
                    
                    # Progress callback
                    if progress_callback and frame_count % 10 == 0:
                        progress = frame_count / (total_frames // self.frame_interval)
                        progress_callback(min(progress, 1.0))
                
            finally:
                cap.release()
        
        # Update face recognition statistics if used
        if use_recognition and self.face_recognizer:
            try:
                self.face_recognizer.save_database()
            except Exception as e:
                logger.warning(f"Could not save recognition database: {e}")
        
        # Create result summary
        result = {
            'video_metadata': metadata,
            'processing_stats': {
                'total_frames_in_video': total_frames,
                'frames_processed': frame_count,
                'frames_with_faces': processed_frames,
                'total_face_detections': len(all_detections),
                'unique_faces': len(unique_faces),
                'recognized_identities': list(recognized_identities),
                'unknown_faces': len([d for d in all_detections if d['identity'] == 'unknown']),
                'fps': fps,
                'duration_seconds': duration,
                'frame_interval': self.frame_interval,
                'recognition_enabled': use_recognition
            },
            'all_detections': all_detections,
            'unique_faces': unique_faces,
            'parent_id': parent_id,
            'identity_timeline': identity_first_seen
        }
        
        logger.info(f"Video processing complete: {len(all_detections)} total detections, "
                   f"{len(unique_faces)} unique faces, "
                   f"{len(recognized_identities)} recognized identities")
        
        return result
    
    def _extract_frames(self, cap: cv2.VideoCapture, 
                       total_frames: int) -> Generator[Dict[str, Any], None, None]:
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
                'frame_number': frame_number,
                'frame': frame,
                'timestamp': timestamp.isoformat(),
                'video_timestamp': timestamp_seconds
            }
            
            # Move to next frame interval
            frame_number += self.frame_interval
            
            if frame_number >= total_frames:
                break
    
    def _detect_scene_change(self, current_frame: np.ndarray, 
                           previous_frame: Optional[np.ndarray]) -> bool:
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
    
    def _is_unique_face(self, face: Dict[str, Any], frame: np.ndarray, 
                       identity: str = "unknown") -> bool:
        """
        Check if face is unique (not similar to previously detected faces)
        
        Args:
            face: Face detection dictionary
            frame: Frame containing the face
            identity: Recognized identity (if any)
            
        Returns:
            True if face is unique
        """
        # If we have a recognized identity, use that for uniqueness
        if identity != "unknown":
            # Check if we've seen this identity before in this video
            face_key = f"identity_{identity}"
            if face_key in self.known_video_faces:
                return False  # Already seen this person
            else:
                self.known_video_faces[face_key] = True
                return True
        
        # For unknown faces, use encoding similarity
        try:
            # Get face encoding
            face_encoding = self.face_detector.get_face_encodings(frame, [face])
            
            if len(face_encoding) == 0:
                return True  # Assume unique if encoding fails
            
            encoding = face_encoding[0]
            
            # Compare with known faces in this video
            for known_id, known_encoding in self.known_video_faces.items():
                if known_id.startswith("unknown_"):
                    try:
                        import face_recognition
                        matches = face_recognition.compare_faces(
                            [known_encoding], encoding, self.face_similarity_threshold
                        )
                        
                        if matches[0]:
                            logger.debug(f"Face matches known video face {known_id}")
                            return False
                            
                    except ImportError:
                        # Fallback to simple comparison if face_recognition not available
                        distance = np.linalg.norm(np.array(known_encoding) - np.array(encoding))
                        if distance < self.face_similarity_threshold:
                            return False
            
            # Add to known faces for this video
            face_id = f"unknown_{str(uuid.uuid4())[:8]}"
            self.known_video_faces[face_id] = encoding
            logger.debug(f"Added new unique unknown face: {face_id}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in face uniqueness check: {e}")
            return True  # Assume unique on error
    
    def create_timeline_json(self, processing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create timeline JSON with all face detections
        
        Args:
            processing_result: Result from process_video
            
        Returns:
            List of JSON objects for all detections
        """
        timeline = []
        
        for detection in processing_result['all_detections']:
            face_info = detection['face_info']
            chip_data = detection['chip_data']
            
            # Create JSON object
            face_json = {
                "file": chip_data.get('base64', chip_data.get('file_path', '')),
                "type": "video_frame",
                "name": chip_data['name'],
                "author": "facial-vision-system",
                "parentId": detection['parent_id'],
                "metadata": {
                    "timestamp": detection['timestamp'],
                    "video_timestamp": detection['video_timestamp'],
                    "frame_number": detection['frame_number'],
                    "confidence": face_info['confidence'],
                    "identity": detection.get('identity', 'unknown'),
                    "source_file": detection['source_file'],
                    "face_bounds": {
                        "x": face_info['bbox'][3],  # left
                        "y": face_info['bbox'][0],  # top
                        "w": face_info['bbox'][1] - face_info['bbox'][3],  # width
                        "h": face_info['bbox'][2] - face_info['bbox'][0]   # height
                    },
                    "is_scene_change": detection['is_scene_change']
                },
                "topics": ["face", "biometric", "person", "video"]
            }
            
            # Add recognition data if available
            if 'recognition_data' in detection and detection['recognition_data']:
                face_json["metadata"].update(detection['recognition_data'])
            
            timeline.append(face_json)
        
        return timeline
    
    def get_unique_faces_json(self, processing_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get JSON for unique faces only
        
        Args:
            processing_result: Result from process_video
            
        Returns:
            List of JSON objects for unique faces
        """
        unique_faces = []
        
        for detection in processing_result['unique_faces']:
            face_info = detection['face_info']
            chip_data = detection['chip_data']
            
            # Create JSON object
            face_json = {
                "file": chip_data.get('base64', chip_data.get('file_path', '')),
                "type": "video_frame",
                "name": chip_data['name'],
                "author": "facial-vision-system",
                "parentId": detection['parent_id'],
                "metadata": {
                    "timestamp": detection['timestamp'],
                    "video_timestamp": detection['video_timestamp'],
                    "frame_number": detection['frame_number'],
                    "confidence": face_info['confidence'],
                    "identity": detection.get('identity', 'unknown'),
                    "source_file": detection['source_file'],
                    "face_bounds": {
                        "x": face_info['bbox'][3],  # left
                        "y": face_info['bbox'][0],  # top
                        "w": face_info['bbox'][1] - face_info['bbox'][3],  # width
                        "h": face_info['bbox'][2] - face_info['bbox'][0]   # height
                    },
                    "best_frame": True  # Mark as best quality frame for this face
                },
                "topics": ["face", "biometric", "person", "video", "unique"]
            }
            
            # Add recognition data if available
            if 'recognition_data' in detection and detection['recognition_data']:
                face_json["metadata"].update(detection['recognition_data'])
                
                # Add special topic for recognized faces
                if detection.get('identity', 'unknown') != 'unknown':
                    face_json["topics"].append("recognized")
            
            unique_faces.append(face_json)
        
        return unique_faces