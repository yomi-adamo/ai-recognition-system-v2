import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import uuid
from datetime import datetime

from src.core.face_detector import FaceDetector
from src.core.face_recognizer import FaceRecognizer
from src.core.metadata_extractor import MetadataExtractor
from src.core.chip_generator import ChipGenerator
from src.utils.logger import get_logger, timing_decorator
from src.utils.config import get_config

logger = get_logger(__name__)


class ImageProcessor:
    """Main processor that combines face detection, recognition, metadata extraction, and chip generation"""
    
    def __init__(self, enable_recognition: bool = True):
        """
        Initialize processor with core components
        
        Args:
            enable_recognition: Whether to enable face recognition (default: True)
        """
        config = get_config()
        
        # Initialize components
        face_config = config.get_face_detection_config()
        self.face_detector = FaceDetector(
            model=face_config.get('model', 'hog'),
            tolerance=face_config.get('tolerance', 0.8),
            min_face_size=face_config.get('min_face_size', 20)
        )
        
        self.metadata_extractor = MetadataExtractor()
        self.chip_generator = ChipGenerator()
        
        # Initialize face recognizer if enabled
        self.enable_recognition = enable_recognition
        self.face_recognizer = None
        
        if enable_recognition:
            try:
                self.face_recognizer = FaceRecognizer()
                logger.info(f"Face recognition enabled: {len(self.face_recognizer)} known faces")
            except Exception as e:
                logger.warning(f"Could not initialize face recognizer: {e}")
                self.enable_recognition = False
        
        logger.info(f"ImageProcessor initialized (recognition: {self.enable_recognition})")
    
    @timing_decorator
    def process_image(self, image_path: Union[str, Path], 
                     output_dir: Optional[Union[str, Path]] = None,
                     save_chips: bool = True,
                     recognize_faces: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Process a single image and return JSON objects for all detected faces
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save face chips
            save_chips: Whether to save face chips to disk
            recognize_faces: Override recognition setting for this call
            
        Returns:
            List of dictionaries (one per face) with the required JSON format
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Processing image: {image_path}")
        
        # Determine if recognition should be used
        use_recognition = recognize_faces if recognize_faces is not None else self.enable_recognition
        use_recognition = use_recognition and self.face_recognizer is not None
        
        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(image_path)
        
        # Detect faces
        faces = self.face_detector.detect_faces(image_path)
        logger.info(f"Detected {len(faces)} faces in {image_path.name}")
        
        if len(faces) == 0:
            logger.warning(f"No faces detected in {image_path}")
            return []
        
        # Generate parent ID for the original image
        parent_id = str(uuid.uuid4())
        
        # Get face encodings for recognition if enabled
        face_encodings = []
        if use_recognition:
            try:
                face_encodings = self.face_detector.get_face_encodings(image_path, faces)
                logger.debug(f"Generated {len(face_encodings)} face encodings")
            except Exception as e:
                logger.warning(f"Could not generate face encodings: {e}")
                use_recognition = False
        
        # Process each face
        results = []
        
        # Create output directory if needed
        if save_chips and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, face in enumerate(faces):
            try:
                # Generate face chip
                chip_output = None
                if save_chips and output_dir:
                    chip_output = output_dir / f"{image_path.stem}_face_{idx:03d}.jpg"
                
                chip_data = self.chip_generator.generate_chip(
                    str(image_path),
                    face['bbox'],
                    output_path=chip_output
                )
                
                # Perform face recognition if enabled
                identity = "unknown"
                recognition_confidence = 0.0
                recognition_data = {}
                
                if use_recognition and idx < len(face_encodings):
                    try:
                        recognition_result = self.face_recognizer.recognize_face(face_encodings[idx])
                        identity = recognition_result.get('name', 'unknown')
                        recognition_confidence = recognition_result.get('confidence', 0.0)
                        
                        # Include detailed recognition data
                        recognition_data = {
                            'recognition_confidence': recognition_confidence,
                            'recognition_distance': recognition_result.get('distance', 1.0),
                            'recognition_votes': recognition_result.get('votes', 0),
                            'total_encodings': recognition_result.get('total_encodings', 0),
                            'recognition_method': 'face_recognition_lib'
                        }
                        
                        # Include top candidates if available
                        if 'all_candidates' in recognition_result:
                            recognition_data['candidates'] = recognition_result['all_candidates']
                        
                        logger.debug(f"Face {idx}: recognized as '{identity}' "
                                   f"(confidence: {recognition_confidence:.2%})")
                        
                    except Exception as e:
                        logger.warning(f"Recognition failed for face {idx}: {e}")
                
                # Create JSON object
                face_json = self._create_face_json(
                    chip_data=chip_data,
                    face_info=face,
                    metadata=metadata,
                    parent_id=parent_id,
                    source_file=image_path.name,
                    identity=identity,
                    recognition_data=recognition_data
                )
                
                results.append(face_json)
                
            except Exception as e:
                logger.error(f"Error processing face {idx}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} faces from {image_path.name}")
        
        # Update recognition statistics
        if use_recognition and self.face_recognizer:
            try:
                self.face_recognizer.save_database()
            except Exception as e:
                logger.warning(f"Could not save recognition database: {e}")
        
        return results
    
    def _create_face_json(self, chip_data: Dict[str, Any], 
                         face_info: Dict[str, Any],
                         metadata: Dict[str, Any],
                         parent_id: str,
                         source_file: str,
                         identity: str = "unknown",
                         recognition_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create JSON object following the required schema"""
        
        # Get timestamp
        timestamp = self.metadata_extractor.get_timestamp(metadata)
        
        # Get GPS if available
        gps = self.metadata_extractor.get_gps_coordinates(metadata)
        
        # Get face bounds
        top, right, bottom, left = face_info['bbox']
        
        # Build the JSON object
        result = {
            "file": chip_data.get('base64', chip_data.get('file_path', '')),
            "type": "image",
            "name": chip_data['name'],
            "author": "facial-vision-system",
            "parentId": parent_id,
            "metadata": {
                "timestamp": timestamp,
                "confidence": face_info['confidence'],
                "identity": identity,
                "source_file": source_file,
                "face_bounds": {
                    "x": left,
                    "y": top,
                    "w": right - left,
                    "h": bottom - top
                }
            },
            "topics": ["face", "biometric", "person"]
        }
        
        # Add GPS if available
        if gps:
            result["metadata"]["gps"] = gps
        
        # Add camera info if available
        if 'camera' in metadata:
            result["metadata"]["camera"] = metadata['camera']
        
        # Add recognition data if available
        if recognition_data:
            result["metadata"].update(recognition_data)
        
        # Add chip file path if saved
        if 'file_path' in chip_data:
            result["chip_path"] = chip_data['file_path']
        
        return result
    
    def add_known_face(self, name: str, image_path: Union[str, Path]) -> bool:
        """
        Add a known face to the recognition database
        
        Args:
            name: Identity name
            image_path: Path to reference image
            
        Returns:
            True if successful
        """
        if not self.face_recognizer:
            logger.error("Face recognizer not available")
            return False
        
        return self.face_recognizer.add_known_face(name, image_path)
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face from the recognition database
        
        Args:
            name: Identity name to remove
            
        Returns:
            True if successful
        """
        if not self.face_recognizer:
            logger.error("Face recognizer not available")
            return False
        
        return self.face_recognizer.remove_face(name)
    
    def list_known_faces(self) -> List[Dict[str, Any]]:
        """
        Get list of all known faces
        
        Returns:
            List of face information
        """
        if not self.face_recognizer:
            return []
        
        return self.face_recognizer.list_known_faces()
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """
        Get face recognition statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.face_recognizer:
            return {"error": "Recognition not available"}
        
        return self.face_recognizer.get_statistics()
    
    def save_recognition_database(self) -> bool:
        """
        Save the recognition database to disk
        
        Returns:
            True if successful
        """
        if not self.face_recognizer:
            return False
        
        return self.face_recognizer.save_database()
    
    def export_recognition_database(self, output_path: Union[str, Path]) -> bool:
        """
        Export recognition database metadata to JSON
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            True if successful
        """
        if not self.face_recognizer:
            return False
        
        return self.face_recognizer.export_database_json(output_path)
    
    def batch_add_faces(self, faces_dict: Dict[str, List[Union[str, Path]]]) -> Dict[str, bool]:
        """
        Add multiple known faces in batch
        
        Args:
            faces_dict: {name: [image_paths]}
            
        Returns:
            Results dictionary
        """
        if not self.face_recognizer:
            return {}
        
        return self.face_recognizer.add_multiple_faces(faces_dict)
    
    def save_results_to_json(self, results: List[Dict[str, Any]], 
                           output_path: Union[str, Path]) -> None:
        """Save processing results to JSON file"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {len(results)} face results to {output_path}")