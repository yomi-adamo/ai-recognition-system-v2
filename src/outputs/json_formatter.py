import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class JSONFormatter:
    """Format face detection results into the required JSON schema"""
    
    def __init__(self):
        self.schema_version = "1.0"
        self.logger = get_logger(__name__)
        
    def format_face_result(self, 
                          face_chip: Union[str, bytes],
                          confidence: float,
                          bbox: Dict[str, int],
                          timestamp: str = None,
                          gps: Optional[Dict[str, float]] = None,
                          source_file: str = None,
                          parent_id: str = None,
                          identity: str = "unknown",
                          camera_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format a single face detection result into JSON schema
        
        Args:
            face_chip: Base64 encoded image or file path
            confidence: Detection confidence score
            bbox: Bounding box with x, y, w, h
            timestamp: ISO 8601 timestamp
            gps: GPS coordinates (lat, lon, optional alt)
            source_file: Original filename
            parent_id: UUID of parent image/video
            identity: Recognized identity or "unknown"
            camera_info: Camera metadata
            
        Returns:
            Formatted JSON object
        """
        # Generate unique ID and name
        unique_id = str(uuid.uuid4())
        timestamp_str = timestamp or datetime.utcnow().isoformat()
        name = f"face_chip_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{unique_id[:8]}"
        
        # Build metadata
        metadata = {
            "timestamp": timestamp_str,
            "confidence": round(float(confidence), 3),
            "identity": identity,
            "source_file": source_file or "unknown",
            "face_bounds": bbox
        }
        
        # Add optional metadata
        if gps:
            metadata["gps"] = gps
        if camera_info:
            metadata["camera"] = camera_info
            
        # Build main object
        result = {
            "file": face_chip,
            "type": "image",
            "name": name,
            "author": "facial-vision-system",
            "parentId": parent_id or str(uuid.uuid4()),
            "metadata": metadata,
            "topics": ["face", "biometric", "person"]
        }
        
        return result
    
    def batch_to_json(self, faces: List[Dict[str, Any]], 
                     pretty_print: bool = True) -> str:
        """
        Convert list of face results to JSON string
        
        Args:
            faces: List of face detection dictionaries
            pretty_print: Whether to format with indentation
            
        Returns:
            JSON string
        """
        if pretty_print:
            return json.dumps(faces, indent=2, ensure_ascii=False)
        else:
            return json.dumps(faces, ensure_ascii=False)
    
    def validate_face_json(self, face_json: Dict[str, Any]) -> bool:
        """
        Validate that a face JSON object has all required fields
        
        Args:
            face_json: Face detection JSON object
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "file", "type", "name", "author", "parentId", "metadata", "topics"
        ]
        
        required_metadata = [
            "timestamp", "confidence", "identity", "source_file", "face_bounds"
        ]
        
        required_bounds = ["x", "y", "w", "h"]
        
        # Check top-level fields
        for field in required_fields:
            if field not in face_json:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check metadata fields
        metadata = face_json.get("metadata", {})
        for field in required_metadata:
            if field not in metadata:
                logger.error(f"Missing required metadata field: {field}")
                return False
        
        # Check face bounds
        face_bounds = metadata.get("face_bounds", {})
        for field in required_bounds:
            if field not in face_bounds:
                logger.error(f"Missing required face_bounds field: {field}")
                return False
        
        # Validate data types
        if not isinstance(face_json["confidence"] if "confidence" in face_json else metadata["confidence"], (int, float)):
            logger.error("Confidence must be a number")
            return False
        
        return True
    
    def save_to_file(self, faces: List[Dict[str, Any]], 
                    output_path: Union[str, Path],
                    pretty_print: bool = True) -> None:
        """
        Save face results to JSON file
        
        Args:
            faces: List of face detection results
            output_path: Path to output file
            pretty_print: Whether to format with indentation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = self.batch_to_json(faces, pretty_print)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        logger.info(f"Saved {len(faces)} face results to {output_path}")
    
    def create_summary_report(self, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary report from face detection results
        
        Args:
            faces: List of face detection results
            
        Returns:
            Summary dictionary
        """
        if not faces:
            return {
                "total_faces": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "no_faces_detected"
            }
        
        # Calculate statistics
        confidences = [face["metadata"]["confidence"] for face in faces]
        identities = [face["metadata"]["identity"] for face in faces]
        
        summary = {
            "total_faces": len(faces),
            "timestamp": datetime.utcnow().isoformat(),
            "confidence_stats": {
                "min": round(min(confidences), 3),
                "max": round(max(confidences), 3),
                "avg": round(sum(confidences) / len(confidences), 3)
            },
            "identities": {
                "unique_count": len(set(identities)),
                "known_faces": len([i for i in identities if i != "unknown"]),
                "unknown_faces": len([i for i in identities if i == "unknown"])
            },
            "source_files": list(set(face["metadata"]["source_file"] for face in faces)),
            "gps_enabled_faces": len([f for f in faces if "gps" in f["metadata"]]),
            "schema_version": self.schema_version
        }
        
        return summary