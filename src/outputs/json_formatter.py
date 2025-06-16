import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


class JSONFormatter:
    """Format face detection results into the required JSON schema"""

    def __init__(self):
        self.schema_version = "2.0"  # Updated for clustering support
        self.logger = get_facial_vision_logger(__name__)

    def format_face_result(
        self,
        face_chip: Union[str, bytes],
        confidence: float,
        bbox: Dict[str, int],
        timestamp: str = None,
        gps: Optional[Dict[str, float]] = None,
        source_file: str = None,
        parent_id: str = None,
        identity: str = "unknown",
        camera_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            "face_bounds": bbox,
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
            "topics": ["face", "biometric", "person"],
        }

        return result

    def batch_to_json(self, faces: List[Dict[str, Any]], pretty_print: bool = True) -> str:
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
        required_fields = ["file", "type", "name", "author", "parentId", "metadata", "topics"]

        required_metadata = ["timestamp", "confidence", "identity", "source_file", "face_bounds"]

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
        if not isinstance(
            face_json["confidence"] if "confidence" in face_json else metadata["confidence"],
            (int, float),
        ):
            logger.error("Confidence must be a number")
            return False

        return True

    def save_to_file(
        self, faces: List[Dict[str, Any]], output_path: Union[str, Path], pretty_print: bool = True
    ) -> None:
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

        with open(output_path, "w", encoding="utf-8") as f:
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
                "status": "no_faces_detected",
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
                "avg": round(sum(confidences) / len(confidences), 3),
            },
            "identities": {
                "unique_count": len(set(identities)),
                "known_faces": len([i for i in identities if i != "unknown"]),
                "unknown_faces": len([i for i in identities if i == "unknown"]),
            },
            "source_files": list(set(face["metadata"]["source_file"] for face in faces)),
            "gps_enabled_faces": len([f for f in faces if "gps" in f["metadata"]]),
            "schema_version": self.schema_version,
        }

        return summary

    def format_processing_result(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format complete processing result for blockchain submission

        Args:
            processing_result: Result from image or video processor

        Returns:
            Formatted result ready for blockchain submission
        """
        # Extract key information
        file_path = processing_result.get("file", "")
        file_type = processing_result.get("type", "unknown")
        metadata = processing_result.get("metadata", {})
        chips = metadata.get("chips", [])

        # Create formatted result
        formatted = {
            "file": file_path,
            "type": file_type,
            "name": processing_result.get("name", Path(file_path).stem),
            "author": processing_result.get("author", "facial-vision-system"),
            "timestamp": processing_result.get("timestamp", datetime.now().isoformat()),
            "parentId": processing_result.get("parentId"),
            "metadata": {
                "processing_stats": metadata.get("processing_stats", {}),
                "chips": self._format_chips_for_blockchain(chips),
            },
            "topics": processing_result.get("topics", ["face_detected"]),
        }

        # Add GPS if available
        if "GPS" in metadata:
            formatted["metadata"]["GPS"] = metadata["GPS"]

        # Add device info if available
        if "device" in metadata:
            formatted["metadata"]["device"] = metadata["device"]

        # Add video-specific metadata
        if file_type == "video" and "video_info" in metadata:
            formatted["metadata"]["video_info"] = metadata["video_info"]

        return formatted

    def _format_chips_for_blockchain(self, chips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format chip metadata for blockchain submission"""
        formatted_chips = []
        
        for chip in chips:
            formatted_chip = {
                "file": chip.get("file", ""),
                "type": "image",
                "name": chip.get("name", ""),
                "author": chip.get("author", "facial-vision-system"),
                "timestamp": chip.get("timestamp", datetime.now().isoformat()),
                "clusterId": chip.get("clusterId", "unknown"),
                "face_bounds": chip.get("face_bounds", {}),
                "confidence": chip.get("confidence", 0.0),
                "topics": chip.get("topics", ["face_detected"]),
            }

            # Add optional fields
            optional_fields = ["deviceId", "frameNumber", "videoTimestamp", "gps", "device"]
            for field in optional_fields:
                if field in chip:
                    formatted_chip[field] = chip[field]

            formatted_chips.append(formatted_chip)

        return formatted_chips

    def format_batch_result(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format batch processing results for comprehensive output

        Args:
            batch_result: Result from batch processor

        Returns:
            Formatted batch result
        """
        # Extract statistics
        stats = batch_result.get("processing_stats", {})
        
        formatted = {
            "batch_id": batch_result.get("batch_id"),
            "timestamp": batch_result.get("timestamp"),
            "schema_version": self.schema_version,
            "processing_summary": {
                "total_files": batch_result.get("input_files", 0),
                "successful_files": stats.get("total_files_processed", 0),
                "failed_files": stats.get("failed_files", 0),
                "images_processed": stats.get("images_processed", 0),
                "videos_processed": stats.get("videos_processed", 0),
                "total_faces_detected": stats.get("total_faces_detected", 0),
                "unique_clusters": stats.get("unique_clusters", 0),
                "clustering_enabled": stats.get("clustering_enabled", False),
            },
            "output_directory": batch_result.get("output_directory"),
            "cluster_summary": batch_result.get("cluster_summary", []),
        }

        # Add detailed results if requested
        results = batch_result.get("results", {})
        if results:
            formatted["detailed_results"] = {
                "images": [self.format_processing_result(img) for img in results.get("images", [])],
                "videos": [self.format_processing_result(vid) for vid in results.get("videos", [])],
                "errors": results.get("errors", []),
            }

        return formatted

    def create_blockchain_asset_json(
        self,
        processing_result: Dict[str, Any],
        include_chips_in_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Create JSON structure specifically for blockchain asset creation

        Args:
            processing_result: Result from processor
            include_chips_in_metadata: Whether to include all chips in metadata

        Returns:
            JSON structure ready for blockchain submission
        """
        formatted = self.format_processing_result(processing_result)
        
        # Create blockchain-specific structure
        blockchain_json = {
            "file": formatted["file"],
            "type": formatted["type"], 
            "name": formatted["name"],
            "author": formatted["author"],
            "parentId": formatted.get("parentId"),
            "metadata": {},
            "topics": formatted["topics"],
        }

        # Add metadata
        if include_chips_in_metadata:
            blockchain_json["metadata"] = formatted["metadata"]
        else:
            # Include only summary statistics
            blockchain_json["metadata"] = {
                "processing_stats": formatted["metadata"]["processing_stats"],
                "total_chips": len(formatted["metadata"].get("chips", [])),
                "clusters": list(set(
                    chip.get("clusterId", "unknown") 
                    for chip in formatted["metadata"].get("chips", [])
                )),
            }

        return blockchain_json

    def validate_blockchain_json(self, blockchain_json: Dict[str, Any]) -> bool:
        """
        Validate JSON structure for blockchain submission

        Args:
            blockchain_json: JSON object to validate

        Returns:
            True if valid for blockchain submission
        """
        required_fields = ["file", "type", "name", "author", "metadata", "topics"]
        
        for field in required_fields:
            if field not in blockchain_json:
                logger.error(f"Missing required field for blockchain: {field}")
                return False

        # Validate topics is a list
        if not isinstance(blockchain_json["topics"], list):
            logger.error("Topics must be a list")
            return False

        # Validate metadata is a dict
        if not isinstance(blockchain_json["metadata"], dict):
            logger.error("Metadata must be a dictionary")
            return False

        return True

    def save_blockchain_json(
        self,
        processing_result: Dict[str, Any],
        output_path: Union[str, Path],
        include_chips: bool = True
    ) -> None:
        """
        Save processing result as blockchain-ready JSON

        Args:
            processing_result: Result from processor
            output_path: Path to save JSON file
            include_chips: Whether to include all chip metadata
        """
        blockchain_json = self.create_blockchain_asset_json(
            processing_result, 
            include_chips_in_metadata=include_chips
        )
        
        if not self.validate_blockchain_json(blockchain_json):
            raise ValueError("Generated JSON is not valid for blockchain submission")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(blockchain_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved blockchain-ready JSON to {output_path}")

    def create_cluster_summary(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary of clusters across multiple processing results

        Args:
            processing_results: List of processing results

        Returns:
            Cluster summary dictionary
        """
        cluster_data = {}
        total_faces = 0
        
        for result in processing_results:
            chips = result.get("metadata", {}).get("chips", [])
            total_faces += len(chips)
            
            for chip in chips:
                cluster_id = chip.get("clusterId", "unknown")
                
                if cluster_id not in cluster_data:
                    cluster_data[cluster_id] = {
                        "cluster_id": cluster_id,
                        "chip_count": 0,
                        "confidence_scores": [],
                        "source_files": set(),
                        "first_seen": None,
                        "last_seen": None,
                    }
                
                cluster_info = cluster_data[cluster_id]
                cluster_info["chip_count"] += 1
                cluster_info["confidence_scores"].append(chip.get("confidence", 0.0))
                cluster_info["source_files"].add(chip.get("sourceFile", "unknown"))
                
                timestamp = chip.get("timestamp")
                if timestamp:
                    if not cluster_info["first_seen"] or timestamp < cluster_info["first_seen"]:
                        cluster_info["first_seen"] = timestamp
                    if not cluster_info["last_seen"] or timestamp > cluster_info["last_seen"]:
                        cluster_info["last_seen"] = timestamp

        # Calculate summary statistics
        for cluster_id, data in cluster_data.items():
            data["source_files"] = list(data["source_files"])
            if data["confidence_scores"]:
                data["avg_confidence"] = sum(data["confidence_scores"]) / len(data["confidence_scores"])
                data["max_confidence"] = max(data["confidence_scores"])
                data["min_confidence"] = min(data["confidence_scores"])
            del data["confidence_scores"]  # Remove raw scores to save space

        summary = {
            "total_clusters": len(cluster_data),
            "total_faces": total_faces,
            "generated_at": datetime.now().isoformat(),
            "clusters": list(cluster_data.values()),
            "schema_version": self.schema_version,
        }

        return summary
