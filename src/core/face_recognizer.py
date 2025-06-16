import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition module not available. Recognition features will be limited.")

from src.utils.config import get_config
from src.utils.logger import get_logger, timing_decorator

logger = get_logger(__name__)


class FaceRecognizer:
    """
    Face recognition system that manages known faces and performs identity matching
    """

    def __init__(self, database_path: Optional[Union[str, Path]] = None):
        """
        Initialize face recognizer

        Args:
            database_path: Path to face database file (default: data/face_db/faces.pkl)
        """
        config = get_config()
        recognition_config = config.get("face_recognition", {})

        if database_path is None:
            database_path = recognition_config.get("database_path", "data/face_db/faces.pkl")

        self.database_path = Path(database_path)
        self.database_dir = self.database_path.parent
        self.database_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.similarity_threshold = recognition_config.get("similarity_threshold", 0.6)
        self.max_matches = recognition_config.get("max_matches", 5)
        self.encoding_jitter = recognition_config.get("encoding_jitter", 1)
        self.min_faces_for_identity = recognition_config.get("min_faces_for_identity", 1)

        # Database storage
        self.known_faces = {}  # {name: [encodings]}
        self.metadata = {}  # {name: {date_added, last_seen, face_count, image_paths}}

        # Performance cache
        self._encoding_cache = {}
        self._stats = {"total_recognitions": 0, "successful_matches": 0, "unknown_faces": 0}

        # Load existing database
        self.load_database()

        logger.info(
            f"FaceRecognizer initialized: {len(self.known_faces)} known faces, "
            f"threshold={self.similarity_threshold}"
        )

    @timing_decorator
    def add_known_face(
        self, name: str, image_path_or_encoding: Union[str, Path, np.ndarray]
    ) -> bool:
        """
        Add a new known face to the database

        Args:
            name: Identity name
            image_path_or_encoding: Path to image file or pre-computed encoding

        Returns:
            True if successfully added
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("face_recognition library not available")
            return False

        try:
            # Get face encoding
            if isinstance(image_path_or_encoding, (str, Path)):
                image_path = Path(image_path_or_encoding)
                if not image_path.exists():
                    logger.error(f"Image not found: {image_path}")
                    return False

                # Load image and extract encoding
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image, num_jitters=self.encoding_jitter)

                if len(encodings) == 0:
                    logger.warning(f"No face found in image: {image_path}")
                    return False

                if len(encodings) > 1:
                    logger.warning(f"Multiple faces found in {image_path}, using the first one")

                encoding = encodings[0]
                source_path = str(image_path)

            else:
                # Pre-computed encoding
                encoding = np.array(image_path_or_encoding)
                source_path = "encoding_provided"

            # Validate encoding
            if not self._validate_encoding(encoding):
                logger.error(f"Invalid face encoding for {name}")
                return False

            # Add to database
            if name not in self.known_faces:
                self.known_faces[name] = []
                self.metadata[name] = {
                    "date_added": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "face_count": 0,
                    "image_paths": [],
                }

            self.known_faces[name].append(encoding)
            self.metadata[name]["face_count"] += 1
            self.metadata[name]["last_seen"] = datetime.now().isoformat()

            if source_path != "encoding_provided":
                self.metadata[name]["image_paths"].append(source_path)

            logger.info(
                f"Added face for '{name}': {self.metadata[name]['face_count']} total encodings"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding face for {name}: {e}")
            return False

    def add_multiple_faces(self, faces_dict: Dict[str, List[Union[str, Path]]]) -> Dict[str, bool]:
        """
        Batch add multiple faces

        Args:
            faces_dict: {name: [image_paths]}

        Returns:
            Dictionary of results {name: success_boolean}
        """
        results = {}

        for name, image_paths in faces_dict.items():
            results[name] = True

            for image_path in image_paths:
                success = self.add_known_face(name, image_path)
                if not success:
                    results[name] = False
                    logger.warning(f"Failed to add some images for {name}")

        logger.info(
            f"Batch add complete: {sum(results.values())}/{len(results)} identities successful"
        )
        return results

    @timing_decorator
    def recognize_face(
        self, face_encoding: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Recognize a face encoding against the database

        Args:
            face_encoding: Face encoding to match
            threshold: Override default similarity threshold

        Returns:
            Dictionary with recognition results
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return {"name": "unknown", "confidence": 0.0, "error": "face_recognition not available"}

        threshold = threshold or self.similarity_threshold

        try:
            self._stats["total_recognitions"] += 1

            if len(self.known_faces) == 0:
                self._stats["unknown_faces"] += 1
                return {"name": "unknown", "confidence": 0.0, "reason": "no_known_faces"}

            # Validate input encoding
            if not self._validate_encoding(face_encoding):
                return {"name": "unknown", "confidence": 0.0, "error": "invalid_encoding"}

            # Find best matches
            matches = []

            for name, encodings in self.known_faces.items():
                # Compare against all encodings for this person
                for encoding in encodings:
                    distance = face_recognition.face_distance([encoding], face_encoding)[0]
                    confidence = max(0.0, 1.0 - distance)

                    if distance <= threshold:
                        matches.append(
                            {"name": name, "distance": distance, "confidence": confidence}
                        )

            if len(matches) == 0:
                self._stats["unknown_faces"] += 1
                return {
                    "name": "unknown",
                    "confidence": 0.0,
                    "reason": "no_matches_above_threshold",
                }

            # Use voting if multiple encodings per person
            name_votes = {}
            for match in matches:
                name = match["name"]
                if name not in name_votes:
                    name_votes[name] = {"votes": 0, "total_confidence": 0.0, "distances": []}

                name_votes[name]["votes"] += 1
                name_votes[name]["total_confidence"] += match["confidence"]
                name_votes[name]["distances"].append(match["distance"])

            # Calculate final scores
            final_candidates = []
            for name, votes in name_votes.items():
                avg_confidence = votes["total_confidence"] / votes["votes"]
                avg_distance = np.mean(votes["distances"])

                final_candidates.append(
                    {
                        "name": name,
                        "confidence": avg_confidence,
                        "distance": avg_distance,
                        "votes": votes["votes"],
                        "total_encodings": len(self.known_faces[name]),
                    }
                )

            # Sort by confidence
            final_candidates.sort(key=lambda x: x["confidence"], reverse=True)

            # Get best match
            best_match = final_candidates[0]

            # Update metadata
            self.metadata[best_match["name"]]["last_seen"] = datetime.now().isoformat()
            self._stats["successful_matches"] += 1

            result = {
                "name": best_match["name"],
                "confidence": best_match["confidence"],
                "distance": best_match["distance"],
                "votes": best_match["votes"],
                "total_encodings": best_match["total_encodings"],
                "all_candidates": final_candidates[: self.max_matches],
            }

            logger.debug(
                f"Recognized face: {best_match['name']} "
                f"(confidence: {best_match['confidence']:.2%})"
            )

            return result

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return {"name": "unknown", "confidence": 0.0, "error": str(e)}

    def recognize_faces_in_image(
        self, image_path: Union[str, Path], face_locations: Optional[List[Tuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recognize all faces in an image

        Args:
            image_path: Path to image
            face_locations: Optional pre-detected face locations

        Returns:
            List of recognition results
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return []

            # Load image
            image = face_recognition.load_image_file(str(image_path))

            # Get face locations if not provided
            if face_locations is None:
                face_locations = face_recognition.face_locations(image)

            if len(face_locations) == 0:
                logger.debug(f"No faces found in {image_path}")
                return []

            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Recognize each face
            results = []
            for i, encoding in enumerate(face_encodings):
                recognition = self.recognize_face(encoding)
                recognition["location"] = face_locations[i]
                recognition["face_index"] = i
                results.append(recognition)

            return results

        except Exception as e:
            logger.error(f"Error recognizing faces in image {image_path}: {e}")
            return []

    def update_face(self, name: str, new_encoding: np.ndarray) -> bool:
        """
        Update existing identity with new encoding

        Args:
            name: Identity name
            new_encoding: New face encoding

        Returns:
            True if successful
        """
        if name not in self.known_faces:
            logger.warning(f"Identity '{name}' not found in database")
            return False

        if not self._validate_encoding(new_encoding):
            logger.error(f"Invalid encoding for update of {name}")
            return False

        self.known_faces[name].append(new_encoding)
        self.metadata[name]["face_count"] += 1
        self.metadata[name]["last_seen"] = datetime.now().isoformat()

        logger.info(f"Updated '{name}': now has {self.metadata[name]['face_count']} encodings")
        return True

    def remove_face(self, name: str) -> bool:
        """
        Remove identity from database

        Args:
            name: Identity name to remove

        Returns:
            True if successful
        """
        if name not in self.known_faces:
            logger.warning(f"Identity '{name}' not found in database")
            return False

        del self.known_faces[name]
        del self.metadata[name]

        logger.info(f"Removed identity '{name}' from database")
        return True

    def list_known_faces(self) -> List[Dict[str, Any]]:
        """
        Get list of all known identities with metadata

        Returns:
            List of identity information
        """
        faces = []

        for name in self.known_faces.keys():
            face_info = {
                "name": name,
                "encoding_count": len(self.known_faces[name]),
                **self.metadata[name],
            }
            faces.append(face_info)

        # Sort by last seen (most recent first)
        faces.sort(key=lambda x: x["last_seen"], reverse=True)
        return faces

    @timing_decorator
    def save_database(self) -> bool:
        """
        Save face database to disk

        Returns:
            True if successful
        """
        try:
            # Save as pickle for fast loading
            database = {
                "known_faces": self.known_faces,
                "metadata": self.metadata,
                "stats": self._stats,
                "config": {
                    "similarity_threshold": self.similarity_threshold,
                    "encoding_jitter": self.encoding_jitter,
                },
                "version": "1.0",
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.database_path, "wb") as f:
                pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(
                f"Saved face database: {len(self.known_faces)} identities to {self.database_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving database: {e}")
            return False

    def load_database(self) -> bool:
        """
        Load face database from disk

        Returns:
            True if successful
        """
        if not self.database_path.exists():
            logger.info("No existing face database found, starting fresh")
            return True

        try:
            with open(self.database_path, "rb") as f:
                database = pickle.load(f)

            self.known_faces = database.get("known_faces", {})
            self.metadata = database.get("metadata", {})
            self._stats = database.get("stats", self._stats)

            # Validate loaded data
            invalid_names = []
            for name, encodings in self.known_faces.items():
                for i, encoding in enumerate(encodings):
                    if not self._validate_encoding(encoding):
                        logger.warning(f"Invalid encoding found for {name}, removing")
                        invalid_names.append(name)
                        break

            # Remove invalid entries
            for name in invalid_names:
                if name in self.known_faces:
                    del self.known_faces[name]
                if name in self.metadata:
                    del self.metadata[name]

            logger.info(
                f"Loaded face database: {len(self.known_faces)} identities from {self.database_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading database: {e}")
            return False

    def export_database_json(self, output_path: Union[str, Path]) -> bool:
        """
        Export database to JSON format (without encodings)

        Args:
            output_path: Path to save JSON file

        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)

            export_data = {
                "identities": [],
                "stats": self._stats,
                "config": {
                    "similarity_threshold": self.similarity_threshold,
                    "encoding_jitter": self.encoding_jitter,
                },
                "exported_at": datetime.now().isoformat(),
                "total_identities": len(self.known_faces),
            }

            for name in self.known_faces.keys():
                identity_data = {
                    "name": name,
                    "encoding_count": len(self.known_faces[name]),
                    **self.metadata[name],
                }
                export_data["identities"].append(identity_data)

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported database metadata to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recognition statistics

        Returns:
            Statistics dictionary
        """
        stats = self._stats.copy()
        stats.update(
            {
                "total_identities": len(self.known_faces),
                "total_encodings": sum(len(encodings) for encodings in self.known_faces.values()),
                "avg_encodings_per_identity": np.mean(
                    [len(encodings) for encodings in self.known_faces.values()]
                )
                if self.known_faces
                else 0,
                "recognition_accuracy": self._stats["successful_matches"]
                / max(1, self._stats["total_recognitions"]),
                "database_path": str(self.database_path),
                "similarity_threshold": self.similarity_threshold,
            }
        )

        return stats

    def _validate_encoding(self, encoding: np.ndarray) -> bool:
        """
        Validate face encoding format

        Args:
            encoding: Face encoding array

        Returns:
            True if valid
        """
        if not isinstance(encoding, np.ndarray):
            return False

        if encoding.shape != (128,):
            return False

        if np.any(np.isnan(encoding)) or np.any(np.isinf(encoding)):
            return False

        return True

    def __len__(self) -> int:
        """Return number of known identities"""
        return len(self.known_faces)

    def __contains__(self, name: str) -> bool:
        """Check if identity exists in database"""
        return name in self.known_faces

    def __str__(self) -> str:
        return f"FaceRecognizer({len(self.known_faces)} identities, threshold={self.similarity_threshold})"

    def __repr__(self) -> str:
        return self.__str__()
