"""
Face clustering module for grouping similar faces without identity labels.

This module implements unsupervised clustering of face embeddings to group
similar faces into person_1, person_2, etc. without requiring manual labeling.
"""

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Clustering libraries
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.core.face_detector import FaceDetection
from src.utils.config import get_config
from src.utils.file_handler import FileHandler
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


@dataclass
class ClusterInfo:
    """Information about a face cluster"""

    cluster_id: str
    centroid: np.ndarray
    chip_count: int
    last_seen: str
    first_seen: str
    representative_chips: List[str]
    confidence_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types"""
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "chip_count": self.chip_count,
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "representative_chips": self.representative_chips,
            "confidence_scores": self.confidence_scores,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterInfo":
        """Create from dictionary"""
        centroid = np.array(data["centroid"]) if data["centroid"] is not None else None
        return cls(
            cluster_id=data["cluster_id"],
            centroid=centroid,
            chip_count=data["chip_count"],
            last_seen=data["last_seen"],
            first_seen=data["first_seen"],
            representative_chips=data["representative_chips"],
            confidence_scores=data["confidence_scores"],
        )


class FaceEmbeddingExtractor:
    """
    Extract face embeddings for clustering using various methods
    """

    def __init__(self, method: str = "face_recognition"):
        """
        Initialize embedding extractor

        Args:
            method: Embedding method - "face_recognition", "deepface", etc.
        """
        self.method = method.lower()
        self.config = get_config()

        if self.method == "face_recognition":
            try:
                import face_recognition

                self.face_recognition = face_recognition
            except ImportError:
                raise ImportError("face_recognition is required for this method")

        elif self.method == "deepface":
            try:
                from deepface import DeepFace

                self.deepface = DeepFace
                self.model_name = "VGG-Face"  # Default model
            except ImportError:
                raise ImportError("deepface is required for this method")

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        logger.info(f"Initialized FaceEmbeddingExtractor", method=self.method)

    def extract_embeddings(
        self, image: np.ndarray, face_detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """
        Extract embeddings for detected faces

        Args:
            image: Source image as numpy array
            face_detections: List of face detections

        Returns:
            List of face embeddings
        """
        embeddings = []

        if self.method == "face_recognition":
            embeddings = self._extract_face_recognition_embeddings(image, face_detections)
        elif self.method == "deepface":
            embeddings = self._extract_deepface_embeddings(image, face_detections)

        logger.debug(f"Extracted {len(embeddings)} embeddings using {self.method}")
        return embeddings

    def _extract_face_recognition_embeddings(
        self, image: np.ndarray, face_detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract embeddings using face_recognition library"""
        face_locations = [detection.bbox for detection in face_detections]

        try:
            encodings = self.face_recognition.face_encodings(image, face_locations)
            return encodings
        except Exception as e:
            logger.error(f"Error extracting face_recognition embeddings", error=str(e))
            return []

    def _extract_deepface_embeddings(
        self, image: np.ndarray, face_detections: List[FaceDetection]
    ) -> List[np.ndarray]:
        """Extract embeddings using DeepFace"""
        embeddings = []

        for detection in face_detections:
            try:
                # Crop face region
                top, right, bottom, left = detection.bbox
                face_crop = image[top:bottom, left:right]

                # Extract embedding
                embedding = self.deepface.represent(
                    face_crop, model_name=self.model_name, enforce_detection=False
                )

                embeddings.append(np.array(embedding[0]["embedding"]))

            except Exception as e:
                logger.warning(f"Failed to extract DeepFace embedding", error=str(e))
                continue

        return embeddings


class ClusterManager:
    """
    Manage face clustering using HDBSCAN or DBSCAN
    """

    def __init__(self, algorithm: str = "hdbscan", **kwargs):
        """
        Initialize cluster manager

        Args:
            algorithm: Clustering algorithm - "hdbscan", "dbscan", "agglomerative"
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        self.config = get_config()
        clustering_config = self.config.get_clustering_config()

        # Set default parameters
        self.min_cluster_size = kwargs.get(
            "min_cluster_size", clustering_config["min_cluster_size"]
        )
        self.min_samples = kwargs.get("min_samples", clustering_config["min_samples"])
        self.cluster_selection_epsilon = kwargs.get(
            "cluster_selection_epsilon", clustering_config["cluster_selection_epsilon"]
        )
        self.metric = kwargs.get("metric", clustering_config["metric"])
        self.similarity_threshold = kwargs.get(
            "similarity_threshold", clustering_config["similarity_threshold"]
        )

        # Initialize clusterer
        if self.algorithm == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise ImportError("hdbscan is required for this algorithm")
            
            # HDBSCAN uses different metric names
            hdbscan_metric = self.metric
            if self.metric == "cosine":
                hdbscan_metric = "euclidean"  # We'll handle cosine with normalization
                
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric=hdbscan_metric,
            )

        elif self.algorithm == "dbscan":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for this algorithm")
            eps = kwargs.get("eps", 1.0 - self.similarity_threshold)
            self.clusterer = DBSCAN(eps=eps, min_samples=self.min_samples, metric=self.metric)

        elif self.algorithm == "agglomerative":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for this algorithm")
            n_clusters = kwargs.get("n_clusters", None)
            distance_threshold = kwargs.get("distance_threshold", 1.0 - self.similarity_threshold)
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, distance_threshold=distance_threshold, linkage="average"
            )

        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")

        logger.info(
            f"Initialized ClusterManager",
            algorithm=self.algorithm,
            min_cluster_size=self.min_cluster_size,
        )

    def fit_predict(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Fit clustering model and predict cluster labels

        Args:
            embeddings: List of face embeddings

        Returns:
            Array of cluster labels
        """
        if not embeddings:
            return np.array([])

        # Convert to numpy array
        X = np.array(embeddings)

        # For cosine similarity, we normalize embeddings and use euclidean distance
        # This is mathematically equivalent to cosine distance
        if self.metric == "cosine":
            X = self._normalize_embeddings(X)

        # Fit and predict
        labels = self.clusterer.fit_predict(X)

        logger.info(
            f"Clustered {len(embeddings)} embeddings",
            unique_clusters=len(np.unique(labels[labels >= 0])),
            noise_points=np.sum(labels == -1),
        )

        return labels

    def predict_incremental(
        self, new_embeddings: List[np.ndarray], existing_centroids: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Predict cluster assignments for new embeddings using existing centroids

        Args:
            new_embeddings: New face embeddings to assign
            existing_centroids: Dictionary of existing cluster centroids

        Returns:
            List of cluster IDs for new embeddings
        """
        if not new_embeddings:
            return []

        assignments = []

        for embedding in new_embeddings:
            if not existing_centroids:
                # No existing clusters, create new one
                assignments.append("person_1")
                continue

            # Calculate similarities to existing centroids
            similarities = {}
            for cluster_id, centroid in existing_centroids.items():
                if self.metric == "cosine":
                    similarity = self._cosine_similarity(embedding, centroid)
                else:
                    similarity = 1.0 / (1.0 + np.linalg.norm(embedding - centroid))

                similarities[cluster_id] = similarity

            # Find best match
            best_cluster = max(similarities.keys(), key=lambda k: similarities[k])
            best_similarity = similarities[best_cluster]

            if best_similarity >= self.similarity_threshold:
                assignments.append(best_cluster)
            else:
                # Create new cluster
                next_id = self._get_next_cluster_id(list(existing_centroids.keys()))
                assignments.append(next_id)

        return assignments

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _get_next_cluster_id(self, existing_ids: List[str]) -> str:
        """Get next available cluster ID"""
        if not existing_ids:
            return "person_1"

        # Extract numbers from existing IDs
        numbers = []
        for cluster_id in existing_ids:
            if cluster_id.startswith("person_"):
                try:
                    num = int(cluster_id.split("_")[1])
                    numbers.append(num)
                except (ValueError, IndexError):
                    continue

        if not numbers:
            return "person_1"

        return f"person_{max(numbers) + 1}"


class ClusterRegistry:
    """
    Registry for tracking face clusters and their metadata
    """

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """
        Initialize cluster registry

        Args:
            registry_path: Path to registry file
        """
        self.config = get_config()

        if registry_path is None:
            clustering_config = self.config.get_clustering_config()
            registry_path = clustering_config["cluster_registry_path"]

        self.registry_path = Path(registry_path)
        self.clusters: Dict[str, ClusterInfo] = {}
        self.file_handler = FileHandler()

        # Load existing registry
        self.load_registry()

        logger.info(
            f"Initialized ClusterRegistry",
            path=str(self.registry_path),
            existing_clusters=len(self.clusters),
        )

    def add_or_update_cluster(
        self, cluster_id: str, embedding: np.ndarray, chip_path: str, confidence: float
    ) -> None:
        """
        Add a new cluster or update existing one

        Args:
            cluster_id: Cluster identifier
            embedding: Face embedding
            chip_path: Path to face chip
            confidence: Detection confidence
        """
        current_time = datetime.utcnow().isoformat()

        if cluster_id in self.clusters:
            # Update existing cluster
            cluster = self.clusters[cluster_id]

            # Update centroid (running average)
            old_count = cluster.chip_count
            new_count = old_count + 1

            cluster.centroid = (cluster.centroid * old_count + embedding) / new_count
            cluster.chip_count = new_count
            cluster.last_seen = current_time
            cluster.representative_chips.append(chip_path)
            cluster.confidence_scores.append(confidence)

            # Keep only top N representative chips
            max_representatives = 5
            if len(cluster.representative_chips) > max_representatives:
                # Keep chips with highest confidence scores
                paired = list(zip(cluster.confidence_scores, cluster.representative_chips))
                paired.sort(reverse=True, key=lambda x: x[0])

                cluster.confidence_scores = [conf for conf, _ in paired[:max_representatives]]
                cluster.representative_chips = [chip for _, chip in paired[:max_representatives]]

        else:
            # Create new cluster
            self.clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                centroid=embedding.copy(),
                chip_count=1,
                first_seen=current_time,
                last_seen=current_time,
                representative_chips=[chip_path],
                confidence_scores=[confidence],
            )

        logger.debug(
            f"Updated cluster",
            cluster_id=cluster_id,
            chip_count=self.clusters[cluster_id].chip_count,
        )

    def get_cluster_centroids(self) -> Dict[str, np.ndarray]:
        """Get centroids of all clusters"""
        return {
            cluster_id: cluster.centroid
            for cluster_id, cluster in self.clusters.items()
            if cluster.centroid is not None
        }

    def get_cluster_info(self, cluster_id: str) -> Optional[ClusterInfo]:
        """Get information about a specific cluster"""
        return self.clusters.get(cluster_id)

    def get_all_clusters(self) -> Dict[str, ClusterInfo]:
        """Get all cluster information"""
        return self.clusters.copy()

    def remove_cluster(self, cluster_id: str) -> bool:
        """
        Remove a cluster from registry

        Args:
            cluster_id: Cluster to remove

        Returns:
            True if cluster was removed, False if not found
        """
        if cluster_id in self.clusters:
            del self.clusters[cluster_id]
            logger.info(f"Removed cluster", cluster_id=cluster_id)
            return True
        return False

    def merge_clusters(self, source_id: str, target_id: str) -> bool:
        """
        Merge two clusters

        Args:
            source_id: Source cluster to merge from
            target_id: Target cluster to merge into

        Returns:
            True if merge was successful
        """
        if source_id not in self.clusters or target_id not in self.clusters:
            return False

        source = self.clusters[source_id]
        target = self.clusters[target_id]

        # Merge centroids (weighted average)
        total_count = source.chip_count + target.chip_count
        target.centroid = (
            source.centroid * source.chip_count + target.centroid * target.chip_count
        ) / total_count

        # Merge other attributes
        target.chip_count = total_count
        target.last_seen = max(source.last_seen, target.last_seen)
        target.first_seen = min(source.first_seen, target.first_seen)
        target.representative_chips.extend(source.representative_chips)
        target.confidence_scores.extend(source.confidence_scores)

        # Remove source cluster
        del self.clusters[source_id]

        logger.info(f"Merged clusters", source=source_id, target=target_id)
        return True

    def save_registry(self) -> None:
        """Save registry to file"""
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                "clusters": {
                    cluster_id: cluster.to_dict() for cluster_id, cluster in self.clusters.items()
                },
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "total_clusters": len(self.clusters),
                    "total_chips": sum(c.chip_count for c in self.clusters.values()),
                },
            }

            # Save to JSON
            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Saved cluster registry", path=str(self.registry_path), clusters=len(self.clusters)
            )

        except Exception as e:
            logger.error(f"Failed to save cluster registry", error=str(e))

    def load_registry(self) -> None:
        """Load registry from file"""
        if not self.registry_path.exists():
            logger.info(f"No existing registry found, starting fresh")
            return

        try:
            with open(self.registry_path, "r") as f:
                data = json.load(f)

            # Load clusters
            self.clusters = {}
            for cluster_id, cluster_data in data.get("clusters", {}).items():
                self.clusters[cluster_id] = ClusterInfo.from_dict(cluster_data)

            logger.info(
                f"Loaded cluster registry",
                clusters=len(self.clusters),
                path=str(self.registry_path),
            )

        except Exception as e:
            logger.error(f"Failed to load cluster registry", error=str(e))
            self.clusters = {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_clusters": len(self.clusters),
            "total_chips": sum(c.chip_count for c in self.clusters.values()),
            "largest_cluster_size": max(c.chip_count for c in self.clusters.values())
            if self.clusters
            else 0,
            "average_cluster_size": np.mean([c.chip_count for c in self.clusters.values()])
            if self.clusters
            else 0.0,
            "clusters": list(self.clusters.keys()),
        }


class FaceClusterer:
    """
    Main face clustering class that combines embedding extraction and clustering
    """

    def __init__(
        self,
        embedding_method: str = "face_recognition",
        clustering_algorithm: str = "hdbscan",
        registry_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize face clusterer

        Args:
            embedding_method: Method for extracting face embeddings
            clustering_algorithm: Clustering algorithm to use
            registry_path: Path to cluster registry file
        """
        self.embedding_extractor = FaceEmbeddingExtractor(embedding_method)
        self.cluster_manager = ClusterManager(clustering_algorithm)
        self.cluster_registry = ClusterRegistry(registry_path)

        logger.info(
            f"Initialized FaceClusterer",
            embedding_method=embedding_method,
            clustering_algorithm=clustering_algorithm,
        )

    def process_faces(
        self,
        image: np.ndarray,
        face_detections: List[FaceDetection],
        chip_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Process detected faces and assign cluster IDs

        Args:
            image: Source image
            face_detections: List of face detections
            chip_paths: Optional paths to saved chips

        Returns:
            List of cluster IDs for each face
        """
        if not face_detections:
            return []

        # Extract embeddings
        embeddings = self.embedding_extractor.extract_embeddings(image, face_detections)

        if not embeddings:
            logger.warning(f"No embeddings extracted from {len(face_detections)} detections")
            return []

        # Get existing cluster centroids
        existing_centroids = self.cluster_registry.get_cluster_centroids()

        # Assign clusters incrementally
        cluster_ids = self.cluster_manager.predict_incremental(embeddings, existing_centroids)

        # Update registry
        for i, (detection, embedding, cluster_id) in enumerate(
            zip(face_detections, embeddings, cluster_ids)
        ):
            chip_path = chip_paths[i] if chip_paths and i < len(chip_paths) else f"face_{i}.jpg"
            self.cluster_registry.add_or_update_cluster(
                cluster_id=cluster_id,
                embedding=embedding,
                chip_path=chip_path,
                confidence=detection.confidence,
            )

        # Save registry
        self.cluster_registry.save_registry()

        logger.info(f"Processed {len(face_detections)} faces", cluster_assignments=cluster_ids)

        return cluster_ids

    def batch_cluster(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Perform batch clustering on a set of embeddings

        Args:
            embeddings: List of face embeddings

        Returns:
            Array of cluster labels
        """
        return self.cluster_manager.fit_predict(embeddings)

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        return self.cluster_registry.get_statistics()

    @classmethod
    def create_from_config(cls, config: Optional[Dict[str, Any]] = None) -> "FaceClusterer":
        """
        Create FaceClusterer from configuration

        Args:
            config: Optional configuration dictionary

        Returns:
            Configured FaceClusterer instance
        """
        if config is None:
            config_manager = get_config()
            config = config_manager.get_clustering_config()

        embedding_method = config.get("embedding_method", "face_recognition")
        clustering_algorithm = config.get("algorithm", "hdbscan")
        registry_path = config.get("cluster_registry_path", None)

        return cls(
            embedding_method=embedding_method,
            clustering_algorithm=clustering_algorithm,
            registry_path=registry_path,
        )
