"""
Tests for face clustering functionality
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.core.face_clusterer import (
    ClusterInfo,
    ClusterManager,
    ClusterRegistry,
    FaceClusterer,
    FaceEmbeddingExtractor,
)
from src.core.face_detector import FaceDetection


class TestFaceEmbeddingExtractor:
    """Test FaceEmbeddingExtractor functionality"""

    def test_initialization_face_recognition(self):
        """Test initialization with face_recognition method"""
        with patch("src.core.face_clusterer.face_recognition") as mock_fr:
            extractor = FaceEmbeddingExtractor(method="face_recognition")
            assert extractor.method == "face_recognition"
            assert extractor.face_recognition is mock_fr

    def test_initialization_invalid_method(self):
        """Test initialization with invalid method"""
        with pytest.raises(ValueError, match="Unknown embedding method"):
            FaceEmbeddingExtractor(method="invalid_method")

    def test_extract_embeddings_face_recognition(self):
        """Test embedding extraction with face_recognition"""
        # Create mock face_recognition
        mock_fr = Mock()
        mock_fr.face_encodings.return_value = [np.random.rand(128), np.random.rand(128)]

        with patch("src.core.face_clusterer.face_recognition", mock_fr):
            extractor = FaceEmbeddingExtractor(method="face_recognition")

            # Create test data
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = [
                FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9),
                FaceDetection(bbox=(300, 400, 400, 300), confidence=0.8),
            ]

            embeddings = extractor.extract_embeddings(image, detections)

            assert len(embeddings) == 2
            assert all(emb.shape == (128,) for emb in embeddings)
            mock_fr.face_encodings.assert_called_once()

    def test_extract_embeddings_empty_detections(self):
        """Test embedding extraction with no detections"""
        with patch("src.core.face_clusterer.face_recognition"):
            extractor = FaceEmbeddingExtractor(method="face_recognition")

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            embeddings = extractor.extract_embeddings(image, [])

            assert embeddings == []


class TestClusterManager:
    """Test ClusterManager functionality"""

    def test_initialization_hdbscan(self):
        """Test initialization with HDBSCAN"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan") as mock_hdbscan:
                manager = ClusterManager(algorithm="hdbscan")
                assert manager.algorithm == "hdbscan"
                mock_hdbscan.HDBSCAN.assert_called_once()

    def test_initialization_dbscan(self):
        """Test initialization with DBSCAN"""
        with patch("src.core.face_clusterer.SKLEARN_AVAILABLE", True):
            with patch("src.core.face_clusterer.DBSCAN") as mock_dbscan:
                manager = ClusterManager(algorithm="dbscan")
                assert manager.algorithm == "dbscan"
                mock_dbscan.assert_called_once()

    def test_initialization_invalid_algorithm(self):
        """Test initialization with invalid algorithm"""
        with pytest.raises(ValueError, match="Unknown clustering algorithm"):
            ClusterManager(algorithm="invalid_algorithm")

    def test_fit_predict_empty_embeddings(self):
        """Test fit_predict with empty embeddings"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan")
                labels = manager.fit_predict([])
                assert len(labels) == 0

    def test_fit_predict_with_embeddings(self):
        """Test fit_predict with embeddings"""
        # Mock clusterer
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])

        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan")
                manager.clusterer = mock_clusterer

                # Create test embeddings
                embeddings = [np.random.rand(128) for _ in range(5)]
                labels = manager.fit_predict(embeddings)

                assert len(labels) == 5
                mock_clusterer.fit_predict.assert_called_once()

    def test_predict_incremental_no_existing_centroids(self):
        """Test incremental prediction with no existing centroids"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan")

                embeddings = [np.random.rand(128)]
                assignments = manager.predict_incremental(embeddings, {})

                assert assignments == ["person_1"]

    def test_predict_incremental_with_existing_centroids(self):
        """Test incremental prediction with existing centroids"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan", similarity_threshold=0.8)

                # Create test data
                embeddings = [np.array([1.0, 0.0, 0.0])]  # Similar to person_1
                existing_centroids = {
                    "person_1": np.array([0.9, 0.1, 0.0]),
                    "person_2": np.array([0.0, 1.0, 0.0]),
                }

                assignments = manager.predict_incremental(embeddings, existing_centroids)

                assert len(assignments) == 1
                # Should assign to existing cluster or create new one
                assert assignments[0] in ["person_1", "person_2", "person_3"]

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan")

                vec1 = np.array([1.0, 0.0, 0.0])
                vec2 = np.array([1.0, 0.0, 0.0])

                similarity = manager._cosine_similarity(vec1, vec2)
                assert abs(similarity - 1.0) < 1e-6

                vec3 = np.array([0.0, 1.0, 0.0])
                similarity = manager._cosine_similarity(vec1, vec3)
                assert abs(similarity - 0.0) < 1e-6

    def test_get_next_cluster_id(self):
        """Test next cluster ID generation"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan"):
                manager = ClusterManager(algorithm="hdbscan")

                # Test with no existing IDs
                next_id = manager._get_next_cluster_id([])
                assert next_id == "person_1"

                # Test with existing IDs
                existing_ids = ["person_1", "person_3", "person_5"]
                next_id = manager._get_next_cluster_id(existing_ids)
                assert next_id == "person_6"


class TestClusterRegistry:
    """Test ClusterRegistry functionality"""

    def test_initialization(self, temp_dir):
        """Test registry initialization"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        assert registry.registry_path == registry_path
        assert isinstance(registry.clusters, dict)
        assert len(registry.clusters) == 0

    def test_add_new_cluster(self, temp_dir):
        """Test adding a new cluster"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        embedding = np.random.rand(128)
        registry.add_or_update_cluster(
            cluster_id="person_1",
            embedding=embedding,
            chip_path="path/to/chip.jpg",
            confidence=0.95,
        )

        assert "person_1" in registry.clusters
        cluster = registry.clusters["person_1"]
        assert cluster.cluster_id == "person_1"
        assert cluster.chip_count == 1
        assert np.allclose(cluster.centroid, embedding)
        assert "path/to/chip.jpg" in cluster.representative_chips

    def test_update_existing_cluster(self, temp_dir):
        """Test updating an existing cluster"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        # Add first embedding
        embedding1 = np.random.rand(128)
        registry.add_or_update_cluster("person_1", embedding1, "chip1.jpg", 0.9)

        # Add second embedding
        embedding2 = np.random.rand(128)
        registry.add_or_update_cluster("person_1", embedding2, "chip2.jpg", 0.8)

        cluster = registry.clusters["person_1"]
        assert cluster.chip_count == 2
        assert len(cluster.representative_chips) == 2
        # Centroid should be average of embeddings
        expected_centroid = (embedding1 + embedding2) / 2
        assert np.allclose(cluster.centroid, expected_centroid)

    def test_get_cluster_centroids(self, temp_dir):
        """Test getting cluster centroids"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        # Add some clusters
        registry.add_or_update_cluster("person_1", np.random.rand(128), "chip1.jpg", 0.9)
        registry.add_or_update_cluster("person_2", np.random.rand(128), "chip2.jpg", 0.8)

        centroids = registry.get_cluster_centroids()

        assert len(centroids) == 2
        assert "person_1" in centroids
        assert "person_2" in centroids
        assert all(isinstance(centroid, np.ndarray) for centroid in centroids.values())

    def test_save_and_load_registry(self, temp_dir):
        """Test saving and loading registry"""
        registry_path = temp_dir / "test_registry.json"

        # Create and populate registry
        registry1 = ClusterRegistry(registry_path)
        embedding = np.random.rand(128)
        registry1.add_or_update_cluster("person_1", embedding, "chip1.jpg", 0.9)
        registry1.save_registry()

        # Load registry in new instance
        registry2 = ClusterRegistry(registry_path)

        assert len(registry2.clusters) == 1
        assert "person_1" in registry2.clusters
        cluster = registry2.clusters["person_1"]
        assert cluster.chip_count == 1
        assert np.allclose(cluster.centroid, embedding)

    def test_merge_clusters(self, temp_dir):
        """Test merging two clusters"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        # Add two clusters
        embedding1 = np.random.rand(128)
        embedding2 = np.random.rand(128)
        registry.add_or_update_cluster("person_1", embedding1, "chip1.jpg", 0.9)
        registry.add_or_update_cluster("person_2", embedding2, "chip2.jpg", 0.8)

        # Merge person_2 into person_1
        success = registry.merge_clusters("person_2", "person_1")

        assert success is True
        assert "person_2" not in registry.clusters
        assert "person_1" in registry.clusters

        cluster = registry.clusters["person_1"]
        assert cluster.chip_count == 2
        assert len(cluster.representative_chips) == 2

    def test_get_statistics(self, temp_dir):
        """Test getting registry statistics"""
        registry_path = temp_dir / "test_registry.json"
        registry = ClusterRegistry(registry_path)

        # Add some clusters with different sizes
        for i in range(3):
            for j in range(i + 1):  # Different cluster sizes
                registry.add_or_update_cluster(
                    f"person_{i+1}", np.random.rand(128), f"chip_{i}_{j}.jpg", 0.9
                )

        stats = registry.get_statistics()

        assert stats["total_clusters"] == 3
        assert stats["total_chips"] == 6  # 1 + 2 + 3
        assert stats["largest_cluster_size"] == 3
        assert len(stats["clusters"]) == 3


class TestClusterInfo:
    """Test ClusterInfo dataclass"""

    def test_to_dict(self):
        """Test converting ClusterInfo to dictionary"""
        centroid = np.random.rand(128)
        cluster = ClusterInfo(
            cluster_id="person_1",
            centroid=centroid,
            chip_count=5,
            last_seen="2024-01-15T10:00:00Z",
            first_seen="2024-01-15T09:00:00Z",
            representative_chips=["chip1.jpg", "chip2.jpg"],
            confidence_scores=[0.9, 0.8],
        )

        data = cluster.to_dict()

        assert data["cluster_id"] == "person_1"
        assert data["chip_count"] == 5
        assert isinstance(data["centroid"], list)
        assert len(data["centroid"]) == 128

    def test_from_dict(self):
        """Test creating ClusterInfo from dictionary"""
        centroid = np.random.rand(128)
        data = {
            "cluster_id": "person_1",
            "centroid": centroid.tolist(),
            "chip_count": 5,
            "last_seen": "2024-01-15T10:00:00Z",
            "first_seen": "2024-01-15T09:00:00Z",
            "representative_chips": ["chip1.jpg", "chip2.jpg"],
            "confidence_scores": [0.9, 0.8],
        }

        cluster = ClusterInfo.from_dict(data)

        assert cluster.cluster_id == "person_1"
        assert cluster.chip_count == 5
        assert np.allclose(cluster.centroid, centroid)


class TestFaceClusterer:
    """Test main FaceClusterer class"""

    def test_initialization(self):
        """Test FaceClusterer initialization"""
        with patch("src.core.face_clusterer.FaceEmbeddingExtractor") as mock_extractor:
            with patch("src.core.face_clusterer.ClusterManager") as mock_manager:
                with patch("src.core.face_clusterer.ClusterRegistry") as mock_registry:
                    clusterer = FaceClusterer(
                        embedding_method="face_recognition", clustering_algorithm="hdbscan"
                    )

                    mock_extractor.assert_called_once_with("face_recognition")
                    mock_manager.assert_called_once_with("hdbscan")
                    mock_registry.assert_called_once()

    def test_process_faces(self):
        """Test processing faces end-to-end"""
        # Create mocks
        mock_extractor = Mock()
        mock_manager = Mock()
        mock_registry = Mock()

        # Set up mock responses
        embeddings = [np.random.rand(128), np.random.rand(128)]
        mock_extractor.extract_embeddings.return_value = embeddings
        mock_manager.predict_incremental.return_value = ["person_1", "person_2"]
        mock_registry.get_cluster_centroids.return_value = {}

        # Create clusterer with mocks
        clusterer = FaceClusterer()
        clusterer.embedding_extractor = mock_extractor
        clusterer.cluster_manager = mock_manager
        clusterer.cluster_registry = mock_registry

        # Create test data
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9),
            FaceDetection(bbox=(300, 400, 400, 300), confidence=0.8),
        ]

        # Process faces
        cluster_ids = clusterer.process_faces(image, detections)

        assert cluster_ids == ["person_1", "person_2"]
        mock_extractor.extract_embeddings.assert_called_once()
        mock_manager.predict_incremental.assert_called_once()
        assert mock_registry.add_or_update_cluster.call_count == 2
        mock_registry.save_registry.assert_called_once()

    def test_process_faces_empty_detections(self):
        """Test processing with no face detections"""
        clusterer = FaceClusterer()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cluster_ids = clusterer.process_faces(image, [])

        assert cluster_ids == []

    def test_batch_cluster(self):
        """Test batch clustering"""
        mock_manager = Mock()
        mock_manager.fit_predict.return_value = np.array([0, 0, 1, 1])

        clusterer = FaceClusterer()
        clusterer.cluster_manager = mock_manager

        embeddings = [np.random.rand(128) for _ in range(4)]
        labels = clusterer.batch_cluster(embeddings)

        assert len(labels) == 4
        mock_manager.fit_predict.assert_called_once_with(embeddings)

    def test_create_from_config(self):
        """Test creating clusterer from configuration"""
        config = {
            "embedding_method": "face_recognition",
            "algorithm": "hdbscan",
            "cluster_registry_path": "test_registry.json",
        }

        with patch("src.core.face_clusterer.FaceEmbeddingExtractor"):
            with patch("src.core.face_clusterer.ClusterManager"):
                with patch("src.core.face_clusterer.ClusterRegistry"):
                    clusterer = FaceClusterer.create_from_config(config)

                    assert clusterer is not None


@pytest.mark.integration
class TestClusteringIntegration:
    """Integration tests for clustering workflow"""

    def test_end_to_end_clustering_workflow(self, temp_dir):
        """Test complete clustering workflow"""
        # This test requires actual dependencies, so we'll mock them
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.SKLEARN_AVAILABLE", True):
                with patch("src.core.face_clusterer.face_recognition") as mock_fr:
                    with patch("src.core.face_clusterer.hdbscan") as mock_hdbscan:
                        # Set up mocks
                        mock_fr.face_encodings.return_value = [
                            np.random.rand(128) for _ in range(3)
                        ]

                        mock_clusterer = Mock()
                        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1])
                        mock_hdbscan.HDBSCAN.return_value = mock_clusterer

                        # Create test data
                        registry_path = temp_dir / "integration_registry.json"

                        clusterer = FaceClusterer(
                            embedding_method="face_recognition",
                            clustering_algorithm="hdbscan",
                            registry_path=registry_path,
                        )

                        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        detections = [
                            FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9),
                            FaceDetection(bbox=(150, 250, 250, 150), confidence=0.8),
                            FaceDetection(bbox=(300, 400, 400, 300), confidence=0.7),
                        ]

                        # Process faces
                        cluster_ids = clusterer.process_faces(image, detections)

                        # Verify results
                        assert len(cluster_ids) == 3
                        assert all(isinstance(cid, str) for cid in cluster_ids)

                        # Check registry was updated
                        stats = clusterer.get_cluster_statistics()
                        assert stats["total_chips"] == 3


class TestClusteringEdgeCases:
    """Test edge cases and error conditions"""

    def test_clustering_with_identical_embeddings(self):
        """Test clustering when all embeddings are identical"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan") as mock_hdbscan:
                mock_clusterer = Mock()
                mock_clusterer.fit_predict.return_value = np.array([0, 0, 0])
                mock_hdbscan.HDBSCAN.return_value = mock_clusterer

                manager = ClusterManager(algorithm="hdbscan")

                # All identical embeddings
                embeddings = [np.ones(128) for _ in range(3)]
                labels = manager.fit_predict(embeddings)

                assert len(labels) == 3

    def test_clustering_with_single_embedding(self):
        """Test clustering with only one embedding"""
        with patch("src.core.face_clusterer.HDBSCAN_AVAILABLE", True):
            with patch("src.core.face_clusterer.hdbscan") as mock_hdbscan:
                mock_clusterer = Mock()
                mock_clusterer.fit_predict.return_value = np.array([-1])  # Noise
                mock_hdbscan.HDBSCAN.return_value = mock_clusterer

                manager = ClusterManager(algorithm="hdbscan")

                embeddings = [np.random.rand(128)]
                labels = manager.fit_predict(embeddings)

                assert len(labels) == 1

    def test_registry_with_corrupted_file(self, temp_dir):
        """Test registry loading with corrupted file"""
        registry_path = temp_dir / "corrupted_registry.json"

        # Create corrupted JSON file
        with open(registry_path, "w") as f:
            f.write("{ invalid json }")

        # Should handle corruption gracefully
        registry = ClusterRegistry(registry_path)
        assert len(registry.clusters) == 0

    def test_embedding_extraction_failure(self):
        """Test handling of embedding extraction failure"""
        mock_fr = Mock()
        mock_fr.face_encodings.side_effect = Exception("Extraction failed")

        with patch("src.core.face_clusterer.face_recognition", mock_fr):
            extractor = FaceEmbeddingExtractor(method="face_recognition")

            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            detections = [FaceDetection(bbox=(100, 200, 200, 100), confidence=0.9)]

            # Should return empty list on failure
            embeddings = extractor.extract_embeddings(image, detections)
            assert embeddings == []
