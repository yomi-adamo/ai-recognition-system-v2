"""
Tests for file handling utilities
"""
import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.utils.file_handler import FileHandler


class TestFileHandler:
    """Test FileHandler functionality"""

    def test_initialization(self, temp_dir):
        """Test FileHandler initialization"""
        handler = FileHandler(base_output_dir=temp_dir)

        assert handler.base_output_dir == temp_dir
        assert handler.input_dir.exists()
        assert handler.temp_dir.exists()
        assert isinstance(handler.processed_files, set)
        assert isinstance(handler.file_hashes, dict)

    def test_create_timestamped_output_dir(self, temp_dir):
        """Test creating timestamped output directory"""
        handler = FileHandler(base_output_dir=temp_dir)

        output_dir = handler.create_timestamped_output_dir("test")

        assert output_dir.exists()
        assert output_dir.parent == temp_dir
        assert output_dir.name.startswith("test_")
        assert len(output_dir.name) > 10  # Has timestamp

    def test_save_json_output(self, temp_dir):
        """Test saving JSON data"""
        handler = FileHandler(base_output_dir=temp_dir)

        test_data = {"key": "value", "number": 42}
        output_path = handler.save_json_output(test_data, temp_dir / "test.json", pretty_print=True)

        assert output_path.exists()

        # Verify content
        with open(output_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

    def test_save_face_chips(self, temp_dir):
        """Test saving face chips"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create mock chip data
        import numpy as np

        chips = [
            {"chip_array": np.ones((100, 100, 3), dtype=np.uint8) * 255, "name": "face_001"},
            {"chip_array": np.ones((100, 100, 3), dtype=np.uint8) * 128, "name": "face_002"},
        ]

        saved_paths = handler.save_face_chips(chips, temp_dir / "chips")

        assert len(saved_paths) == 2
        assert all(p.exists() for p in saved_paths)
        assert saved_paths[0].name == "face_001.jpg"
        assert saved_paths[1].name == "face_002.jpg"

    def test_get_input_files(self, temp_dir):
        """Test getting input files"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create test files in input directory
        input_dir = handler.input_dir
        (input_dir / "image1.jpg").touch()
        (input_dir / "image2.png").touch()
        (input_dir / "video.mp4").touch()
        (input_dir / "document.txt").touch()

        # Test default extensions (images)
        image_files = handler.get_input_files()
        assert len(image_files) == 2

        # Test specific extensions
        video_files = handler.get_input_files(extensions=[".mp4"])
        assert len(video_files) == 1
        assert video_files[0].name == "video.mp4"

    def test_mark_as_processed(self, temp_dir):
        """Test marking files as processed"""
        handler = FileHandler(base_output_dir=temp_dir)

        test_file = temp_dir / "test.jpg"
        test_file.touch()

        assert not handler.is_processed(test_file)

        handler.mark_as_processed(test_file)

        assert handler.is_processed(test_file)

    def test_resolve_naming_conflict(self, temp_dir):
        """Test resolving file naming conflicts"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create existing file
        existing = temp_dir / "test.txt"
        existing.touch()

        # Test conflict resolution
        resolved = handler._resolve_naming_conflict(existing)
        assert resolved != existing
        assert resolved.name == "test_001.txt"

    def test_create_cluster_directories(self, temp_dir):
        """Test creating cluster directories"""
        handler = FileHandler(base_output_dir=temp_dir)

        cluster_ids = ["person_1", "person_2", "person_3"]
        cluster_dirs = handler.create_cluster_directories(cluster_ids, temp_dir)

        assert len(cluster_dirs) == 3
        assert all(d.exists() for d in cluster_dirs.values())
        assert cluster_dirs["person_1"] == temp_dir / "person_1"

    def test_save_clustered_face_chips(self, temp_dir):
        """Test saving clustered face chips"""
        handler = FileHandler(base_output_dir=temp_dir)

        import numpy as np

        chips_by_cluster = {
            "person_1": [
                {"chip_array": np.ones((100, 100, 3), dtype=np.uint8) * 255, "name": "chip_001"},
                {"chip_array": np.ones((100, 100, 3), dtype=np.uint8) * 200, "name": "chip_002"},
            ],
            "person_2": [
                {"chip_array": np.ones((100, 100, 3), dtype=np.uint8) * 150, "name": "chip_003"}
            ],
        }

        saved_paths = handler.save_clustered_face_chips(chips_by_cluster, temp_dir)

        assert len(saved_paths) == 2
        assert len(saved_paths["person_1"]) == 2
        assert len(saved_paths["person_2"]) == 1

        # Check directory structure
        assert (temp_dir / "person_1").exists()
        assert (temp_dir / "person_2").exists()

    def test_calculate_file_hash(self, temp_dir):
        """Test file hash calculation"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = handler.calculate_file_hash(test_file)
        hash2 = handler.calculate_file_hash(test_file)

        # Should return same hash and use cache
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 char hex string

    def test_get_file_metadata(self, temp_dir):
        """Test getting file metadata"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create test file
        test_file = temp_dir / "test_file.jpg"
        test_file.write_text("Test content")

        metadata = handler.get_file_metadata(test_file)

        assert metadata["name"] == "test_file.jpg"
        assert metadata["extension"] == ".jpg"
        assert metadata["size"] > 0
        assert "hash" in metadata
        assert "created" in metadata
        assert "modified" in metadata

    def test_organize_files_by_type(self, temp_dir):
        """Test organizing files by type"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create test files
        test_dir = temp_dir / "mixed"
        test_dir.mkdir()

        (test_dir / "image1.jpg").touch()
        (test_dir / "image2.png").touch()
        (test_dir / "video1.mp4").touch()
        (test_dir / "document.pdf").touch()

        organized = handler.organize_files_by_type(test_dir)

        assert len(organized["images"]) == 2
        assert len(organized["videos"]) == 1
        assert len(organized["other"]) == 1

    def test_clean_temp_files(self, temp_dir):
        """Test cleaning temporary files"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create old and new temp files
        old_file = handler.temp_dir / "old_temp.tmp"
        new_file = handler.temp_dir / "new_temp.tmp"

        old_file.touch()
        new_file.touch()

        # Make old file appear old
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_file, (old_time, old_time))

        removed = handler.clean_temp_files(max_age_hours=24)

        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_get_disk_usage(self, temp_dir):
        """Test disk usage calculation"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create some test files
        for i in range(3):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text("x" * 1000)  # 1KB each

        usage = handler.get_disk_usage(temp_dir)

        assert usage["file_count"] == 3
        assert usage["total_size"] >= 3000
        assert "total_size_mb" in usage

    def test_batch_process_files(self, temp_dir):
        """Test batch file processing"""
        handler = FileHandler(base_output_dir=temp_dir)

        # Create test files
        files = []
        for i in range(5):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(file_path)

        # Simple processing function
        def process_func(path):
            return path.read_text()

        results = handler.batch_process_files(files, process_func, max_workers=2)

        assert len(results) == 5
        assert all(result[1] is not None for result in results)
        assert results[0][1] == "Content 0"


import os
