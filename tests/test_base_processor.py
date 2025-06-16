"""
Tests for base processor classes
"""
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.processors.base_processor import (
    BaseProcessor,
    BatchProcessorBase,
    ProcessorChain,
    StreamProcessor,
)


class TestProcessor(BaseProcessor):
    """Concrete implementation for testing"""

    def process(self, input_path, **kwargs):
        # Simple implementation for testing
        self.stats["files_processed"] += 1
        return {"result": "processed", "input": str(input_path)}


class TestBatchProcessor(BatchProcessorBase):
    """Concrete batch processor for testing"""

    def process(self, input_path, **kwargs):
        return {"result": "processed", "input": str(input_path)}

    def process_batch(self, input_paths, **kwargs):
        results = []
        for path in input_paths:
            try:
                result = self.process(path)
                result["success"] = True
                results.append(result)
            except Exception as e:
                results.append({"input": str(path), "success": False, "error": str(e)})
        return results


class TestStreamProcessorImpl(StreamProcessor):
    """Concrete stream processor for testing"""

    def process(self, input_path, **kwargs):
        return {"result": "processed"}

    def process_chunk(self, chunk):
        if len(self.buffer) >= self.buffer_size - 1:
            # Process when buffer is full
            data = self.flush_buffer()
            data.append(chunk)
            return {"processed_chunks": len(data)}
        else:
            self.add_to_buffer(chunk)
            return None


class TestBaseProcessor:
    """Test BaseProcessor functionality"""

    def test_initialization(self):
        """Test processor initialization"""
        processor = TestProcessor()

        assert processor.config_manager is not None
        assert processor.file_handler is not None
        assert processor.stats["files_processed"] == 0
        assert processor.stats["errors"] == 0

    def test_initialization_with_config(self):
        """Test processor initialization with custom config"""
        custom_config = {"custom_key": "custom_value"}
        processor = TestProcessor(config=custom_config)

        assert processor.config == custom_config

    def test_preprocess(self, temp_dir):
        """Test preprocessing hook"""
        processor = TestProcessor()

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        metadata = processor.preprocess(test_file)

        assert metadata["name"] == "test.txt"
        assert metadata["size"] > 0
        assert "hash" in metadata

    def test_preprocess_file_not_found(self):
        """Test preprocessing with non-existent file"""
        processor = TestProcessor()

        with pytest.raises(FileNotFoundError):
            processor.preprocess("/non/existent/file.txt")

    def test_postprocess(self):
        """Test postprocessing hook"""
        processor = TestProcessor()

        results = {"data": "test"}
        processed = processor.postprocess(results)

        assert processed["processor"] == "TestProcessor"
        assert "processing_timestamp" in processed
        assert processed["data"] == "test"

    def test_validate_input(self, temp_dir):
        """Test input validation"""
        processor = TestProcessor()

        # Valid file
        valid_file = temp_dir / "valid.txt"
        valid_file.touch()
        assert processor.validate_input(valid_file) is True

        # Invalid file
        assert processor.validate_input("/non/existent/file.txt") is False

    def test_handle_error(self):
        """Test error handling"""
        processor = TestProcessor()

        error = ValueError("Test error")
        context = {"file": "test.txt", "operation": "processing"}

        processor.handle_error(error, context)

        assert processor.stats["errors"] == 1

    def test_get_stats(self):
        """Test getting processing statistics"""
        processor = TestProcessor()

        processor.stats["start_time"] = time.time()
        processor.stats["files_processed"] = 5
        time.sleep(0.01)
        processor.stats["end_time"] = time.time()

        stats = processor.get_stats()

        assert stats["files_processed"] == 5
        assert stats["processing_time"] > 0

    def test_reset_stats(self):
        """Test resetting statistics"""
        processor = TestProcessor()

        processor.stats["files_processed"] = 10
        processor.stats["errors"] = 2

        processor.reset_stats()

        assert processor.stats["files_processed"] == 0
        assert processor.stats["errors"] == 0


class TestBatchProcessorBase:
    """Test BatchProcessorBase functionality"""

    def test_initialization(self):
        """Test batch processor initialization"""
        processor = TestBatchProcessor(max_workers=8)

        assert processor.max_workers == 8
        assert processor.results == []

    def test_filter_inputs(self, temp_dir):
        """Test input filtering"""
        processor = TestBatchProcessor()

        # Create valid files
        valid1 = temp_dir / "valid1.txt"
        valid2 = temp_dir / "valid2.txt"
        valid1.touch()
        valid2.touch()

        # Mix valid and invalid paths
        input_paths = [valid1, valid2, "/invalid/path.txt"]

        filtered = processor.filter_inputs(input_paths)

        assert len(filtered) == 2
        assert all(p.exists() for p in filtered)

    def test_process_batch(self, temp_dir):
        """Test batch processing"""
        processor = TestBatchProcessor()

        # Create test files
        files = []
        for i in range(3):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.touch()
            files.append(file_path)

        results = processor.process_batch(files)

        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert all("result" in r for r in results)

    def test_aggregate_results(self):
        """Test result aggregation"""
        processor = TestBatchProcessor()

        results = [
            {"success": True, "data": 1},
            {"success": True, "data": 2},
            {"success": False, "error": "Failed"},
        ]

        summary = processor.aggregate_results(results)

        assert summary["total_processed"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["results"] == results


class TestStreamProcessor:
    """Test StreamProcessor functionality"""

    def test_initialization(self):
        """Test stream processor initialization"""
        processor = TestStreamProcessorImpl(buffer_size=10)

        assert processor.buffer_size == 10
        assert processor.buffer == []

    def test_add_to_buffer(self):
        """Test adding data to buffer"""
        processor = TestStreamProcessorImpl(buffer_size=3)

        assert processor.add_to_buffer("chunk1") is False
        assert processor.add_to_buffer("chunk2") is False
        assert processor.add_to_buffer("chunk3") is True  # Buffer full

        assert len(processor.buffer) == 3

    def test_flush_buffer(self):
        """Test flushing buffer"""
        processor = TestStreamProcessorImpl(buffer_size=3)

        processor.add_to_buffer("chunk1")
        processor.add_to_buffer("chunk2")

        flushed = processor.flush_buffer()

        assert flushed == ["chunk1", "chunk2"]
        assert processor.buffer == []

    def test_process_chunk(self):
        """Test processing chunks with buffering"""
        processor = TestStreamProcessorImpl(buffer_size=3)

        # First chunks return None (buffering)
        assert processor.process_chunk("chunk1") is None
        assert processor.process_chunk("chunk2") is None

        # Third chunk triggers processing
        result = processor.process_chunk("chunk3")
        assert result is not None
        assert result["processed_chunks"] == 3


class TestProcessorChain:
    """Test ProcessorChain functionality"""

    def test_chain_processing(self):
        """Test chaining processors"""
        # Create mock processors
        proc1 = Mock(spec=BaseProcessor)
        proc1.process.return_value = {"step1": "done"}

        proc2 = Mock(spec=BaseProcessor)
        proc2.process.return_value = {"step2": "done"}

        chain = ProcessorChain([proc1, proc2])

        result = chain.process("input_data")

        assert proc1.process.called
        assert proc2.process.called
        assert result == {"step2": "done"}

    def test_chain_error_handling(self):
        """Test error handling in chain"""
        # Create processors where second one fails
        proc1 = Mock(spec=BaseProcessor)
        proc1.process.return_value = {"step1": "done"}

        proc2 = Mock(spec=BaseProcessor)
        proc2.process.side_effect = ValueError("Processing failed")

        chain = ProcessorChain([proc1, proc2])

        with pytest.raises(ValueError):
            chain.process("input_data")

    def test_add_remove_processor(self):
        """Test adding and removing processors from chain"""
        proc1 = Mock(spec=BaseProcessor)
        chain = ProcessorChain([proc1])

        assert len(chain.processors) == 1

        # Add processor
        proc2 = Mock(spec=BaseProcessor)
        chain.add_processor(proc2)
        assert len(chain.processors) == 2

        # Remove processor
        chain.remove_processor(0)
        assert len(chain.processors) == 1
        assert chain.processors[0] is proc2
