"""
Tests for logging framework
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pytest

from src.utils.logger import (
    LoggerAdapter,
    StructuredFormatter,
    get_facial_vision_logger,
    get_logger,
    initialize_logging,
    performance_logger,
    setup_logger,
    timing_decorator,
)


class TestLogger:
    """Test logging functionality"""

    def test_setup_logger(self, temp_dir):
        """Test basic logger setup"""
        logger = setup_logger(
            name="test_logger",
            level="DEBUG",
            log_dir=str(temp_dir),
            console=True,
            json_format=False,
        )

        assert logger.name == "test_logger"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2  # File and console

    def test_structured_formatter(self):
        """Test JSON structured formatter"""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert "timestamp" in log_data

    def test_get_logger(self):
        """Test getting logger by name"""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2
        assert logger1.name == "test.module"

    def test_timing_decorator(self):
        """Test function timing decorator"""
        call_count = 0

        @timing_decorator
        def test_function():
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return "result"

        result = test_function()

        assert result == "result"
        assert call_count == 1

    def test_timing_decorator_with_exception(self):
        """Test timing decorator handles exceptions"""

        @timing_decorator
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

    def test_performance_logger_context(self):
        """Test performance logger context manager"""
        with performance_logger("test_operation") as perf:
            time.sleep(0.01)

        # Context manager should complete without errors
        assert True

    def test_performance_logger_with_exception(self):
        """Test performance logger handles exceptions"""
        with pytest.raises(ValueError):
            with performance_logger("failing_operation"):
                raise ValueError("Test error")

    def test_logger_adapter(self):
        """Test LoggerAdapter for extra fields"""
        base_logger = get_logger("test_adapter")
        adapter = LoggerAdapter(base_logger, {"user_id": "123", "session": "abc"})

        # The adapter should add extra fields to log records
        assert adapter.extra == {"user_id": "123", "session": "abc"}

    def test_facial_vision_logger(self):
        """Test FacialVisionLogger unified interface"""
        logger = get_facial_vision_logger("test_module")

        assert logger.name == "test_module"
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "critical")
        assert hasattr(logger, "exception")

    def test_facial_vision_logger_with_context(self):
        """Test FacialVisionLogger context variables"""
        logger = get_facial_vision_logger("test_context")

        # Test with_context method
        contextualized = logger.with_context(user_id="123", request_id="abc")
        assert contextualized is not None

    def test_initialize_logging(self, temp_dir, monkeypatch):
        """Test logging system initialization"""

        # Mock config to use temp directory
        def mock_get_paths_config():
            return {"logs": temp_dir}

        from src.utils import logger as logger_module

        monkeypatch.setattr(logger_module.get_config().get_paths_config, mock_get_paths_config)

        initialize_logging()

        # Check that log directory was created
        assert temp_dir.exists()

    def test_convenience_functions(self):
        """Test module-level convenience logging functions"""
        from src.utils.logger import critical, debug, error, info, warning

        # These should not raise exceptions
        debug("Debug message")
        info("Info message")
        warning("Warning message")
        error("Error message")
        critical("Critical message")

        assert True  # If we get here, all functions worked


class TestLogOutput:
    """Test actual log output"""

    def test_log_file_creation(self, temp_dir):
        """Test that log files are created correctly"""
        logger = setup_logger(name="file_test", level="INFO", log_dir=str(temp_dir), console=False)

        logger.info("Test message")

        # Check that log file was created
        log_files = list(temp_dir.glob("*.log"))
        assert len(log_files) == 1

        # Check log content
        with open(log_files[0], "r") as f:
            content = f.read()
            assert "Test message" in content

    def test_json_log_format(self, temp_dir):
        """Test JSON formatted logs"""
        logger = setup_logger(
            name="json_test", level="INFO", log_dir=str(temp_dir), console=False, json_format=True
        )

        logger.info("JSON test message")

        # Read and parse JSON log
        log_files = list(temp_dir.glob("*.log"))
        with open(log_files[0], "r") as f:
            line = f.readline()
            log_data = json.loads(line)

            assert log_data["message"] == "JSON test message"
            assert log_data["level"] == "INFO"
            assert "timestamp" in log_data
