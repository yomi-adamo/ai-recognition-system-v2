import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.config import get_config
from src.utils.file_handler import FileHandler
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for all processors in the facial vision system
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base processor

        Args:
            config: Optional configuration dictionary
        """
        self.config_manager = get_config()
        self.config = config or {}
        self.file_handler = FileHandler()

        # Processing statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "files_processed": 0,
            "errors": 0,
            "processing_time": 0.0,
        }

        self._initialize()
        logger.info(f"{self.__class__.__name__} initialized")

    def _initialize(self):
        """
        Hook for subclasses to perform additional initialization
        """
        pass

    @abstractmethod
    def process(self, input_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process a single input file

        Args:
            input_path: Path to input file
            **kwargs: Additional processing parameters

        Returns:
            Processing results dictionary
        """
        pass

    def preprocess(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Preprocessing hook that runs before main processing

        Args:
            input_path: Path to input file

        Returns:
            Preprocessing metadata
        """
        input_path = Path(input_path)

        # Validate file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Get file metadata
        metadata = self.file_handler.get_file_metadata(input_path)

        logger.debug(f"Preprocessing", file=metadata["name"], size_mb=metadata["size_mb"])

        return metadata

    def postprocess(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocessing hook that runs after main processing

        Args:
            results: Processing results

        Returns:
            Modified results
        """
        # Add common metadata
        results["processor"] = self.__class__.__name__
        results["processing_timestamp"] = datetime.utcnow().isoformat()

        return results

    def validate_input(self, input_path: Union[str, Path]) -> bool:
        """
        Validate input file is suitable for processing

        Args:
            input_path: Path to input file

        Returns:
            True if valid, False otherwise
        """
        return Path(input_path).exists()

    def handle_error(self, error: Exception, context: Dict[str, Any] = None):
        """
        Handle processing errors

        Args:
            error: Exception that occurred
            context: Additional error context
        """
        self.stats["errors"] += 1

        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if context:
            error_data.update(context)

        logger.error(f"Processing error", **error_data)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics

        Returns:
            Statistics dictionary
        """
        if self.stats["start_time"] and self.stats["end_time"]:
            self.stats["processing_time"] = self.stats["end_time"] - self.stats["start_time"]

        return self.stats.copy()

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            "start_time": None,
            "end_time": None,
            "files_processed": 0,
            "errors": 0,
            "processing_time": 0.0,
        }


class BatchProcessorBase(BaseProcessor):
    """
    Base class for processors that handle multiple files
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, max_workers: int = 4):
        """
        Initialize batch processor

        Args:
            config: Optional configuration dictionary
            max_workers: Maximum number of worker threads/processes
        """
        super().__init__(config)
        self.max_workers = max_workers
        self.results = []

    @abstractmethod
    def process_batch(self, input_paths: List[Union[str, Path]], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple input files

        Args:
            input_paths: List of input file paths
            **kwargs: Additional processing parameters

        Returns:
            List of processing results
        """
        pass

    def filter_inputs(self, input_paths: List[Union[str, Path]]) -> List[Path]:
        """
        Filter input paths based on validation criteria

        Args:
            input_paths: List of input paths

        Returns:
            Filtered list of valid paths
        """
        valid_paths = []

        for path in input_paths:
            path = Path(path)
            if self.validate_input(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Skipping invalid input", path=str(path))

        return valid_paths

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate batch processing results

        Args:
            results: List of individual processing results

        Returns:
            Aggregated results summary
        """
        summary = {
            "total_processed": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results,
        }

        return summary


class StreamProcessor(BaseProcessor):
    """
    Base class for processors that handle streaming data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, buffer_size: int = 1024):
        """
        Initialize stream processor

        Args:
            config: Optional configuration dictionary
            buffer_size: Size of processing buffer
        """
        super().__init__(config)
        self.buffer_size = buffer_size
        self.buffer = []

    @abstractmethod
    def process_chunk(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """
        Process a single chunk of streaming data

        Args:
            chunk: Data chunk to process

        Returns:
            Processing results or None if buffering
        """
        pass

    def add_to_buffer(self, data: Any) -> bool:
        """
        Add data to processing buffer

        Args:
            data: Data to buffer

        Returns:
            True if buffer is full
        """
        self.buffer.append(data)
        return len(self.buffer) >= self.buffer_size

    def flush_buffer(self) -> List[Any]:
        """
        Flush and return buffer contents

        Returns:
            Buffer contents
        """
        data = self.buffer.copy()
        self.buffer.clear()
        return data


class ProcessorChain:
    """
    Chain multiple processors together for sequential processing
    """

    def __init__(self, processors: List[BaseProcessor]):
        """
        Initialize processor chain

        Args:
            processors: List of processors to chain
        """
        self.processors = processors
        self.logger = get_facial_vision_logger(__name__)

    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process data through the chain

        Args:
            input_data: Initial input data
            **kwargs: Additional parameters passed to each processor

        Returns:
            Final processed result
        """
        result = input_data

        for i, processor in enumerate(self.processors):
            self.logger.debug(
                f"Running processor {i+1}/{len(self.processors)}",
                processor=processor.__class__.__name__,
            )

            try:
                result = processor.process(result, **kwargs)
            except Exception as e:
                self.logger.error(
                    f"Chain processing failed at step {i+1}",
                    error=str(e),
                    processor=processor.__class__.__name__,
                )
                raise

        return result

    def add_processor(self, processor: BaseProcessor):
        """Add a processor to the chain"""
        self.processors.append(processor)

    def remove_processor(self, index: int):
        """Remove a processor from the chain"""
        if 0 <= index < len(self.processors):
            self.processors.pop(index)
