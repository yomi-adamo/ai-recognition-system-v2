"""
Batch processor for handling multiple files and bulk operations
"""
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.processors.media_processor_base import MediaProcessorBase
from src.utils.config import get_config
from src.utils.logger import get_facial_vision_logger, timing_decorator

logger = get_facial_vision_logger(__name__)


class BatchProcessor(MediaProcessorBase):
    """Process multiple images and videos in batch with clustering and metadata extraction"""

    def __init__(self, enable_clustering: bool = True, max_workers: int = None):
        """
        Initialize batch processor

        Args:
            enable_clustering: Whether to enable face clustering
            max_workers: Maximum number of worker threads for parallel processing
        """
        super().__init__()
        
        self.enable_clustering = enable_clustering
        self.max_workers = max_workers or 4

        # Initialize processors
        self.image_processor = ImageProcessor(enable_clustering=enable_clustering)
        self.video_processor = VideoProcessor(enable_clustering=enable_clustering)

        # Supported file extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}

        logger.info(
            f"BatchProcessor initialized: clustering={enable_clustering}, workers={self.max_workers}"
        )

    @timing_decorator
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = True,
        save_chips: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Process all supported files in a directory

        Args:
            input_dir: Directory containing files to process
            output_dir: Directory to save outputs
            recursive: Whether to process subdirectories recursively
            save_chips: Whether to save face chips to disk
            progress_callback: Optional callback for progress updates (progress, current_file)
            file_filter: Optional function to filter files

        Returns:
            Dictionary with comprehensive batch processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing directory: {input_dir}")

        # Discover files
        files = self._discover_files(input_dir, recursive, file_filter)
        
        if not files:
            logger.warning(f"No supported files found in {input_dir}")
            return self._create_empty_batch_result(input_dir, output_dir)

        logger.info(f"Found {len(files)} files to process")

        # Process files
        return self._process_files_batch(
            files=files,
            output_dir=output_dir,
            save_chips=save_chips,
            progress_callback=progress_callback
        )

    @timing_decorator
    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        save_chips: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a specific list of files

        Args:
            file_paths: List of file paths to process
            output_dir: Directory to save outputs
            save_chips: Whether to save face chips to disk
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with comprehensive batch processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate files
        valid_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists() and self._is_supported_file(path):
                valid_files.append(path)
            else:
                logger.warning(f"Skipping unsupported or missing file: {path}")

        if not valid_files:
            logger.warning("No valid files to process")
            return self._create_empty_batch_result(Path("."), output_dir)

        logger.info(f"Processing {len(valid_files)} files")

        return self._process_files_batch(
            files=valid_files,
            output_dir=output_dir,
            save_chips=save_chips,
            progress_callback=progress_callback
        )

    def _process_files_batch(
        self,
        files: List[Path],
        output_dir: Path,
        save_chips: bool,
        progress_callback: Optional[Callable[[float, str], None]],
    ) -> Dict[str, Any]:
        """Process files in batch with parallel processing"""
        
        results = {
            "images": [],
            "videos": [],
            "errors": [],
        }
        
        completed_count = 0
        total_files = len(files)

        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_file = {}
            
            for file_path in files:
                if self._is_image_file(file_path):
                    future = executor.submit(
                        self._process_single_image,
                        file_path,
                        output_dir,
                        save_chips
                    )
                elif self._is_video_file(file_path):
                    future = executor.submit(
                        self._process_single_video,
                        file_path,
                        output_dir,
                        save_chips
                    )
                else:
                    continue
                    
                future_to_file[future] = file_path

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    
                    if result["type"] == "image":
                        results["images"].append(result)
                    elif result["type"] == "video":
                        results["videos"].append(result)
                        
                    logger.info(f"Completed {file_path.name} ({completed_count}/{total_files})")
                    
                except Exception as e:
                    error_info = {
                        "file": str(file_path),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Failed to process {file_path}: {e}")

                # Progress callback
                if progress_callback:
                    progress = completed_count / total_files
                    progress_callback(progress, str(file_path))

        # Calculate statistics
        total_faces = 0
        total_clusters = set()
        
        for result in results["images"] + results["videos"]:
            chips = result.get("metadata", {}).get("chips", [])
            total_faces += len(chips)
            
            for chip in chips:
                cluster_id = chip.get("clusterId")
                if cluster_id:
                    total_clusters.add(cluster_id)

        # Create comprehensive batch result
        batch_result = {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "input_files": len(files),
            "output_directory": str(output_dir),
            "processing_stats": {
                "images_processed": len(results["images"]),
                "videos_processed": len(results["videos"]),
                "total_files_processed": len(results["images"]) + len(results["videos"]),
                "failed_files": len(results["errors"]),
                "total_faces_detected": total_faces,
                "unique_clusters": len(total_clusters),
                "clustering_enabled": self.enable_clustering,
            },
            "results": results,
            "cluster_summary": list(total_clusters) if total_clusters else [],
        }

        logger.info(
            f"Batch processing complete: {batch_result['processing_stats']['total_files_processed']} "
            f"files processed, {total_faces} faces detected, {len(total_clusters)} clusters"
        )

        return batch_result

    def _process_single_image(
        self, image_path: Path, output_dir: Path, save_chips: bool
    ) -> Dict[str, Any]:
        """Process a single image file"""
        try:
            result = self.image_processor.process_image(
                image_path=image_path,
                output_dir=output_dir / "images" / image_path.stem,
                save_chips=save_chips
            )
            result["batch_file_type"] = "image"
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def _process_single_video(
        self, video_path: Path, output_dir: Path, save_chips: bool
    ) -> Dict[str, Any]:
        """Process a single video file"""
        try:
            result = self.video_processor.process_video(
                video_path=video_path,
                output_dir=output_dir / "videos" / video_path.stem,
                save_chips=save_chips
            )
            result["batch_file_type"] = "video"
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            raise

    def _discover_files(
        self,
        input_dir: Path,
        recursive: bool,
        file_filter: Optional[Callable[[Path], bool]],
    ) -> List[Path]:
        """Discover supported files in directory"""
        files = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
            
        for file_path in input_dir.glob(pattern):
            if file_path.is_file() and self._is_supported_file(file_path):
                if file_filter is None or file_filter(file_path):
                    files.append(file_path)
                    
        return sorted(files)

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported for processing"""
        extension = file_path.suffix.lower()
        return extension in self.image_extensions or extension in self.video_extensions

    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is an image"""
        return file_path.suffix.lower() in self.image_extensions

    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video"""
        return file_path.suffix.lower() in self.video_extensions

    def _create_empty_batch_result(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "input_files": 0,
            "output_directory": str(output_dir),
            "processing_stats": {
                "images_processed": 0,
                "videos_processed": 0,
                "total_files_processed": 0,
                "failed_files": 0,
                "total_faces_detected": 0,
                "unique_clusters": 0,
                "clustering_enabled": self.enable_clustering,
            },
            "results": {
                "images": [],
                "videos": [],
                "errors": [],
            },
            "cluster_summary": [],
        }

    def save_batch_results(
        self, batch_result: Dict[str, Any], output_path: Union[str, Path]
    ) -> None:
        """Save batch processing results to JSON file"""
        output_path = Path(output_path)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(batch_result, f, indent=2)
            
        logger.info(f"Saved batch results to {output_path}")

    def load_media(self, file_path: Path) -> Any:
        """Load media file - delegated to appropriate processor"""
        if self._is_image_file(file_path):
            return self.image_processor.load_media(file_path)
        elif self._is_video_file(file_path):
            return self.video_processor.load_media(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def extract_frames(self, media: Any) -> List[Any]:
        """Extract frames - returns single frame for images, multiple for videos"""
        # This is a simplified implementation for the abstract requirement
        # In practice, batch processor handles this through individual processors
        if isinstance(media, type(None)):
            return []
        return [media]

    def process(self, input_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process method required by BaseProcessor - process single file
        
        Args:
            input_path: Path to input file
            **kwargs: Additional processing parameters
            
        Returns:
            Processing results dictionary
        """
        input_path = Path(input_path)
        
        if self._is_image_file(input_path):
            return self.image_processor.process_image(input_path, **kwargs)
        elif self._is_video_file(input_path):
            return self.video_processor.process_video(input_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {input_path}")

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get combined clustering statistics from all processors"""
        stats = {
            "image_processor": self.image_processor.get_cluster_statistics(),
            "video_processor": self.video_processor.get_cluster_statistics(),
        }
        
        # Combine stats if available
        if (stats["image_processor"].get("total_clusters", 0) > 0 or 
            stats["video_processor"].get("total_clusters", 0) > 0):
            
            combined_stats = {
                "total_clusters": max(
                    stats["image_processor"].get("total_clusters", 0),
                    stats["video_processor"].get("total_clusters", 0)
                ),
                "clustering_enabled": self.enable_clustering,
            }
            stats["combined"] = combined_stats
            
        return stats

    def create_processing_summary(
        self, batch_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of processing results"""
        stats = batch_result["processing_stats"]
        
        summary = {
            "batch_id": batch_result["batch_id"],
            "processing_date": batch_result["timestamp"],
            "files_summary": {
                "total_input_files": batch_result["input_files"],
                "successfully_processed": stats["total_files_processed"],
                "failed_files": stats["failed_files"],
                "images": stats["images_processed"],
                "videos": stats["videos_processed"],
            },
            "detection_summary": {
                "total_faces_detected": stats["total_faces_detected"],
                "unique_clusters": stats["unique_clusters"],
                "clustering_enabled": stats["clustering_enabled"],
            },
            "output_location": batch_result["output_directory"],
        }
        
        # Add cluster breakdown
        if batch_result.get("cluster_summary"):
            summary["clusters"] = batch_result["cluster_summary"]
            
        return summary