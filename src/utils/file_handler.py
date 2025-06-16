import hashlib
import json
import mimetypes
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.utils.config import get_config
from src.utils.logger import get_facial_vision_logger

logger = get_facial_vision_logger(__name__)


class FileHandler:
    """Manage input/output directories and file operations"""

    def __init__(self, base_output_dir: Union[str, Path] = None):
        """
        Initialize file handler

        Args:
            base_output_dir: Base directory for outputs (default: from config)
        """
        self.config = get_config()
        self.paths = self.config.get_paths_config()

        self.project_root = Path(__file__).parent.parent.parent
        self.base_output_dir = Path(base_output_dir) if base_output_dir else self.paths["output"]
        self.input_dir = self.paths["input"]
        self.temp_dir = self.paths["temp"]

        # Track processed files to avoid duplicates
        self.processed_files: Set[str] = set()
        self.file_hashes: Dict[str, str] = {}

        # Create directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Supported file types
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        logger.info(f"FileHandler initialized", output_dir=str(self.base_output_dir))

    def create_timestamped_output_dir(self, prefix: str = "run") -> Path:
        """
        Create a timestamped output directory

        Args:
            prefix: Prefix for the directory name

        Returns:
            Path to the created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.base_output_dir / f"{prefix}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def save_face_chips(
        self, chips: List[Dict[str, Any]], output_dir: Union[str, Path]
    ) -> List[Path]:
        """
        Save face chips to directory

        Args:
            chips: List of chip dictionaries with chip_array
            output_dir: Directory to save chips

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for idx, chip in enumerate(chips):
            if "chip_array" in chip:
                import cv2

                # Generate filename
                chip_name = chip.get("name", f"face_chip_{idx:03d}")
                output_path = output_dir / f"{chip_name}.jpg"

                # Handle naming conflicts
                output_path = self._resolve_naming_conflict(output_path)

                # Save chip
                quality = [cv2.IMWRITE_JPEG_QUALITY, 85]
                cv2.imwrite(str(output_path), chip["chip_array"], quality)

                saved_paths.append(output_path)
                logger.debug(f"Saved face chip: {output_path}")

        logger.info(f"Saved {len(saved_paths)} face chips to {output_dir}")
        return saved_paths

    def save_json_output(
        self,
        data: Union[List[Dict], Dict[str, Any]],
        output_path: Union[str, Path],
        pretty_print: bool = True,
    ) -> Path:
        """
        Save JSON data to file

        Args:
            data: Data to save
            output_path: Output file path
            pretty_print: Whether to format with indentation

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle naming conflicts
        output_path = self._resolve_naming_conflict(output_path)

        # Save JSON
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty_print:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        logger.info(f"Saved JSON output: {output_path}")
        return output_path

    def get_input_files(self, extensions: List[str] = None, recursive: bool = False) -> List[Path]:
        """
        Get list of input files

        Args:
            extensions: File extensions to include (e.g., ['.jpg', '.png'])
            recursive: Whether to search recursively

        Returns:
            List of input file paths
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Normalize extensions
        extensions = [ext.lower() for ext in extensions]

        files = []

        if recursive:
            for ext in extensions:
                files.extend(self.input_dir.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                files.extend(self.input_dir.glob(f"*{ext}"))

        # Filter out already processed files if tracking
        if hasattr(self, "track_processed") and self.track_processed:
            files = [f for f in files if str(f) not in self.processed_files]

        logger.info(f"Found {len(files)} input files")
        return sorted(files)

    def mark_as_processed(self, file_path: Union[str, Path]) -> None:
        """Mark a file as processed"""
        self.processed_files.add(str(Path(file_path).absolute()))

    def is_processed(self, file_path: Union[str, Path]) -> bool:
        """Check if a file has been processed"""
        return str(Path(file_path).absolute()) in self.processed_files

    def save_processing_log(self, log_data: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
        """
        Save processing log with statistics and metadata

        Args:
            log_data: Processing statistics and metadata
            output_dir: Directory to save log

        Returns:
            Path to saved log file
        """
        output_dir = Path(output_dir)
        log_path = output_dir / "processing_log.json"

        # Add timestamp if not present
        if "timestamp" not in log_data:
            log_data["timestamp"] = datetime.now().isoformat()

        return self.save_json_output(log_data, log_path)

    def cleanup_output_dir(self, output_dir: Union[str, Path], keep_recent: int = 5) -> None:
        """
        Clean up old output directories, keeping only recent ones

        Args:
            output_dir: Base output directory
            keep_recent: Number of recent directories to keep
        """
        output_dir = Path(output_dir)

        if not output_dir.exists():
            return

        # Get all timestamped directories
        dirs = [d for d in output_dir.iterdir() if d.is_dir() and "_" in d.name]

        # Sort by creation time
        dirs.sort(key=lambda x: x.stat().st_ctime)

        # Remove old directories
        to_remove = dirs[:-keep_recent] if len(dirs) > keep_recent else []

        for old_dir in to_remove:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"Removed old output directory: {old_dir}")
            except Exception as e:
                logger.error(f"Failed to remove directory {old_dir}: {e}")

    def copy_to_output(
        self,
        source_path: Union[str, Path],
        output_dir: Union[str, Path],
        preserve_structure: bool = False,
    ) -> Path:
        """
        Copy a file to output directory

        Args:
            source_path: Source file path
            output_dir: Output directory
            preserve_structure: Whether to preserve directory structure

        Returns:
            Path to copied file
        """
        source_path = Path(source_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if preserve_structure:
            # Preserve relative path structure
            rel_path = source_path.relative_to(self.input_dir)
            dest_path = output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest_path = output_dir / source_path.name

        # Handle naming conflicts
        dest_path = self._resolve_naming_conflict(dest_path)

        # Copy file
        shutil.copy2(source_path, dest_path)
        logger.debug(f"Copied {source_path} to {dest_path}")

        return dest_path

    def _resolve_naming_conflict(self, file_path: Path) -> Path:
        """
        Resolve naming conflicts by adding a counter

        Args:
            file_path: Desired file path

        Returns:
            Available file path
        """
        if not file_path.exists():
            return file_path

        # Extract parts
        stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent

        # Find available name
        counter = 1
        while True:
            new_name = f"{stem}_{counter:03d}{suffix}"
            new_path = parent / new_name

            if not new_path.exists():
                return new_path

            counter += 1

            # Safety check
            if counter > 999:
                raise ValueError(f"Too many naming conflicts for {file_path}")

    def get_disk_usage(self, directory: Union[str, Path]) -> Dict[str, int]:
        """
        Get disk usage statistics for a directory

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary with size information
        """
        directory = Path(directory)

        if not directory.exists():
            return {"total_size": 0, "file_count": 0}

        total_size = 0
        file_count = 0

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
        }

    def create_cluster_directories(
        self, cluster_ids: List[str], output_dir: Union[str, Path]
    ) -> Dict[str, Path]:
        """
        Create directories for face clusters

        Args:
            cluster_ids: List of cluster identifiers (e.g., ['person_1', 'person_2'])
            output_dir: Base output directory

        Returns:
            Dictionary mapping cluster_id to directory path
        """
        output_dir = Path(output_dir)
        cluster_dirs = {}

        for cluster_id in cluster_ids:
            cluster_dir = output_dir / cluster_id
            cluster_dir.mkdir(parents=True, exist_ok=True)
            cluster_dirs[cluster_id] = cluster_dir
            logger.debug(f"Created cluster directory", cluster_id=cluster_id, path=str(cluster_dir))

        logger.info(f"Created {len(cluster_dirs)} cluster directories")
        return cluster_dirs

    def save_clustered_face_chips(
        self, chips_by_cluster: Dict[str, List[Dict[str, Any]]], output_dir: Union[str, Path]
    ) -> Dict[str, List[Path]]:
        """
        Save face chips organized by cluster

        Args:
            chips_by_cluster: Dictionary mapping cluster_id to list of chip data
            output_dir: Base output directory

        Returns:
            Dictionary mapping cluster_id to list of saved file paths
        """
        output_dir = Path(output_dir)
        saved_paths = {}

        # Create cluster directories
        cluster_ids = list(chips_by_cluster.keys())
        cluster_dirs = self.create_cluster_directories(cluster_ids, output_dir)

        # Save chips to respective cluster directories
        for cluster_id, chips in chips_by_cluster.items():
            cluster_dir = cluster_dirs[cluster_id]
            saved_paths[cluster_id] = self.save_face_chips(chips, cluster_dir)

        total_chips = sum(len(paths) for paths in saved_paths.values())
        logger.info(f"Saved {total_chips} chips across {len(cluster_ids)} clusters")

        return saved_paths

    def calculate_file_hash(self, file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """
        Calculate hash of a file

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (sha256, md5, etc.)

        Returns:
            Hex digest of file hash
        """
        file_path = Path(file_path)

        # Check cache first
        cache_key = str(file_path.absolute())
        if cache_key in self.file_hashes:
            return self.file_hashes[cache_key]

        hash_obj = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        file_hash = hash_obj.hexdigest()

        # Cache the result
        self.file_hashes[cache_key] = file_hash

        logger.debug(f"Calculated {algorithm} hash", file=str(file_path), hash=file_hash[:8])
        return file_hash

    def get_file_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file metadata

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)
        stat = file_path.stat()

        metadata = {
            "name": file_path.name,
            "path": str(file_path.absolute()),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower(),
            "mime_type": mimetypes.guess_type(str(file_path))[0],
            "hash": self.calculate_file_hash(file_path),
        }

        return metadata

    def batch_process_files(
        self, file_paths: List[Path], process_func: Callable[[Path], Any], max_workers: int = 4
    ) -> List[Tuple[Path, Any]]:
        """
        Process multiple files in parallel

        Args:
            file_paths: List of file paths to process
            process_func: Function to apply to each file
            max_workers: Maximum number of worker threads

        Returns:
            List of (file_path, result) tuples
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(process_func, path): path for path in file_paths}

            # Collect results as they complete
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append((file_path, result))
                except Exception as e:
                    logger.error(f"Error processing file", file=str(file_path), error=str(e))
                    results.append((file_path, None))

        return results

    def organize_files_by_type(self, input_dir: Union[str, Path] = None) -> Dict[str, List[Path]]:
        """
        Organize files by type (images, videos, etc.)

        Args:
            input_dir: Directory to scan (default: self.input_dir)

        Returns:
            Dictionary mapping file type to list of paths
        """
        input_dir = Path(input_dir) if input_dir else self.input_dir

        organized = {"images": [], "videos": [], "other": []}

        for file_path in input_dir.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()

                if ext in self.image_extensions:
                    organized["images"].append(file_path)
                elif ext in self.video_extensions:
                    organized["videos"].append(file_path)
                else:
                    organized["other"].append(file_path)

        logger.info(
            f"Organized files",
            images=len(organized["images"]),
            videos=len(organized["videos"]),
            other=len(organized["other"]),
        )

        return organized

    def clean_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files

        Args:
            max_age_hours: Maximum age of temp files in hours

        Returns:
            Number of files removed
        """
        removed_count = 0
        current_time = datetime.now()

        for file_path in self.temp_dir.rglob("*"):
            if file_path.is_file():
                file_age_hours = (
                    current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                ).total_seconds() / 3600

                if file_age_hours > max_age_hours:
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to remove temp file", file=str(file_path), error=str(e)
                        )

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} temporary files")

        return removed_count
