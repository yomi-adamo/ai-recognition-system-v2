import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Set
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Manage input/output directories and file operations"""
    
    def __init__(self, base_output_dir: Union[str, Path] = None):
        """
        Initialize file handler
        
        Args:
            base_output_dir: Base directory for outputs (default: data/output)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.base_output_dir = Path(base_output_dir) if base_output_dir else self.project_root / "data" / "output"
        self.input_dir = self.project_root / "data" / "input"
        
        # Track processed files to avoid duplicates
        self.processed_files: Set[str] = set()
        
        # Create directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileHandler initialized: output={self.base_output_dir}")
    
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
    
    def save_face_chips(self, chips: List[Dict[str, Any]], 
                       output_dir: Union[str, Path]) -> List[Path]:
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
            if 'chip_array' in chip:
                import cv2
                
                # Generate filename
                chip_name = chip.get('name', f'face_chip_{idx:03d}')
                output_path = output_dir / f"{chip_name}.jpg"
                
                # Handle naming conflicts
                output_path = self._resolve_naming_conflict(output_path)
                
                # Save chip
                quality = [cv2.IMWRITE_JPEG_QUALITY, 85]
                cv2.imwrite(str(output_path), chip['chip_array'], quality)
                
                saved_paths.append(output_path)
                logger.debug(f"Saved face chip: {output_path}")
        
        logger.info(f"Saved {len(saved_paths)} face chips to {output_dir}")
        return saved_paths
    
    def save_json_output(self, data: Union[List[Dict], Dict[str, Any]], 
                        output_path: Union[str, Path],
                        pretty_print: bool = True) -> Path:
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
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        logger.info(f"Saved JSON output: {output_path}")
        return output_path
    
    def get_input_files(self, extensions: List[str] = None, 
                       recursive: bool = False) -> List[Path]:
        """
        Get list of input files
        
        Args:
            extensions: File extensions to include (e.g., ['.jpg', '.png'])
            recursive: Whether to search recursively
            
        Returns:
            List of input file paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
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
        if hasattr(self, 'track_processed') and self.track_processed:
            files = [f for f in files if str(f) not in self.processed_files]
        
        logger.info(f"Found {len(files)} input files")
        return sorted(files)
    
    def mark_as_processed(self, file_path: Union[str, Path]) -> None:
        """Mark a file as processed"""
        self.processed_files.add(str(Path(file_path).absolute()))
    
    def is_processed(self, file_path: Union[str, Path]) -> bool:
        """Check if a file has been processed"""
        return str(Path(file_path).absolute()) in self.processed_files
    
    def save_processing_log(self, log_data: Dict[str, Any], 
                          output_dir: Union[str, Path]) -> Path:
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
        if 'timestamp' not in log_data:
            log_data['timestamp'] = datetime.now().isoformat()
        
        return self.save_json_output(log_data, log_path)
    
    def cleanup_output_dir(self, output_dir: Union[str, Path], 
                          keep_recent: int = 5) -> None:
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
        dirs = [d for d in output_dir.iterdir() if d.is_dir() and '_' in d.name]
        
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
    
    def copy_to_output(self, source_path: Union[str, Path], 
                      output_dir: Union[str, Path],
                      preserve_structure: bool = False) -> Path:
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
            "file_count": file_count
        }