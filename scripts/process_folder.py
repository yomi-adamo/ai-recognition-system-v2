#!/usr/bin/env python3
"""
Process a folder of images for face detection using BatchProcessor
Usage: python scripts/process_folder.py <folder_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.batch_processor import BatchProcessor
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Process folder of images for face detection')
    parser.add_argument('folder_path', help='Path to input folder')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory (default: auto-generated timestamped directory)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process images recursively')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('--extensions', nargs='+', 
                       default=None,
                       help='File extensions to process (default: common image/video formats)')
    parser.add_argument('--no-recognition', action='store_true',
                       help='Disable face recognition')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted processing')
    parser.add_argument('--resume-file', default=None,
                       help='Path to resume state file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("process_folder", level=args.log_level)
    
    try:
        # Validate input
        folder_path = Path(args.folder_path)
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            sys.exit(1)
        
        if not folder_path.is_dir():
            logger.error(f"Path is not a directory: {folder_path}")
            sys.exit(1)
        
        # Initialize BatchProcessor
        batch_processor = BatchProcessor(
            max_workers=args.workers,
            enable_clustering=not args.no_recognition
        )
        
        # Progress callback
        def progress_callback(progress, current_file):
            print(f"Progress: {progress:.1%} - Processing: {current_file}")
        
        # Process folder
        logger.info(f"Starting batch processing of {folder_path}")
        report = batch_processor.process_directory(
            input_dir=folder_path,
            output_dir=args.output_dir,
            recursive=args.recursive,
            progress_callback=progress_callback
        )
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Total files: {report['summary']['total_files']}")
        print(f"Successfully processed: {report['summary']['processed_files']}")
        print(f"Failed: {report['summary']['failed_files']}")
        print(f"Total faces detected: {report['summary']['total_faces_detected']}")
        print(f"Success rate: {report['summary']['success_rate']:.1%}")
        print(f"\nPerformance:")
        print(f"  Total time: {report['performance']['total_time_seconds']:.2f} seconds")
        print(f"  Average time per file: {report['performance']['avg_time_per_file']:.2f} seconds")
        print(f"  Files per second: {report['performance']['files_per_second']:.2f}")
        print(f"  Workers used: {report['performance']['workers_used']}")
        
        if report['failed_files']:
            print(f"\nFailed files:")
            for file_path, error in report['failed_files'].items():
                print(f"  {file_path}: {error}")
        
        # Exit with appropriate code
        if report['summary']['failed_files'] > 0:
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()