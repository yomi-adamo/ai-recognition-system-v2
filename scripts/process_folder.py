#!/usr/bin/env python3
"""
Process a folder of images for face detection
Usage: python scripts/process_folder.py <folder_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.image_processor import ImageProcessor
from src.utils.file_handler import FileHandler
from src.utils.logger import setup_logger
from src.outputs.json_formatter import JSONFormatter


def process_single_image(args_tuple):
    """Process a single image - used for multiprocessing"""
    image_path, output_dir, save_chips = args_tuple
    
    try:
        processor = ImageProcessor()
        results = processor.process_image(
            image_path,
            output_dir=output_dir if save_chips else None,
            save_chips=save_chips
        )
        
        return {
            'image_path': str(image_path),
            'success': True,
            'face_count': len(results),
            'results': results,
            'error': None
        }
    except Exception as e:
        return {
            'image_path': str(image_path),
            'success': False,
            'face_count': 0,
            'results': [],
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Process folder of images for face detection')
    parser.add_argument('folder_path', help='Path to input folder')
    parser.add_argument('--output-dir', '-o', default='data/output',
                       help='Output directory (default: data/output)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process images recursively')
    parser.add_argument('--workers', '-w', type=int, 
                       default=multiprocessing.cpu_count(),
                       help='Number of worker processes')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='Image file extensions to process')
    parser.add_argument('--no-save-chips', action='store_true',
                       help='Do not save face chips to disk')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted processing')
    
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
        
        # Initialize components
        file_handler = FileHandler(args.output_dir)
        formatter = JSONFormatter()
        
        # Create output directory
        output_dir = file_handler.create_timestamped_output_dir("batch_processing")
        
        # Get image files
        extensions = [ext.lower() for ext in args.extensions]
        
        image_files = []
        if args.recursive:
            for ext in extensions:
                image_files.extend(folder_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
        
        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        
        if len(image_files) == 0:
            logger.warning(f"No image files found in {folder_path}")
            print("No images to process")
            sys.exit(0)
        
        logger.info(f"Found {len(image_files)} images to process")
        print(f"Processing {len(image_files)} images with {args.workers} workers...")
        
        # Prepare processing arguments
        save_chips = not args.no_save_chips
        processing_args = [
            (img_path, output_dir, save_chips) 
            for img_path in image_files
        ]
        
        # Process images
        start_time = time.time()
        all_results = []
        successful_count = 0
        failed_count = 0
        total_faces = 0
        
        if args.workers == 1:
            # Single-threaded processing
            for i, args_tuple in enumerate(processing_args):
                result = process_single_image(args_tuple)
                all_results.append(result)
                
                if result['success']:
                    successful_count += 1
                    total_faces += result['face_count']
                else:
                    failed_count += 1
                    logger.error(f"Failed to process {result['image_path']}: {result['error']}")
                
                # Progress update
                if (i + 1) % 10 == 0 or i == len(processing_args) - 1:
                    progress = (i + 1) / len(processing_args) * 100
                    print(f"Progress: {i + 1}/{len(processing_args)} ({progress:.1f}%)")
        else:
            # Multi-threaded processing
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Submit all jobs
                future_to_args = {
                    executor.submit(process_single_image, args_tuple): args_tuple 
                    for args_tuple in processing_args
                }
                
                # Collect results
                for i, future in enumerate(as_completed(future_to_args)):
                    result = future.result()
                    all_results.append(result)
                    
                    if result['success']:
                        successful_count += 1
                        total_faces += result['face_count']
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process {result['image_path']}: {result['error']}")
                    
                    # Progress update
                    if (i + 1) % 10 == 0 or i == len(processing_args) - 1:
                        progress = (i + 1) / len(processing_args) * 100
                        print(f"Progress: {i + 1}/{len(processing_args)} ({progress:.1f}%)")
        
        # Collect all face results
        all_face_results = []
        for result in all_results:
            if result['success'] and result['results']:
                all_face_results.extend(result['results'])
        
        processing_time = time.time() - start_time
        
        # Save results
        if all_face_results:
            # Save all faces JSON
            all_faces_path = output_dir / "all_faces.json"
            formatter.save_to_file(all_face_results, all_faces_path)
            
            # Create summary report
            summary = formatter.create_summary_report(all_face_results)
            summary.update({
                'processing_stats': {
                    'total_images': len(image_files),
                    'successful_images': successful_count,
                    'failed_images': failed_count,
                    'total_faces_detected': total_faces,
                    'processing_time_seconds': round(processing_time, 2),
                    'images_per_second': round(len(image_files) / processing_time, 2),
                    'workers_used': args.workers
                },
                'failed_files': [
                    r['image_path'] for r in all_results if not r['success']
                ]
            })
            
            summary_path = output_dir / "processing_summary.json"
            file_handler.save_json_output(summary, summary_path)
            
            # Save detailed results
            detailed_results_path = output_dir / "detailed_results.json"
            file_handler.save_json_output(all_results, detailed_results_path)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {successful_count}")
        print(f"Failed: {failed_count}")
        print(f"Total faces detected: {total_faces}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Speed: {len(image_files)/processing_time:.2f} images/second")
        print(f"Output directory: {output_dir}")
        
        if all_face_results:
            print(f"All faces JSON: {output_dir}/all_faces.json")
            print(f"Summary report: {output_dir}/processing_summary.json")
        
        if failed_count > 0:
            print(f"\nFailed files:")
            for result in all_results:
                if not result['success']:
                    print(f"  {result['image_path']}: {result['error']}")
    
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()