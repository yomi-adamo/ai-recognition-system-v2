#!/usr/bin/env python3
"""
Process a single image for face detection and generate JSON output
Usage: python scripts/process_image.py <image_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processors.image_processor import ImageProcessor
from src.utils.file_handler import FileHandler
from src.utils.logger import setup_logger
from src.outputs.json_formatter import JSONFormatter


def main():
    parser = argparse.ArgumentParser(description='Process image for face detection')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', '-o', default='data/output', 
                       help='Output directory (default: data/output)')
    parser.add_argument('--base64', action='store_true', 
                       help='Include base64 encoded chips in JSON')
    parser.add_argument('--no-save-chips', action='store_true',
                       help='Do not save face chips to disk')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--json-only', action='store_true',
                       help='Output JSON to stdout only')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("process_image", level=args.log_level, 
                         console=not args.json_only)
    
    try:
        # Validate input
        image_path = Path(args.image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
        
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            logger.error(f"Unsupported image format: {image_path.suffix}")
            sys.exit(1)
        
        # Initialize components
        processor = ImageProcessor()
        file_handler = FileHandler(args.output_dir)
        formatter = JSONFormatter()
        
        # Create output directory
        output_dir = file_handler.create_timestamped_output_dir("single_image")
        
        logger.info(f"Processing image: {image_path}")
        
        # Process image
        save_chips = not args.no_save_chips
        results = processor.process_image(
            image_path, 
            output_dir=output_dir if save_chips else None,
            save_chips=save_chips
        )
        
        if len(results) == 0:
            logger.warning("No faces detected in image")
            if args.json_only:
                print("[]")
            else:
                print("No faces detected")
            sys.exit(0)
        
        # Save results
        if not args.json_only:
            json_output_path = output_dir / f"{image_path.stem}_faces.json"
            formatter.save_to_file(results, json_output_path)
            
            # Create summary
            summary = formatter.create_summary_report(results)
            summary_path = output_dir / "summary.json"
            file_handler.save_json_output(summary, summary_path)
            
            # Print summary
            print(f"\n{'='*50}")
            print(f"Face Detection Results")
            print(f"{'='*50}")
            print(f"Image: {image_path.name}")
            print(f"Faces detected: {len(results)}")
            print(f"Output directory: {output_dir}")
            print(f"JSON output: {json_output_path}")
            
            if save_chips:
                chip_files = [r.get('chip_path') for r in results if 'chip_path' in r]
                print(f"Face chips saved: {len(chip_files)}")
            
            # Print face details
            for idx, face in enumerate(results):
                metadata = face['metadata']
                print(f"\nFace {idx + 1}:")
                print(f"  Confidence: {metadata['confidence']:.2%}")
                print(f"  Bounds: {metadata['face_bounds']}")
                print(f"  Identity: {metadata['identity']}")
                if 'gps' in metadata:
                    gps = metadata['gps']
                    print(f"  GPS: {gps['lat']:.6f}, {gps['lon']:.6f}")
        else:
            # JSON-only output
            print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()