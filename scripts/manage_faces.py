#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face database management CLI
Usage: python scripts/manage_faces.py <command> [options]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.face_recognizer import FaceRecognizer
from src.utils.logger import setup_logger


def cmd_add(args):
    """Add a new face to the database"""
    recognizer = FaceRecognizer()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    print(f"Adding face: {args.name}")
    print(f"Image: {image_path}")
    
    success = recognizer.add_known_face(args.name, image_path)
    
    if success:
        print(f"✓ Successfully added '{args.name}' to the database")
        recognizer.save_database()
        
        # Show updated stats
        stats = recognizer.get_statistics()
        print(f"Database now contains {stats['total_identities']} identities")
        return True
    else:
        print(f"✗ Failed to add '{args.name}'")
        return False


def cmd_remove(args):
    """Remove a face from the database"""
    recognizer = FaceRecognizer()
    
    if args.name not in recognizer:
        print(f"Error: '{args.name}' not found in database")
        return False
    
    print(f"Removing face: {args.name}")
    
    success = recognizer.remove_face(args.name)
    
    if success:
        print(f"✓ Successfully removed '{args.name}' from the database")
        recognizer.save_database()
        
        # Show updated stats
        stats = recognizer.get_statistics()
        print(f"Database now contains {stats['total_identities']} identities")
        return True
    else:
        print(f"✗ Failed to remove '{args.name}'")
        return False


def cmd_list(args):
    """List all known faces"""
    recognizer = FaceRecognizer()
    
    faces = recognizer.list_known_faces()
    
    if len(faces) == 0:
        print("No faces in database")
        return True
    
    print(f"Known faces ({len(faces)} total):")
    print("-" * 80)
    
    for face in faces:
        print(f"Name: {face['name']}")
        print(f"  Encodings: {face['encoding_count']}")
        print(f"  Added: {face['date_added'][:19]}")  # Remove microseconds
        print(f"  Last seen: {face['last_seen'][:19]}")
        
        if 'image_paths' in face and face['image_paths']:
            print(f"  Images: {len(face['image_paths'])}")
            if args.verbose:
                for img_path in face['image_paths'][:3]:  # Show first 3
                    print(f"    - {img_path}")
                if len(face['image_paths']) > 3:
                    print(f"    ... and {len(face['image_paths']) - 3} more")
        print()
    
    return True


def cmd_update(args):
    """Update a face with new image"""
    recognizer = FaceRecognizer()
    
    if args.name not in recognizer:
        print(f"Error: '{args.name}' not found in database")
        return False
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    print(f"Updating face: {args.name}")
    print(f"New image: {image_path}")
    
    success = recognizer.add_known_face(args.name, image_path)
    
    if success:
        print(f"✓ Successfully updated '{args.name}' with new image")
        recognizer.save_database()
        
        # Show updated info
        faces = recognizer.list_known_faces()
        face_info = next((f for f in faces if f['name'] == args.name), None)
        if face_info:
            print(f"'{args.name}' now has {face_info['encoding_count']} encodings")
        return True
    else:
        print(f"✗ Failed to update '{args.name}'")
        return False


def cmd_export(args):
    """Export database to JSON"""
    recognizer = FaceRecognizer()
    
    output_path = Path(args.output_path)
    
    print(f"Exporting database to: {output_path}")
    
    success = recognizer.export_database_json(output_path)
    
    if success:
        print(f"✓ Successfully exported database metadata")
        
        # Show export stats
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        print(f"Exported {data['total_identities']} identities")
        print(f"Total recognitions performed: {data['stats']['total_recognitions']}")
        return True
    else:
        print(f"✗ Failed to export database")
        return False


def cmd_import(args):
    """Import faces from a folder"""
    recognizer = FaceRecognizer()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return False
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        if args.recursive:
            image_files.extend(folder_path.rglob(f"*{ext}"))
    
    if len(image_files) == 0:
        print(f"No image files found in {folder_path}")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Group by name (use filename without extension as identity)
    faces_dict = {}
    for img_file in image_files:
        # Use parent directory name as identity if in subdirectories
        if args.use_folders:
            name = img_file.parent.name
        else:
            name = img_file.stem
        
        if name not in faces_dict:
            faces_dict[name] = []
        faces_dict[name].append(img_file)
    
    print(f"Importing {len(faces_dict)} identities...")
    
    # Import faces
    results = recognizer.add_multiple_faces(faces_dict)
    
    # Save database
    recognizer.save_database()
    
    # Show results
    successful = sum(results.values())
    failed = len(results) - successful
    
    print(f"✓ Successfully imported: {successful} identities")
    if failed > 0:
        print(f"✗ Failed to import: {failed} identities")
        
        # Show failed identities
        failed_names = [name for name, success in results.items() if not success]
        print("Failed identities:", ", ".join(failed_names[:5]))
        if len(failed_names) > 5:
            print(f"... and {len(failed_names) - 5} more")
    
    return successful > 0


def cmd_stats(args):
    """Show recognition statistics"""
    recognizer = FaceRecognizer()
    
    stats = recognizer.get_statistics()
    
    print("Face Recognition Statistics")
    print("=" * 50)
    print(f"Total identities: {stats['total_identities']}")
    print(f"Total encodings: {stats['total_encodings']}")
    print(f"Avg encodings per identity: {stats['avg_encodings_per_identity']:.1f}")
    print(f"Similarity threshold: {stats['similarity_threshold']}")
    print()
    print("Recognition Performance:")
    print(f"Total recognitions: {stats['total_recognitions']}")
    print(f"Successful matches: {stats['successful_matches']}")
    print(f"Unknown faces: {stats['unknown_faces']}")
    print(f"Recognition accuracy: {stats['recognition_accuracy']:.1%}")
    print()
    print(f"Database path: {stats['database_path']}")
    
    # Show top identities by encoding count
    faces = recognizer.list_known_faces()
    if faces:
        print("\nTop identities by encoding count:")
        top_faces = sorted(faces, key=lambda x: x['encoding_count'], reverse=True)[:5]
        for face in top_faces:
            print(f"  {face['name']}: {face['encoding_count']} encodings")
    
    return True


def cmd_search(args):
    """Search for faces in an image"""
    recognizer = FaceRecognizer()
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    print(f"Searching for faces in: {image_path}")
    
    results = recognizer.recognize_faces_in_image(image_path)
    
    if len(results) == 0:
        print("No faces found in image")
        return True
    
    print(f"Found {len(results)} faces:")
    print("-" * 50)
    
    for i, result in enumerate(results):
        print(f"Face {i + 1}:")
        print(f"  Identity: {result['name']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        
        if 'location' in result:
            top, right, bottom, left = result['location']
            print(f"  Location: ({left}, {top}) to ({right}, {bottom})")
        
        if 'all_candidates' in result:
            candidates = result['all_candidates'][:3]  # Top 3
            print(f"  Top candidates:")
            for candidate in candidates:
                print(f"    - {candidate['name']}: {candidate['confidence']:.1%}")
        print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Face database management')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add new face to database')
    add_parser.add_argument('name', help='Identity name')
    add_parser.add_argument('image_path', help='Path to image file')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove face from database')
    remove_parser.add_argument('name', help='Identity name to remove')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all known faces')
    list_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Show detailed information')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update face with new image')
    update_parser.add_argument('name', help='Identity name')
    update_parser.add_argument('image_path', help='Path to new image file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export database to JSON')
    export_parser.add_argument('output_path', help='Output JSON file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import faces from folder')
    import_parser.add_argument('folder_path', help='Folder containing images')
    import_parser.add_argument('--recursive', '-r', action='store_true',
                             help='Search recursively')
    import_parser.add_argument('--use-folders', action='store_true',
                             help='Use folder names as identities')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show recognition statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for faces in image')
    search_parser.add_argument('image_path', help='Path to image file')
    
    # Global options
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logger("manage_faces", level=args.log_level, console=True)
    
    # Execute command
    try:
        command_map = {
            'add': cmd_add,
            'remove': cmd_remove,
            'list': cmd_list,
            'update': cmd_update,
            'export': cmd_export,
            'import': cmd_import,
            'stats': cmd_stats,
            'search': cmd_search
        }
        
        success = command_map[args.command](args)
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()