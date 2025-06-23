#!/usr/bin/env python3
"""
Test face clustering accuracy to ensure no different faces are grouped together.
Creates timestamped output folders for easy tracking of improvements.
"""

import sys
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.processors.image_processor import ImageProcessor
from src.core.chip_generator import ChipGenerator


class ClusteringAccuracyTester:
    """Test clustering accuracy with timestamped outputs"""
    
    def __init__(self, base_output_dir="clustering_tests"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / f"test_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Test output directory: {self.output_dir}")
    
    def reset_clustering_registry(self):
        """Reset the clustering registry for a fresh test"""
        print("🔄 Resetting clustering registry...")
        
        registry_path = Path("data/cluster_registry.json")
        
        if registry_path.exists():
            # Backup to test directory
            backup_path = self.output_dir / "cluster_registry_backup.json"
            shutil.copy2(registry_path, backup_path)
            print(f"   Backed up existing registry to: {backup_path}")
        
        # Create fresh empty registry
        fresh_registry = {
            "clusters": {},
            "metadata": {
                "last_updated": "",
                "total_clusters": 0,
                "total_chips": 0
            }
        }
        
        registry_path.parent.mkdir(exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(fresh_registry, f, indent=2)
        
        print("✅ Registry reset complete")
    
    def test_image(self, image_path: Path):
        """Test clustering on a single image"""
        print(f"\n🖼️  Testing image: {image_path}")
        print("-" * 50)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            
            # Create Photos directory if it doesn't exist
            photos_dir = Path("data/input/Photos")
            photos_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📌 Please add your test image to: {photos_dir}")
            print("   Expected file: people-collage-design.jpg")
            return None
        
        try:
            # Reset registry for clean test
            self.reset_clustering_registry()
            
            # Process image with clustering
            processor = ImageProcessor(enable_clustering=True)
            result = processor.process_image(
                image_path=image_path,
                output_dir=str(self.output_dir / "chips"),
                save_chips=True
            )
            
            # Save raw results
            result_file = self.output_dir / "raw_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"📄 Raw results saved to: {result_file}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_clustering_results(self, result):
        """Analyze clustering results for accuracy"""
        if not result:
            return
        
        print("\n📊 Clustering Analysis")
        print("-" * 50)
        
        # Extract statistics
        stats = result.get('metadata', {}).get('processing_stats', {})
        faces_detected = stats.get('faces_detected', 0)
        clusters_assigned = stats.get('clusters_assigned', 0)
        
        print(f"Faces detected: {faces_detected}")
        print(f"Clusters created: {clusters_assigned}")
        
        # Analyze cluster distribution
        chips = result.get('metadata', {}).get('chips', [])
        cluster_info = defaultdict(list)
        
        for chip in chips:
            cluster_id = chip.get('clusterId', 'unknown')
            cluster_info[cluster_id].append({
                'bbox': chip.get('bbox'),
                'confidence': chip.get('confidence'),
                'chipPath': chip.get('chipPath')
            })
        
        print(f"\n📋 Cluster Distribution:")
        for cluster_id, faces in sorted(cluster_info.items()):
            print(f"\n  Cluster '{cluster_id}': {len(faces)} faces")
            for i, face in enumerate(faces):
                print(f"    Face {i+1}: confidence={face['confidence']:.3f}")
        
        # Generate analysis report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'faces_detected': faces_detected,
                'clusters_created': clusters_assigned,
                'cluster_distribution': {
                    cluster_id: len(faces) 
                    for cluster_id, faces in cluster_info.items()
                }
            },
            'cluster_details': dict(cluster_info),
            'potential_issues': []
        }
        
        # Check for potential issues
        if clusters_assigned == 1 and faces_detected > 5:
            report['potential_issues'].append(
                "All faces grouped in single cluster - may need more aggressive separation"
            )
        
        if clusters_assigned > faces_detected * 0.8:
            report['potential_issues'].append(
                "Too many clusters - possibly over-separating similar faces"
            )
        
        # Save analysis report
        report_file = self.output_dir / "clustering_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Analysis report saved to: {report_file}")
        
        return report
    
    def create_visual_report(self, image_path: Path, result):
        """Create visual report showing clustering results"""
        if not result or not image_path.exists():
            return
        
        print("\n🎨 Creating visual report...")
        
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        ax.set_title(f"Clustering Results - {image_path.name}")
        
        # Get cluster colors
        chips = result.get('metadata', {}).get('chips', [])
        unique_clusters = list(set(chip.get('clusterId', 'unknown') for chip in chips))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {cluster: color for cluster, color in zip(unique_clusters, colors)}
        
        # Draw bounding boxes with cluster colors
        for chip in chips:
            bbox = chip.get('bbox', {})
            if bbox:
                x = bbox.get('x', 0)
                y = bbox.get('y', 0)
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
                cluster_id = chip.get('clusterId', 'unknown')
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=2,
                    edgecolor=cluster_colors[cluster_id],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add cluster label
                ax.text(
                    x, y - 5, f"{cluster_id}",
                    color=cluster_colors[cluster_id],
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                )
        
        # Add legend
        legend_elements = [
            patches.Patch(color=color, label=f'Cluster {cluster}')
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save visual report
        visual_report_path = self.output_dir / "clustering_visual_report.png"
        plt.savefig(visual_report_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📄 Visual report saved to: {visual_report_path}")
    
    def create_summary_report(self):
        """Create a summary report for easy access"""
        summary = {
            'test_timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'files_generated': [],
            'quick_access': {}
        }
        
        # List all generated files
        for file_path in self.output_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.output_dir)
                summary['files_generated'].append(str(relative_path))
                
                # Add quick access paths for important files
                if file_path.name == 'clustering_analysis.json':
                    summary['quick_access']['analysis'] = str(file_path)
                elif file_path.name == 'clustering_visual_report.png':
                    summary['quick_access']['visual_report'] = str(file_path)
                elif file_path.name == 'raw_result.json':
                    summary['quick_access']['raw_data'] = str(file_path)
        
        # Save summary
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Test Summary")
        print("-" * 50)
        print(f"Output directory: {self.output_dir}")
        print(f"Files generated: {len(summary['files_generated'])}")
        print("\nQuick access:")
        for key, path in summary['quick_access'].items():
            print(f"  {key}: {path}")
        
        # Create a latest symlink for easy access
        latest_link = self.base_output_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(self.output_dir.name)
        print(f"\n🔗 Latest test: {latest_link} -> {self.output_dir.name}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test face clustering accuracy with timestamped outputs"
    )
    parser.add_argument(
        "--image",
        default="data/input/Photos/people-collage-design.jpg",
        help="Path to test image (default: data/input/Photos/people-collage-design.jpg)"
    )
    parser.add_argument(
        "--output-dir",
        default="clustering_tests",
        help="Base directory for test outputs (default: clustering_tests)"
    )
    
    args = parser.parse_args()
    
    print("🧪 Face Clustering Accuracy Test")
    print("=" * 50)
    
    # Create tester
    tester = ClusteringAccuracyTester(base_output_dir=args.output_dir)
    
    # Test image
    image_path = Path(args.image)
    result = tester.test_image(image_path)
    
    if result:
        # Analyze results
        analysis = tester.analyze_clustering_results(result)
        
        # Create visual report
        tester.create_visual_report(image_path, result)
        
        # Create summary
        tester.create_summary_report()
        
        print("\n✅ Test completed successfully!")
        print(f"📁 Results available in: {tester.output_dir}")
        print(f"   - Analysis: clustering_analysis.json")
        print(f"   - Visual report: clustering_visual_report.png")
        print(f"   - Face chips: chips/")
    else:
        print("\n❌ Test failed - please check the error messages above")


if __name__ == "__main__":
    main()