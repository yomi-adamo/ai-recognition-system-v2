#!/bin/bash
echo "Processing all photos in batch..."
source venv/bin/activate

for photo in data/input/Photos/*.jpg data/input/Photos/*.jpeg data/input/Photos/*.png; do
    if [ -f "$photo" ]; then
        echo "Processing: $photo"
        python scripts/process_image.py "$photo" --output-dir data/output/batch_test_photos_videos
        echo "Completed: $photo"
        echo "---"
    fi
done

echo "Batch processing complete!"