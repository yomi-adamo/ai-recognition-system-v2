# Blockchain Integration Test Commands

## Prerequisites

### 1. Start Maverix Services
```bash
# Terminal 1: Start FireFly and dependencies
cd ~/code/maverix/maverix
docker-compose up
```

Wait for services to start (about 30-60 seconds). You should see:
- FireFly running on http://localhost:5000
- Maverix API running on http://localhost:3000

### 2. Verify Services are Running
```bash
# Check Maverix API
curl http://localhost:3000/provenance/assets?limit=1

# Check FireFly (optional)
curl http://localhost:5000/api/v1/status
```

## Running Tests

### 1. Unit Tests (No Services Required)
```bash
cd ~/code/facial-vision

# Activate virtual environment
source venv/bin/activate

# Install required dependencies if not already installed
pip install aiohttp pytest pytest-asyncio pytest-cov

# Run blockchain integration unit tests
python -m pytest tests/test_blockchain_integration.py -v

# Run with coverage
python -m pytest tests/test_blockchain_integration.py -v --cov=src.core.blockchain_integration
```

### 2. Live Integration Tests (Requires Maverix Running)
```bash
cd ~/code/facial-vision

# Activate virtual environment
source venv/bin/activate

# Make test script executable
chmod +x test_blockchain_integration_live.py

# Run the live test script
python test_blockchain_integration_live.py
```

This will test:
- ✅ Connection to Maverix
- ✅ Asset existence checking
- ✅ Image upload to blockchain
- ✅ Video upload to blockchain
- ✅ Face analysis results upload
- ✅ Provenance chain retrieval
- ✅ Asset download from IPFS

### 3. Test with Real Face Detection Pipeline
```bash
cd ~/code/facial-vision

# Activate virtual environment
source venv/bin/activate

# Process an image and upload to blockchain
python scripts/process_image.py data/input/Photos/group.jpg --blockchain

# Process a video and upload to blockchain  
python scripts/process_video.py data/input/test_stroll.mp4 --blockchain

# Process batch with blockchain
python scripts/process_folder.py data/input/Photos --blockchain
```

### 4. Manual API Testing with curl

#### Upload an Image Asset
```bash
curl -X POST http://localhost:3000/provenance/assets \
  -F "file=@data/input/Photos/yomi1.jpg" \
  -F "type=image" \
  -F "name=Test Portrait" \
  -F "author=test-user" \
  -F 'metadata={"deviceId":"CAM-001","location":"Test Lab"}' \
  -F 'topics=["test","portrait"]'
```

#### Query Assets
```bash
# Get all assets
curl http://localhost:3000/provenance/assets

# Filter by type
curl http://localhost:3000/provenance/assets?type=image

# Filter by author
curl http://localhost:3000/provenance/assets?author=test-user
```

#### Get Asset Details
```bash
# Replace ASSET_ID with actual ID from upload response
curl http://localhost:3000/provenance/assets/ASSET_ID
```

#### Get Provenance Chain
```bash
curl http://localhost:3000/provenance/assets/ASSET_ID/provenance
```

#### Download Asset File
```bash
curl -o downloaded_file.jpg http://localhost:3000/provenance/assets/ASSET_ID/download
```

## Integration with Face Detection Pipeline

### 1. Create a Script to Process and Upload
```bash
cd ~/code/facial-vision
cat > process_with_blockchain.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import sys
sys.path.insert(0, 'src')

from processors.image_processor import ImageProcessor
from core.blockchain_integration import BlockchainIntegration

async def process_and_upload(image_path):
    # Process image
    processor = ImageProcessor()
    result = processor.process(image_path)
    
    # Upload to blockchain
    async with BlockchainIntegration() as blockchain:
        # Upload original
        original = await blockchain.upload_original_asset(
            file_path=image_path,
            asset_type="image",
            name=f"Processed - {image_path}",
            author="facial-vision",
            metadata=result['metadata']
        )
        
        # Upload analysis
        analysis = await blockchain.upload_face_analysis_results(
            parent_asset_id=original['assetId'],
            face_chips_metadata=result['faces'],
            analysis_metadata={
                "sourceFile": image_path,
                "totalFaces": len(result['faces'])
            }
        )
        
        print(f"Original Asset: {original['assetId']}")
        print(f"Analysis Asset: {analysis['assetId']}")

if __name__ == "__main__":
    asyncio.run(process_and_upload(sys.argv[1]))
EOF

chmod +x process_with_blockchain.py
```

### 2. Run the Integration
```bash
# Activate virtual environment first
source venv/bin/activate

# Run the integration
python process_with_blockchain.py data/input/Photos/group.jpg
```

## Troubleshooting

### If Maverix is not running:
```bash
# Check if containers are running
docker ps

# Check logs
cd ~/code/maverix/maverix
docker-compose logs -f

# Restart services
docker-compose down
docker-compose up
```

### If tests fail with connection errors:
1. Ensure Maverix is fully started (wait 60 seconds after docker-compose up)
2. Check if port 3000 is available: `lsof -i :3000`
3. Try accessing http://localhost:3000 in a browser

### If IPFS uploads fail:
1. Check FireFly logs: `docker-compose logs firefly_core`
2. Ensure IPFS is running: `docker-compose ps ipfs`
3. Check disk space: `df -h`

## Expected Output

When running `test_blockchain_integration_live.py`, you should see:

```
🚀 Starting Blockchain Integration Tests
==================================================

=== Test 1: Testing Connection to Maverix ===
✅ Successfully connected to Maverix
   Found 0 existing assets

=== Test 2: Testing Asset Existence Checks ===
📄 File: data/input/Photos/yomi1.jpg
   Hash: a1b2c3d4e5f6...
ℹ️  Asset not found by hash (expected for first run)
ℹ️  Asset not found by metadata (expected for first run)

=== Test 3: Testing Image Upload ===
📤 Uploading: data/input/Photos/yomi1.jpg
✅ Upload successful!
   Asset ID: 123e4567-e89b-12d3-a456-426614174000
   IPFS CID: QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco
   Topics: test, portrait, facial_vision...

[... more test output ...]
```