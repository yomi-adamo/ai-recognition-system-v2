 ## Blockchain Integration Plan for Facial-Vision

  1. Checking if a File Already Exists on Blockchain

  Approach 1: Query by File Hash
  // Calculate file hash (SHA-256)
  const fileHash =
  crypto.createHash('sha256').update(fileBuffer).digest('hex');

  // Query blockchain using metadata search
  GET
  http://localhost:3000/provenance/assets?metadata.fileHash={fileHash}

  Approach 2: Query by Filename and Author
  GET http://localhost:3000/provenance/assets?name={filename}&author={d
  eviceId}

  Implementation Strategy:
  1. Before processing any video/image, compute its SHA-256 hash
  2. Store hash in asset metadata during initial upload
  3. Query by hash to check existence
  4. Cache results locally to avoid repeated queries

  2. Uploading New Root Assets

  For Original Video/Image:
  curl -X POST http://localhost:3000/provenance/assets \
    -F "file=@bodycam_footage.mp4" \
    -F "type=video" \
    -F "name=Bodycam Footage 2024-01-15 13:24:00" \
    -F "author=AXIS-W120" \
    -F 'metadata={
      "fileHash": "sha256-hash-here",
      "duration": "00:05:32",
      "resolution": "1920x1080",
      "fps": 30,
      "GPS": {"lat": 39.2557, "lon": -76.7112},
      "recordingStart": "2024-01-15T13:24:00Z",
      "deviceModel": "AXIS W120",
      "analysisStatus": "pending"
    }' \
    -F 'topics=["raw_footage", "bodycam", "2024-01-15"]'

  3. Adding Derived Assets (Face Chips)

  Strategy: Batch Upload Approach
  Instead of uploading each chip individually, create a comprehensive
  analysis result:

  curl -X POST http://localhost:3000/provenance/assets \
    -F "file=@analysis_result.json" \
    -F "type=face_analysis_result" \
    -F "name=Face Analysis - bodycam_footage.mp4" \
    -F "author=facial-vision-v1.0" \
    -F "parentId={original-video-asset-id}" \
    -F 'metadata={
      "sourceFile": "bodycam_footage.mp4",
      "analysisTimestamp": "2024-01-15T14:30:00Z",
      "facesDetected": 15,
      "clustersFound": 3,
      "processingTime": 45.2,
      "chipStorageLocation": "ipfs://QmXxx...",
      "clusters": {
        "person_1": {
          "chipCount": 10,
          "representative": "person_1/chip_001.jpg",
          "timeRange": ["13:24:00", "13:28:45"]
        },
        "person_2": {
          "chipCount": 5,
          "representative": "person_2/chip_003.jpg",
          "timeRange": ["13:24:02", "13:26:30"]
        }
      }
    }' \
    -F 'topics=["face_analysis", "derived", "clustered"]'

  4. Complete Integration Workflow

  class BlockchainIntegration:
      def __init__(self, maverix_url="http://localhost:3000"):
          self.base_url = maverix_url
          self.asset_cache = {}

      async def process_media_file(self, file_path):
          # Step 1: Check if already processed
          file_hash = self.calculate_file_hash(file_path)
          existing_asset = await self.check_existing_asset(file_hash)

          if existing_asset:
              # Check if analysis already exists
              derived_assets = await
  self.get_derived_assets(existing_asset['id'])
              if self.has_face_analysis(derived_assets):
                  return {"status": "already_processed", "assetId":
  existing_asset['id']}

          # Step 2: Upload original if needed
          if not existing_asset:
              asset_id = await self.upload_original_asset(file_path,
  file_hash)
          else:
              asset_id = existing_asset['id']

          # Step 3: Process faces
          analysis_result = await self.process_faces(file_path)

          # Step 4: Upload chips to IPFS (via FireFly)
          chips_ipfs_cid = await
  self.upload_chips_bundle(analysis_result['chips'])

          # Step 5: Create derived asset with analysis results
          derived_asset_id = await self.upload_analysis_result(
              asset_id,
              analysis_result,
              chips_ipfs_cid
          )

          return {
              "status": "processed",
              "originalAssetId": asset_id,
              "analysisAssetId": derived_asset_id,
              "clusters": analysis_result['clusters']
          }

  5. Asset Tree Structure

  Root: Original Video (bodycam_footage.mp4)
  │   Type: video
  │   ID: uuid-1234
  │   Metadata: {fileHash, GPS, deviceId, duration}
  │
  └── Derived: Face Analysis Result
      │   Type: face_analysis_result
      │   ID: uuid-5678
      │   ParentId: uuid-1234
      │   Metadata: {clusters, chipCount, ipfsCid}
      │
      └── Referenced: Face Chips Bundle (IPFS)
          │   Storage: IPFS via FireFly
          │   Structure:
          │   ├── person_1/
          │   │   ├── chip_001.jpg
          │   │   ├── chip_002.jpg
          │   │   └── metadata.json
          │   ├── person_2/
          │   │   ├── chip_003.jpg
          │   │   └── metadata.json
          │   └── analysis_summary.json

  6. Test Suite Design

  # test_blockchain_integration.py

  class TestBlockchainIntegration:

      def test_asset_existence_check(self):
          # Test checking for existing assets by hash
          # Test caching mechanism
          # Test handling of non-existent assets

      def test_original_asset_upload(self):
          # Test video upload with metadata
          # Test image upload with EXIF data
          # Test error handling for failed uploads
          # Test duplicate prevention

      def test_derived_asset_creation(self):
          # Test parent-child relationship
          # Test metadata inheritance
          # Test topics propagation
          # Test batch chip upload

      def test_ipfs_integration(self):
          # Test chip bundle creation
          # Test IPFS upload via FireFly
          # Test retrieval of chips
          # Test handling large bundles

      def test_provenance_chain(self):
          # Test querying full provenance
          # Test finding all derived assets
          # Test lineage tracking

      def test_error_recovery(self):
          # Test network failures
          # Test partial upload recovery
          # Test transaction rollback
          # Test retry mechanisms

  7. Implementation Considerations

  Performance Optimizations:
  - Batch chip uploads as tar.gz archives
  - Use async/concurrent processing
  - Implement local caching for asset lookups
  - Process video in chunks for memory efficiency

  Data Integrity:
  - Store file hashes for deduplication
  - Implement checksums for chip bundles
  - Validate parent-child relationships
  - Ensure atomic operations

  Scalability:
  - Design for incremental processing
  - Support resumable uploads
  - Implement pagination for large result sets
  - Consider sharding for massive datasets

  Error Handling:
  - Retry failed blockchain submissions
  - Queue system for resilience
  - Detailed logging for audit trail
  - Graceful degradation options

  This plan provides a complete integration strategy between
  facial-vision and the maverix blockchain system, ensuring proper
  provenance tracking while maintaining efficiency and scalability.