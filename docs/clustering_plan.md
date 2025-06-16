 ## Facial Clustering System Implementation Plan

  1. Core Module Architecture

  face_clusterer.py (New Module)

  - Purpose: Extract face embeddings and perform unsupervised
  clustering
  - Key Components:
    - FaceEmbeddingExtractor: Generate 128-dimensional face encodings
  using face_recognition library
    - ClusterManager: DBSCAN/HDBSCAN clustering with cosine similarity
  metric
    - ClusterRegistry: Track cluster assignments and manage person_X
  folder mappings
  - Flow:
    a. Receive detected face regions from face_detector
    b. Generate embeddings for each face
    c. Compare against existing cluster centroids
    d. Assign to existing cluster or create new one
    e. Return cluster_id (e.g., "person_1")

  chip_generator.py (Enhanced)

  - Purpose: Save cropped face images to cluster-specific directories
  - Enhancements:
    - Dynamic folder creation based on cluster_id
    - Unique chip naming: person_X/chip_YYY.jpg
    - Quality checks (blur detection, minimum resolution)
    - Maintain chip-to-source mapping
  - Output Structure:
  data/output/
  ├── person_1/
  │   ├── chip_001.jpg
  │   ├── chip_002.jpg
  ├── person_2/
  │   ├── chip_003.jpg

  metadata_extractor.py (Enhanced)

  - Purpose: Extract and structure comprehensive metadata
  - Data Sources:
    - Video metadata (creation time, duration, codec)
    - EXIF data from images
    - GPS coordinates (if available)
    - Device ID extraction from filename patterns
    - Frame timestamps for video sources
  - Output Format:
  {
    "file": "person_1/chip_001.jpg",
    "type": "image",
    "name": "chip_001",
    "author": "facial-vision-system",
    "timestamp": "2024-01-15T13:24:00Z",
    "deviceId": "AXIS-W120",
    "clusterId": "person_1",
    "parentId": "original-asset-blockchain-id",
    "sourceFile": "video.mp4",
    "frameNumber": 720,
    "videoTimestamp": "00:00:24.000",
    "face_bounds": {"x": 84, "y": 122, "w": 64, "h": 64},
    "confidence": 0.98,
    "topics": ["face_detected", "clustered"]
  }

  2. Processing Flow

  Input (Video/Image) → Face Detection → Face Embedding → Clustering →
  Chip Generation → Metadata Assembly → Blockchain Registration

  Detailed Steps:
  1. Input Processing: video_processor.py/image_processor.py loads
  media
  2. Face Detection: face_detector.py finds all faces with bounding
  boxes
  3. Embedding Generation: face_clusterer.py creates 128D vectors
  4. Clustering Decision:
    - Compare embedding to existing cluster centroids
    - If similarity > threshold (0.6), assign to cluster
    - Else create new cluster
  5. Chip Generation: Save cropped face to appropriate person_X folder
  6. Metadata Assembly: Combine all data points into structured JSON
  7. Blockchain Registration: Submit as derived asset via FireFly

  3. Clustering Strategy

  Algorithm: HDBSCAN (preferred) or DBSCAN
  - Metric: Cosine similarity (better for face embeddings than
  Euclidean)
  - Parameters:
    - min_cluster_size: 2 (minimum faces to form cluster)
    - min_samples: 1 (core point density)
    - cluster_selection_epsilon: 0.4 (similarity threshold)

  Cluster Management:
  - Maintain cluster_registry.json with:
  {
    "person_1": {
      "centroid": [0.123, -0.456, ...],
      "chip_count": 10,
      "last_seen": "2024-01-15T13:24:00Z",
      "representative_chips": ["chip_001.jpg", "chip_005.jpg"]
    }
  }
  - Incremental clustering for real-time processing
  - Periodic re-clustering for optimization

  4. Blockchain Integration

  Asset Tree Structure:
  Original Video/Image (Root Asset)
      └── Face Detection Result (Derived Asset)
          ├── person_1 chips collection
          ├── person_2 chips collection
          └── metadata.json

  Integration Flow:
  1. Check Existing Asset:
    - Query maverix/maverix-demo for file hash
    - If exists, retrieve parentId
  2. Register New Asset (if needed):
    - POST to /provenance/assets with original file
    - Store returned asset ID
  3. Submit Derived Asset:
    - Bundle all chips and metadata
    - Include parentId reference
    - POST complete clustering result

  5. Test Suite Design

  test_clustering.py:
  - Test embedding generation consistency
  - Verify cluster assignment logic
  - Test threshold sensitivity
  - Validate cluster persistence

  Key Test Scenarios:
  1. Same person across multiple frames → Same cluster
  2. Different people → Different clusters
  3. Poor quality faces → Rejection handling
  4. Cluster merging/splitting edge cases
  5. Metadata completeness validation
  6. Blockchain submission success/retry logic

  6. Advantages & Considerations

  Pros:
  - No manual labeling required
  - Scales automatically with new identities
  - Maintains complete provenance chain
  - Modular design allows feature extensions

  Cons & Mitigations:
  - Clustering accuracy depends on embedding quality → Use multiple
  face encodings
  - Cluster drift over time → Implement periodic re-clustering
  - Storage grows with chips → Implement retention policies

  Alternative Approaches:
  - ArcFace/FaceNet instead of face_recognition for better embeddings
  - Agglomerative Clustering for hierarchical person grouping
  - Online clustering algorithms for streaming video processing