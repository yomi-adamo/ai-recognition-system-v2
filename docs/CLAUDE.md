## Facial Vision System Design & Claude Code Instructions
🔍 Project Overview
Facial Vision is a backend system for detecting, cropping, and clustering faces from images or videos (e.g., bodycam footage). Rather than identifying individuals by name, this system groups similar face chips together using unsupervised clustering (e.g., person_1, person_2, etc.). Each cluster of chips corresponds to a visually consistent face and includes associated metadata.

All chips and metadata are registered with the provenance system via FireFly and linked to the original file asset (image/video) using the parentId field, making each chip a derived asset in a blockchain-based asset tree.

## Artifacts

- [clustering_plan.md](./clustering_plan.md): Full modular plan for facial clustering architecture and blockchain integration.
- [blockchain_integration_plan.md](./blockchain_integration_plan.md): Detailed implementation strategy for registering facial vision results to the blockchain using Maverix and FireFly.


## Project Structure

facial-vision/
├── docs/
│   ├── CLAUDE.md               # Claude Code instructions and prompts
│   ├── specs.md                # Technical specifications
│   ├── architecture.md         # System architecture design
│   ├── tasks.md                # Task breakdown
│   └── api.md                  # API documentation
├── src/
│   ├── core/
│   │   ├── face_detector.py        # Face detection logic
│   │   ├── face_clusterer.py       # NEW: Clustering via embeddings
│   │   ├── metadata_extractor.py   # Timestamp, GPS, etc.
│   │   └── chip_generator.py       # Save cropped faces
│   ├── processors/
│   │   ├── image_processor.py
│   │   ├── video_processor.py
│   │   └── batch_processor.py
│   ├── outputs/
│   │   ├── json_formatter.py
│   │   ├── ipfs_uploader.py
│   │   └── blockchain_logger.py
│   └── utils/
│       ├── file_handler.py
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_clustering.py          # NEW: Clustering test suite
│   ├── test_face_detection.py
│   ├── test_video_processing.py
│   └── test_output_format.py
├── config/
├── data/
│   ├── input/
│   ├── output/
│   │   ├── person_1/
│   │   ├── person_2/
│   └── models/
├── scripts/
├── bugs/
├── requirements.txt
└── README.md

## Key Features to Support

### Face Clustering
- Extract face embeddings (from face_recognition, MTCNN, or DeepFace)
- Cluster with cosine similarity using DBSCAN/HDBSCAN
- Assign folders per person cluster: person_1/, person_2/, etc.
- No manual identity labels needed
- Store chips + JSON metadata per image

## ⚙️ Key Design Updates

### 🔗 Blockchain Provenance Flow

1. When a new file is received:
   - Check if it is already registered on the blockchain via `maverix`/`maverix-demo` query pattern.
   - If not present, create a new asset using the provenance API.

2. Process the file (face detection, clustering, etc.).

3. Generate final metadata and update a local JSON.

4. Submit a derived asset using the updated JSON to the provenance service via FireFly.

> ✅ Important: Metadata updates to local JSON **will not change** the on-chain version unless submitted as a new derived asset.

### 👥 Face Clustering Design

- Extract embeddings from each detected face.
- Cluster embeddings based on cosine similarity (e.g., using DBSCAN or HDBSCAN).
- Assign chips from the same cluster into folders like `person_1`, `person_2`, etc.
- Identity labels are **not** required.
- Each chip contains metadata for:
  - file: path to chip
  - Timestamp
  - Time of video (UTC) (time in reference to when the video was made)
  - GPS (if available)
  - Parent ID (original asset ID)
  - device ID
  - Source file
  - Bounding box
  - Cluster ID
  - Topics like face_detected etc.


### 🧩 Modular Extensions (Planned)
These extensions will follow the same detection → chip → metadata → blockchain flow.

Tattoo Detection: tattoo_detector.py

License Plate OCR: license_plate_reader.py

Vehicle Classification: vehicle_detector.py

Voice-to-Text Transcription: voice_transcriber.py

---



### 🔍 Face Clustering Setup
> "Implement a face clustering module that groups detected face chips into folders like person_1, person_2, etc. Use DBSCAN on face encodings with adjustable similarity threshold."

### 🔗 Provenance Integration Planning
> Can you look through maverix and walk me through how I can:  
>  1. Check if a file is already on the blockchain using `maverix/maverix-demo` or/and maverix/maverix  
>  2. Upload a new asset if not found  
>  3. Add a derived asset with additional metadata  

Asset Lookup:

Query Maverix to check if the raw video/image is already registered.

Root Asset Registration:

If not present, register it via /assets endpoint.

Derived Assets (chips):

Store each chip on IPFS and submit as a derived asset.

Include parentId pointing to root asset ID.

Tree Structure:

Forms a clear parent–child relationship between original video and its face chips.

>  Return a plan and test suite, but don't code yet."

---

## 🧪 JSON Output Example

~~~json
{
  "file": "video.mp4",
  "type": "video",
  "name": "Security FOotage 2024-01-15",
  "author": "author",
  "timestamp": "2024-01-15T13:24:00Z",
  "parentId": "original-asset-id",
  "metadata": {
    "GPS": {
      "lat": 39.2557,
      "lon": -76.7112
    },
    "chips": [
      {
        "file": "person_1/chip_001.jpg",
        "type": "image",
        "name": "chip_001",
        "author": "author",
        "timestamp": "2024-01-15T13:24:00Z",
        "deviceId": "AXIS-W120",
        "clusterId": "person_1",
        "face_bounds": {
          "x": 84,
          "y": 122,
          "w": 64,
          "h": 64
        },
        "topics": ["face_detected"]
      },
      {
        "file": "person_1/chip_002.jpg",
        "type": "image",
        "name": "chip_002",
        "author": "author",
        "timestamp": "2024-01-15T13:24:02Z",
        "deviceId": "AXIS-W120",
        "clusterId": "person_1",
        "face_bounds": {
          "x": 81,
          "y": 120,
          "w": 62,
          "h": 63
        },
        "topics": ["face_detected"]
      },
      {
        "file": "person_2/chip_003.jpg",
        "type": "image",
        "name": "chip_003",
        "author": "author",
        "timestamp": "2024-01-15T13:24:02Z",
        "deviceId": "AXIS-W120",
        "clusterId": "person_2",
        "face_bounds": {
          "x": 91,
          "y": 118,
          "w": 60,
          "h": 60
        },
        "topics": ["face_detected"]
      }
    ]
  }, 
  "topics": ["face_detected", "video_analysis"]
}
~~~

Example of adding to the blockchain, All of the chips would be in the metadata section of the JSON for the derived asset.:
~~~bash
curl -X POST http://localhost:3000/provenance/assets \
  -F "file=@video.mp4" \
  -F "type=video" \
  -F "name=Security Footage 2024-01-15" \
  -F "author=camera-01" \
  -F "parentId=original-asset-id" \
  -F 'metadata={
        "GPS": {
          "lat": 39.2557,
          "lon": -76.7112
        },
        "chips": [
          {
            "file": "person_1/chip_001.jpg",
            "type": "image",
            "name": "chip_001",
            "author": "author",
            "timestamp": "2024-01-15T13:24:00Z",
            "deviceId": "AXIS-W120",
            "clusterId": "person_1",
            "face_bounds": {
              "x": 84,
              "y": 122,
              "w": 64,
              "h": 64
            },
            "topics": ["face_detected"]
          },
          {
            "file": "person_1/chip_002.jpg",
            "type": "image",
            "name": "chip_002",
            "author": "author",
            "timestamp": "2024-01-15T13:24:02Z",
            "deviceId": "AXIS-W120",
            "clusterId": "person_1",
            "face_bounds": {
              "x": 81,
              "y": 120,
              "w": 62,
              "h": 63
            },
            "topics": ["face_detected"]
          },
          {
            "file": "person_2/chip_003.jpg",
            "type": "image",
            "name": "chip_003",
            "author": "author",
            "timestamp": "2024-01-15T13:24:02Z",
            "deviceId": "AXIS-W120",
            "clusterId": "person_2",
            "face_bounds": {
              "x": 91,
              "y": 118,
              "w": 60,
              "h": 60
            },
            "topics": ["face_detected"]
          }
        ]  
      } 
    }' \
  -F 'topics=["face_detected", "vidoe_analysis"]'
~~~

### 🧠

I want to change the design of my facial recognition pipeline. Please do not write any code yet — I need you to come up with a new step-by-step implementation plan I can review first.

🔄 Context:
My project is located in `facial-vision/`, next to `maverix/`. Those use FireFly for blockchain provenance. I already have a working face detector and chip generator using OpenCV, MTCNN, and face_recognition.

🎯 New Design Goal:
I want to group face chips by similarity, not by identity.

For example:
- If a person is seen 10 times, I want all 10 chips saved in `person_1/`
- Another person’s 5 chips should go in `person_2/`
- No manual identity labeling (like names) is required
- Each chip will include metadata like timestamp, GPS, bounding box, source file, cluster ID, etc.

🧠 Claude Task:
Help me plan a complete redesign of the clustering system:
- Extract face embeddings and cluster them using unsupervised techniques (e.g., DBSCAN)
- Design how to organize and name folders per cluster (e.g., `person_1`, `person_2`)
- Define metadata structure per chip (timestamp, bounds, etc.)
- Suggest how this should be modularized in `src/core/`
- Propose a test suite to verify clustering and chip storage
- Explain how this will integrate with the blockchain via FireFly, using `maverix/maverix` and `maverix/maverix-demo`
- Emphasize asset tree design (original → chip → cluster)

Do NOT write code yet — just return a complete plan with pros/cons and alternatives where helpful.

## GPS Coordinate extraction
In facial-vision there needs to be a change. I have noticed that on the AXIS W120 body camera, when I export the mp4 file, the GPS   │
│   data isnt embed into the file, but they provide notes that contain the coordinates, teh notes file I am referring to is              │
│   facial-vision/data/input/Videos/location_test_metadata.txt which is the metadata for location_test.mp4. I dont want you to directly  │
│   take the coordinates from the txt file and add it to the chip metadata because we dont know what from the coordinates are            │
│   associated with, so this is my plan. There is overlay on the location_test.mp4 and the coordinates are toward the bottom right of    │
│   the screen, so when a face is detected I want the GPS coordinates that are in the bottom left to be read and I want that to be put   │
│   as as the gps metadata for that chip. Once this is fully functional i want you to commit this to my github.  

## Commit significant changes
Once you verify that a change you have made works, you should commit the changes to my github https://github.com/yomi-adamo/ai-recognition-system-v2.