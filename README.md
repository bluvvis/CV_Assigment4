# 🎬 3D Gaussian Splatting Video Generator

A system for generating cinematic videos from 3D Gaussian Splatting scenes with automatic object detection and intelligent camera path planning.

## 📋 Table of Contents

- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Algorithm Descriptions](#algorithm-descriptions)
- [Dependencies and Requirements](#dependencies-and-requirements)
- [Known Limitations](#known-limitations)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (recommended) for YOLO acceleration
- ~10-20 GB free disk space for output files

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Playwright Browser

```bash
# Windows
python -m playwright install chromium

# Linux/Mac
python -m playwright install chromium
```

### Step 3: Download YOLO Model (Automatic)

The YOLOv8 model (`yolov8n.pt`) will be automatically downloaded on first run. Alternatively, you can download it manually from [Ultralytics](https://github.com/ultralytics/ultralytics).

### Step 4: Verify Installation

```bash
python main.py --scene Theater.splat --task path_planning
```

If the command runs without errors, installation is complete.

---

## Usage Guide

### Basic Usage

```bash
python main.py --scene <scene_file.splat> --task <task_name>
```

### Available Tasks

#### 1. Render Video from Inside Scene (5 points)
```bash
python main.py --scene Theater.splat --task video
```
**Output:** `output/videos/Theater_video.mp4` (300 frames, ~10 seconds)

#### 2. Render 360° Orbital Video (5 points)
```bash
python main.py --scene Theater.splat --task 360
```
**Output:** `output/videos/Theater_360.mp4` (360 frames, ~12 seconds)

#### 3. Path Planning (5 points)
```bash
python main.py --scene Theater.splat --task path_planning
```
**Output:** `output/keyframes.json` with planned camera path

#### 4. Coverage Video (5 points)
```bash
python main.py --scene Theater.splat --task coverage
```
**Output:** `output/videos/Theater_coverage.mp4` (500 frames, ~16 seconds)

#### 5. Object Detection
```bash
python main.py --scene Theater.splat --task detect
```
**Output:** `output/reports/Theater_detections.json`

### Command Line Arguments

- `--scene <file>`: Path to .splat scene file (required)
- `--task <name>`: Task to execute (required)
  - Options: `video`, `360`, `path_planning`, `coverage`, `detect`, `obstacle_avoidance`, `interactive`, `realtime`, `artistic`
- `--config <file>`: Path to config file (optional, defaults to `configs/config.yaml`)

### Examples

```bash
# Render a basic video
python main.py --scene Theater.splat --task video

# Generate 360° orbital video
python main.py --scene Theater.splat --task 360

# Plan a path through the scene
python main.py --scene Theater.splat --task path_planning

# Create coverage video (covers most of the scene)
python main.py --scene Theater.splat --task coverage
```

### Output Structure

All results are saved in the `output/` directory:

```
output/
├── videos/           # Rendered video files (.mp4)
├── frames/           # Individual frame images (.png)
├── reports/          # Object detection reports (.json)
├── keyframes.json    # Camera path keyframes
└── viewpoints.json   # Scene exploration viewpoints
```

---

## Algorithm Descriptions

The system processes 3D Gaussian Splatting scenes through 6 main stages:

### Stage 1: Scene Loading (PLY Loader)

**Purpose:** Load and parse 3D Gaussian Splatting data from binary `.splat` files.

**Algorithm:**
- Detects binary `.splat` format (32-byte structure per splat)
- Extracts point coordinates (x, y, z) from binary data
- Computes scene metadata: bounds, center, radius
- Handles large scenes (34+ million splats) efficiently

**Output:** Point cloud array (N×3) and scene metadata

### Stage 2: Scene Exploration (Scene Explorer)

**Purpose:** Generate candidate camera viewpoints for scene analysis.

**Algorithm:** Spherical Grid Sampling
- Creates spherical grid around scene center
- Multiple radius levels and elevation angles
- Generates ~500 viewpoints with position and lookAt vectors
- Used for preliminary scene understanding

**Output:** JSON file with viewpoint coordinates

### Stage 3: Space Analysis (Space Analyzer)

**Purpose:** Identify free space and walkable paths for camera navigation.

**Algorithm:**
- **Voxel Grid:** Divides 3D space into cubic cells (voxel_size = 1.0m)
- **Density Calculation:** Counts points per voxel to identify occupied vs. free space
- **Floor Detection:** Finds floor level using Y-coordinate histogram
- **Walkable Cells:** Extracts cells at camera height with low density
- **Simplified Mode:** For large scenes (>10M points), uses sampling-based analysis

**Output:** List of walkable cell positions

### Stage 4: Path Planning (Path Planner)

**Purpose:** Generate smooth camera trajectories through the scene.

**Algorithms:**

#### Flythrough Path (for interiors)
- **A* Algorithm:** 2D pathfinding on XZ projection
- **Walkable Filtering:** Only uses positions near scene points
- **Catmull-Rom Splines:** Smooth interpolation between waypoints
- **Scene Constraints:** Ensures path stays within scene bounds

#### Coverage Path (for full scene coverage)
- **Grid-Based Selection:** Divides scene into 3×3×2 grid regions
- **Density Filtering:** Only selects regions with sufficient point density
- **Greedy Path Construction:** Visits all key regions using nearest-neighbor heuristic
- **A* Segments:** Uses A* between regions for smooth transitions

#### Orbital Path (for 360° videos)
- **Spiral Motion:** Circular orbit with vertical spiral
- **Multiple Orbits:** Configurable number of full rotations
- **Center-Focused:** Camera always looks toward scene center

**Output:** List of keyframes with position and lookAt vectors

### Stage 5: Video Rendering (Video Renderer)

**Purpose:** Render frames using browser-based 3D visualization.

**Technology Stack:**
- **Playwright:** Browser automation (headless Chromium)
- **Three.js:** 3D graphics rendering in browser
- **Point Cloud Rendering:** Displays Gaussian splats as point cloud

**Process:**
1. Loads scene data into browser
2. For each keyframe: sets camera position and lookAt
3. Captures screenshot of rendered frame
4. Saves frames as PNG images

**Output:** Sequence of frame images

### Stage 6: Object Detection (Object Detector)

**Purpose:** Detect objects in rendered frames using YOLOv8.

**Algorithm:**
- **YOLOv8 Model:** Pre-trained object detection (COCO dataset)
- **GPU Acceleration:** Uses CUDA if available
- **Class Filtering:** Only detects allowed object classes
- **Bounding Box Visualization:** Draws boxes on detected objects

**Output:** JSON report with detections per frame

### Video Encoding

**Methods:**
1. **FFmpeg** (preferred): Fast, high-quality encoding
2. **OpenCV** (fallback): If FFmpeg not found, uses OpenCV VideoWriter

**Output:** MP4 video file (H.264 codec, 30 fps)

---

## Dependencies and Requirements

### Python Packages

See `requirements.txt` for complete list. Main dependencies:

- **numpy** (1.24+): Numerical operations, array handling
- **playwright** (1.40+): Browser automation
- **opencv-python** (4.8+): Video encoding (fallback)
- **ultralytics** (8.0+): YOLOv8 object detection
- **scipy** (1.11+): KD-tree for spatial queries
- **tqdm** (4.66+): Progress bars
- **pyyaml** (6.0+): Configuration file parsing

### System Requirements

- **OS:** Windows 10+, Linux, macOS
- **Python:** 3.8 or higher
- **RAM:** 8 GB minimum (16 GB recommended for large scenes)
- **GPU:** NVIDIA GPU with CUDA support (optional, for YOLO acceleration)
- **Disk Space:** 10-20 GB for output files
- **Browser:** Chromium (installed via Playwright)

### External Tools

- **FFmpeg** (optional): For video encoding. If not found, OpenCV is used as fallback.
- **YOLOv8 Model:** Automatically downloaded on first run (~6 MB)

### Installation Commands

```bash
# Install all Python dependencies
pip install -r requirements.txt

# Install Playwright browser
python -m playwright install chromium

# Verify CUDA (for GPU acceleration)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Known Limitations

### Performance Limitations

1. **Large Scenes:** Scenes with >10 million points use simplified analysis mode, which may be less accurate
2. **Rendering Speed:** Browser-based rendering is slower than native GPU rendering (~2-3 FPS)
3. **Memory Usage:** Large scenes (34+ million points) require significant RAM

### Technical Limitations

1. **Binary .splat Format:** Only supports 32-byte binary structure. Other formats may not work
2. **Point Cloud Rendering:** Gaussian splats are rendered as simple points, not true splat rendering
3. **FFmpeg Dependency:** Video encoding requires FFmpeg or falls back to slower OpenCV
4. **Browser Requirements:** Requires Chromium browser (installed via Playwright)

### Algorithm Limitations

1. **Path Planning:** A* algorithm uses 2D projection (XZ plane), may not handle complex 3D obstacles perfectly
2. **Space Analysis:** Voxel grid resolution is fixed (1.0m), may miss small details
3. **Coverage Path:** Grid-based selection may miss some areas in complex scenes
4. **Object Detection:** Limited to COCO dataset classes (80 classes)

### Platform Limitations

1. **Windows:** Some path handling may differ from Linux/Mac
2. **GPU:** CUDA acceleration requires compatible NVIDIA GPU and drivers
3. **Browser:** Headless mode may have rendering differences from visible browser

### Workarounds

- For very large scenes, reduce `voxel_size` in config or use simplified analysis
- For faster rendering, reduce video resolution in config
- For better path planning, increase `num_keyframes` in config
- For GPU issues, YOLO will fall back to CPU (slower but works)

---

## Project Structure

```
.
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── configs/
│   └── config.yaml        # System configuration
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── ply_loader.py      # Scene loading (.splat files)
│   ├── scene_explorer.py  # Viewpoint generation
│   ├── space_analyzer.py  # Space analysis (voxel grid)
│   ├── path_planner.py    # Path planning (A*, splines)
│   ├── video_renderer.py  # Video rendering (Playwright)
│   └── object_detector.py # Object detection (YOLO)
├── viewer/
│   └── index.html         # Three.js viewer for rendering
└── output/                 # Generated files (created automatically)
    ├── videos/            # Rendered videos
    ├── frames/            # Frame images
    ├── reports/           # Detection reports
    ├── keyframes.json    # Camera path
    └── viewpoints.json    # Viewpoints
```

---

## Configuration

All settings can be adjusted in `configs/config.yaml`:

- **Video settings:** Resolution, FPS, quality
- **Path planning:** Waypoint spacing, spline parameters
- **Space analysis:** Voxel size, density thresholds
- **Object detection:** Confidence threshold, allowed classes
- **Rendering:** Browser settings, screenshot delay

---

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**: System will use OpenCV fallback (slower but works)
2. **"Browser not found"**: Run `python -m playwright install chromium`
3. **"CUDA not available"**: YOLO will use CPU (slower but works)
4. **"Out of memory"**: Reduce scene size or use simplified analysis mode

### Getting Help

1. Check configuration in `configs/config.yaml`
2. Verify all dependencies are installed
3. Check that scene file exists and is valid
4. Review error messages for specific issues

---

## License

This project is created for educational purposes as part of Computer Vision course assignment.

---

## Additional Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Detailed technical report covering problem solving, algorithms, challenges, and future improvements
- **[ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md)** - Detailed algorithm descriptions for each stage

---

**For detailed technical information, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**
