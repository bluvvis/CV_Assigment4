# Algorithm Guide

## Overall Architecture

The system generates cinematic videos from 3D Gaussian Splatting scenes through 6 stages:

---

## Stage 1: Scene Loading (PLY Loader)

**Concept:** Loading and parsing 3D Gaussian Splatting data

**Technologies:**

- PLY format (uncompressed)
- Binary .splat format (32-byte structure)

- Extracting point coordinates (x, y, z)

- Metadata: bounds, center, radius of scene

**Result:** Point array (N×3) and scene metadata

---

## Stage 2: Scene Exploration (Scene Explorer)

**Concept:** Generating multiple viewpoints for scene analysis

**Strategy:** Spherical Grid Sampling

- Spherical grid around scene center

- Multiple radius and height levels

- ~500 viewpoints

**Result:** JSON with viewpoint coordinates (position, lookAt)

---

## Stage 3: Space Analysis (Space Analyzer)

**Concept:** Finding free space and walkable paths for camera navigation

**Methods:**

1. Voxel grid

   - Dividing 3D space into cubes (voxel_size = 1.0-2.0m)

   - Counting point density in each voxel

2. Floor detection

   - Y-coordinate percentile (10th percentile)

3. Finding free space

   - Voxels with low density = free space

4. Extracting walkable paths

   - Cells at camera height (floor + 1.6–1.8m)

   - In free space

   - Surrounded by structure (walls/objects)

**Result:** List of walkable cells for path planning

**Simplified Mode:** For large scenes (>10M points), uses sampling-based analysis for efficiency

---

## Stage 4: Path Planning (Path Planner)

**Concept:** Generating smooth camera trajectories

**Path Types:**

1. Flythrough (for interiors)

   - A* on 2D XZ projection

   - Obstacle avoidance (walls)

   - Interpolation between waypoints

2. Coverage (for full scene coverage)

   - Grid-based region selection

   - Density filtering (only regions with sufficient points)

   - Greedy path construction through key regions

   - A* segments between regions

3. Orbital/Spiral (for exteriors)

   - Orbital movement around scene

   - Spiral ascents/descents

**Techniques:**

- Catmull-Rom splines for smoothness

- Easing functions (cubic ease-in/out)

- Collision checking via KD-tree

- Speed smoothing

**Result:** JSON with keyframes (position, lookAt, frame)

---

## Stage 5: Video Rendering (Video Renderer)

**Concept:** Capturing frames through browser and assembling video

**Technologies:**

1. Playwright — browser automation

   - Loading Three.js viewer

   - Camera control via JavaScript API

   - Frame screenshots (1920×1080)

2. FFmpeg / OpenCV — video encoding

   - H.264 (libx264)

   - CRF 18 (high quality)

   - 30 fps

**Process:**

- For each keyframe:

  1. Set camera position

  2. Wait for rendering

  3. Screenshot

  4. (Optional) Object detection

5. Encode all frames to MP4

---

## Stage 6: Object Detection (Object Detector) — integrated into rendering

**Concept:** Recognizing objects on frames during rendering

**Technologies:**

- YOLOv8 (Ultralytics)

- 2D detection on screenshots

- Bounding box visualization

**Filtering:**

1. Confidence threshold: 0.3

2. Allowed classes (only relevant for interiors):

   - `person`, `chair`, `couch`, `table`, `laptop`, `keyboard`, `mouse`, `cell phone`, `book`, `tv`, `monitor`, `clock`, `vase`, `bottle`, `cup`, `bowl`

   - Excluded: `umbrella`, `boat` and other inappropriate classes

**Process:**

- After each screenshot:

  1. YOLO detection

  2. Class filtering

  3. Drawing bounding boxes on frame

  4. Saving results to JSON

**Note:** YOLO is trained on real photos, so on 3D renders false positives are possible → class filtering helps

---

## Key Concepts

### 1. 3D Gaussian Splatting

- Representing scene as point cloud with color and transparency

- Rendering via Three.js in browser

### 2. Voxel Grid

- Dividing 3D space into cubes

- Density analysis to find free space

### 3. A* Algorithm

- Finding optimal path on 2D obstacle grid

- Avoiding walls and obstacles

### 4. Spline Interpolation

- Catmull-Rom for smooth trajectories

- Easing for natural camera movement

### 5. 2D Object Detection

- YOLO on screenshots instead of 3D analysis

- Easier to integrate, works in real-time

---

## Data Flow

```
PLY/Splat file 
  → Points (N×3) 
    → Exploration (viewpoints)
      → Space analysis (walkable cells)
        → Path planning (keyframes)
          → Rendering + Detection (frames + detections)
            → MP4 video + JSON reports
```

---

## Current Limitations and Solutions

1. Problem: YOLO works poorly on 3D renders

   - Solution: Class filtering + increased confidence threshold

2. Problem: Wall detection via XZ projection included floor/ceiling

   - Solution: Using slice at camera height (partially implemented)

3. Problem: False positives (umbrella, boat)

   - Solution: Whitelist of allowed classes for interiors

---

## Configuration

All parameters in `configs/config.yaml`:

- Video dimensions, FPS

- Path planning parameters

- Detection settings (threshold, allowed classes)

- Space analysis parameters

The system is modular: each component can be configured independently.

---

## Assignment Tasks (each = 5 points)

1. Render a video from inside the scene. (no need to be realistic)
2. Detect objects in the rendered video.
3. 3D object detection.
4. Path planning.
5. Obstacle avoidance.
6. Rendered video that covers most of the scene/area.
7. Render a 360° video.
8. An interactive demo.
9. Real-time preview of the scene or pipeline.
10. Produce artistic / professional / innovative / realistic result videos (high-quality rendering).

**Deadline:** December 2  
**Scoring:** 15 points total (10% weight + 5% bonus)  
- Each task = 5 points
- Maximum = 15 points (complete any 3 tasks perfectly)
